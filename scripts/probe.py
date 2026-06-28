"""Probe abliterated model behavior by sweeping factor_a / factor_b.

Loads the original model, applies ablation in-place with the given recipe,
runs harmful prompts, and reports compliance rate. Cancels generation after
5 tokens if a refusal opening is detected.

Usage:
  uv run python scripts/probe.py \
    --model os.path.join(os.path.expanduser("~"),"models")/Qwen3.6-27B \
    --run-id run_2026-05-17T19-57-17-480Z_8696f34a \
    --onset 29 --split 38 \
    --factor-a 0.93 --factor-b 0.265 \
    [--samples 10]

  # Sweep mode — omit --factor-a/b to sweep a grid:
  uv run python scripts/probe.py --model ... --run-id ... --onset 29 --split 38 --sweep
"""

import argparse
import gc
import json
import re
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
import threading

from backend.inference.ablation import apply_ablation_in_place, restore_model_weights
from backend.inference.recipe import build_recipe
from backend.inference.run_loader import rebuild_directions

RUNS_DIR = _root / "data" / "runs"
DEVICE   = "cuda:0"

REFUSAL_RE = re.compile(
  r"^(I('m| am) (sorry|unable|not able)|I can'?t|I cannot|I (will|won)'?t|"
  r"I (don'?t|do not) (think|feel|believe)|As an AI|I'?m not going to|"
  r"That'?s (not something|something I)|I (must|need to) (decline|refuse))",
  re.IGNORECASE,
)

def looks_like_refusal_prefix(text: str) -> bool:
  return bool(REFUSAL_RE.match(text.strip()))

def load_model(model_path: str):
  print(f"Loading {model_path}...", flush=True)
  model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map=DEVICE, trust_remote_code=True,
    attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16,
  )
  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
  return model, tokenizer

def generate(model, tokenizer, prompt: str, max_tokens: int = 200, cancel_on_refusal_after: int = 5) -> tuple[str, bool]:
  messages = [{"role": "user", "content": prompt}]
  encoded = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=False
  )
  input_ids = (encoded["input_ids"] if hasattr(encoded, "keys") else encoded).to(DEVICE)
  attention_mask = torch.ones_like(input_ids)

  abort = threading.Event()

  class EarlyStop(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
      return abort.is_set()

  streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60)

  def _run():
    with torch.inference_mode():
      model.generate(
        input_ids, attention_mask=attention_mask,
        max_new_tokens=max_tokens, do_sample=False,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([EarlyStop()]),
      )

  t = threading.Thread(target=_run, daemon=True)
  t.start()

  collected = ""
  cancelled = False
  token_count = 0
  for token in streamer:
    collected += token
    token_count += 1
    if token_count >= cancel_on_refusal_after and looks_like_refusal_prefix(collected):
      abort.set()
      cancelled = True
      break

  t.join(timeout=10)
  return collected.strip(), cancelled

def probe_recipe(model, tokenizer, recipe: dict, prompts: list[str]) -> list[dict]:
  snapshots = apply_ablation_in_place(recipe, model)
  try:
    results = []
    for idx, prompt in enumerate(prompts):
      response, _ = generate(model, tokenizer, prompt, max_tokens=512, cancel_on_refusal_after=9999)
      results.append({"idx": idx, "prompt": prompt, "response": response})
      print(f"  PROMPT {idx}: {prompt}", flush=True)
      print(f"  RESPONSE {idx}: {response}", flush=True)
      print(f"  ---", flush=True)
    return results
  finally:
    restore_model_weights(snapshots)
    gc.collect()
    torch.cuda.empty_cache()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--onset", type=int, required=True)
  parser.add_argument("--split", type=int, required=True)
  parser.add_argument("--factor-a", type=float)
  parser.add_argument("--factor-b", type=float)
  parser.add_argument("--samples", type=int, default=10)
  parser.add_argument("--sweep", action="store_true")
  args = parser.parse_args()

  run_data = json.loads((RUNS_DIR / f"{args.run_id}.json").read_text())
  state_dir = RUNS_DIR / args.run_id
  step = run_data["sequence"][0]

  # Use only prompts the base model originally hard-refused — gives a real compliance signal
  refused_prompts = [
    p["text"] for p in run_data["prompts"]
    if p.get("text") and
    p.get("model_results", {}).get(step["model"], {}).get(step["mode"], {}).get("refused") and
    p.get("model_results", {}).get(step["model"], {}).get(step["mode"], {}).get("refusal_mode", "hard") in ("hard", "redirect")
  ]
  prompts = refused_prompts[:args.samples]
  if not prompts:
    data = json.loads((_root / "data" / "harmful_behaviors.json").read_text())
    prompts = data[:args.samples]
  print(f"Using {len(prompts)} hard-refused prompts", flush=True)

  model, tokenizer = load_model(args.model)

  if args.sweep:
    fa_vals = [round(x * 0.05, 2) for x in range(16, 25)]  # 0.80 .. 1.20
    fb_vals = [round(x * 0.05, 2) for x in range(3, 10)]   # 0.15 .. 0.45
    for fa in fa_vals:
      for fb in fb_vals:
        recipe = build_recipe(run_data, step["model"], step["mode"], args.onset, args.split, fa, fb, state_dir)
        print(f"\n===== factor_a={fa} factor_b={fb} =====", flush=True)
        probe_recipe(model, tokenizer, recipe, prompts)
  else:
    if args.factor_a is None or args.factor_b is None:
      print("Provide --factor-a and --factor-b or use --sweep")
      sys.exit(1)
    recipe = build_recipe(run_data, step["model"], step["mode"], args.onset, args.split, args.factor_a, args.factor_b, state_dir)
    print(f"\n===== factor_a={args.factor_a} factor_b={args.factor_b} =====", flush=True)
    probe_recipe(model, tokenizer, recipe, prompts)

if __name__ == "__main__":
  main()
