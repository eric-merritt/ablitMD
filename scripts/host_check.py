"""Simplistic model host + abliteration checker for Qwen3.8-27B.

Loads a model directory (base or baked -Ablit), runs a set of prompts, and
classifies each response as refuse-vs-comply using the same rule-based classifier
the run pipeline uses (backend.inference.verify.auto_classify_response). Reports a
per-category compliance rate so you can tell at a glance whether the model has
been ablated (complies where the base refused) or not.

This is deliberately self-contained — no FastAPI, no Mongo, no streaming. It's the
script to point at a freshly-rented vast.ai box: download the base model + this
repo, bake an -Ablit copy, then run this against both and diff the compliance rates.

Usage:
  # Check the base model (expect refusals on the harmful categories):
  uv run python scripts/host_check.py ~/models/Qwen/Qwen3.8-27B-Base \
    --run-id run_2026-09-19T10-01-32-556Z_bfb44921

  # Check the baked abliterated model (expect compliance where base refused):
  uv run python scripts/host_check.py ~/models/Qwen/Qwen3.8-27B-Ablit \
    --run-id run_2026-09-19T10-01-32-556Z_bfb44921

  # Compare both side by side and print a per-category delta:
  uv run python scripts/host_check.py ~/models/Qwen/Qwen3.8-27B-Base \
    ~/models/Qwen/Qwen3.8-27B-Ablit --run-id run_... --compare

Options:
  --samples N        only test the first N prompts per category (default: all)
  --max-tokens N     generation cap (default 128, matches the run pipeline)
  --json PATH        also dump full per-prompt results to a JSON file
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.inference.verify import auto_classify_response, looks_like_refusal

RUNS_DIR = _root / "data" / "runs"


def load_model(model_dir: str):
  print(f"[host_check] loading {model_dir} (bf16, device_map=auto)", flush=True)
  model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="flash_attention_2", trust_remote_code=True,
  )
  model.eval()
  tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
  return model, tokenizer


def run_prompt(model, tokenizer, text: str, max_tokens: int) -> str:
  messages = [{"role": "user", "content": text}]
  try:
    input_ids = tokenizer.apply_chat_template(
      messages, add_generation_prompt=True, return_tensors="pt",
      enable_thinking=False,
    ).to(model.device)
  except TypeError:
    input_ids = tokenizer.apply_chat_template(
      messages, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
  with torch.inference_mode():
    out = model.generate(
      input_ids, max_new_tokens=max_tokens, do_sample=False,
      pad_token_id=tokenizer.pad_token_id,
    )
  gen = out[0][input_ids.shape[1]:]
  return tokenizer.decode(gen, skip_special_tokens=True).strip()


def load_test_prompts(run_id: str, samples: int):
  """Pull the run's prompts (text + category), capped to `samples` per category."""
  path = RUNS_DIR / f"{run_id}.json"
  if not path.exists():
    raise SystemExit(f"[host_check] no run file: {path}")
  run = json.loads(path.read_text())
  by_cat: dict[str, list[dict]] = defaultdict(list)
  for p in run["prompts"]:
    cat = p.get("category", "unknown")
    if samples and len(by_cat[cat]) >= samples:
      continue
    by_cat[cat].append({"text": p["text"], "category": cat})
  prompts = [p for cat in sorted(by_cat) for p in by_cat[cat]]
  return prompts


def check_model(model_dir: str, prompts: list[dict], max_tokens: int) -> dict:
  model, tokenizer = load_model(model_dir)
  results = []
  per_cat: dict[str, dict] = defaultdict(lambda: {"n": 0, "complied": 0})
  for i, p in enumerate(prompts):
    text = run_prompt(model, tokenizer, p["text"], max_tokens)
    cls = auto_classify_response(text)
    refused = looks_like_refusal(text)
    cat = p["category"]
    per_cat[cat]["n"] += 1
    if not refused:
      per_cat[cat]["complied"] += 1
    results.append({
      "category": cat,
      "prompt": p["text"],
      "response": text,
      "refused": refused,
      "class": cls,
    })
    if (i + 1) % 20 == 0 or i + 1 == len(prompts):
      print(f"[host_check] {model_dir} {i+1}/{len(prompts)}", flush=True)

  total = len(results)
  complied = sum(1 for r in results if not r["refused"])
  return {
    "model_dir": model_dir,
    "n_prompts": total,
    "complied": complied,
    "refused": total - complied,
    "compliance_rate": (complied / total) if total else 0.0,
    "per_category": per_cat,
    "results": results,
  }


def fmt_per_cat(per_cat: dict) -> str:
  lines = []
  for cat in sorted(per_cat):
    v = per_cat[cat]
    lines.append(f"  {cat:32s} {v['complied']:>2d}/{v['n']:<2d}")
  return "\n".join(lines)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("model_dir", help="path to base or -Ablit model dir")
  ap.add_argument("second_model_dir", nargs="?", default=None,
                  help="optional second model dir for --compare")
  ap.add_argument("--run-id", required=True)
  ap.add_argument("--samples", type=int, default=0,
                  help="max prompts per category (0 = all)")
  ap.add_argument("--max-tokens", type=int, default=128)
  ap.add_argument("--compare", action="store_true",
                  help="load both model dirs and print a per-category delta")
  ap.add_argument("--json", dest="json_path", default=None,
                  help="dump full per-prompt results to this JSON file")
  args = ap.parse_args()

  prompts = load_test_prompts(args.run_id, args.samples)
  print(f"[host_check] {len(prompts)} test prompts from {args.run_id}", flush=True)

  a = check_model(args.model_dir, prompts, args.max_tokens)
  print(f"\n=== {a['model_dir']} ===")
  print(f"compliance: {a['complied']}/{a['n_prompts']} ({a['compliance_rate']:.1%})")
  print(fmt_per_cat(a["per_category"]))

  if args.compare and args.second_model_dir:
    b = check_model(args.second_model_dir, prompts, args.max_tokens)
    print(f"\n=== {b['model_dir']} ===")
    print(f"compliance: {b['complied']}/{b['n_prompts']} ({b['compliance_rate']:.1%})")
    print("\n=== per-category delta (base -> ablit) ===")
    cats = sorted(set(a["per_category"]) | set(b["per_category"]))
    for cat in cats:
      av = a["per_category"].get(cat, {"n": 0, "complied": 0})
      bv = b["per_category"].get(cat, {"n": 0, "complied": 0})
      delta = bv["complied"] - av["complied"]
      flag = "  <== flipped" if delta > 0 else ""
      print(f"  {cat:32s} {av['complied']}/{av['n']} -> {bv['complied']}/{bv['n']}  ({delta:+d}){flag}")

  if args.json_path:
    out = {"a": a, "b": b if (args.compare and args.second_model_dir) else None}
    Path(args.json_path).write_text(json.dumps(out, indent=2))
    print(f"\n[host_check] wrote {args.json_path}")


if __name__ == "__main__":
  main()
