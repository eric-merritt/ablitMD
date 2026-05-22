"""Auto-tune ablation factors per category.

Usage:
  uv run python scripts/sweep_ablation.py <run_id> --onset 30 --split 39

Finds optimal factors for each category independently.
Logs all responses to <run_id>.sweep_log.jsonl
"""

import argparse
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from backend.inference.verify import looks_like_refusal

RUNS_DIR = _root / "data" / "runs"
INFERENCE_URL = "http://localhost:8238"

_log_file = None


def log_entry(entry: dict):
  global _log_file
  if _log_file:
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    _log_file.write(json.dumps(entry) + "\n")
    _log_file.flush()

DEGRADATION_WORDS = {"inage", "dubious", "enforcement", "sizzle", "sin"}
DEGRADATION_WORDS_RE = re.compile(r'\b(' + '|'.join(DEGRADATION_WORDS) + r')\b', re.IGNORECASE)
DEGRADATION_CJK_RE = re.compile(r'[一-鿿㐀-䶿]{3,}')


def is_degraded(text: str) -> tuple[bool, str]:
  lower = text.lower()
  if lower.count("aid") > 3:
    return True, "aid_spam"
  match = DEGRADATION_WORDS_RE.search(text)
  if match:
    return True, f"word:{match.group(1)}"
  if DEGRADATION_CJK_RE.search(text):
    return True, "cjk_spam"
  if re.search(r'(.{10,}?)\1{2,}', text):
    return True, "loop"
  return False, ""


def get_categories(run: dict) -> list[str]:
  return list({p["category"] for p in run["prompts"]})


def sample_prompt(run: dict, category: str) -> dict:
  prompts = [p for p in run["prompts"] if p["category"] == category]
  return random.choice(prompts)


def build_recipe(run_id: str, onset: int, split: int, factor_a: float, factor_b: float):
  subprocess.run([
    "uv", "run", "python", "scripts/build_recipe.py", run_id,
    "--onset", str(onset), "--split", str(split),
    "--factor-a", str(factor_a), "--factor-b", str(factor_b),
  ], cwd=_root, check=True, capture_output=True)


def test_category(run_id: str, model_id: str, gen_mode: str, category: str,
                  onset: int, split: int, factor_a: float, factor_b: float) -> dict:
  build_recipe(run_id, onset, split, factor_a, factor_b)

  with httpx.Client(base_url=INFERENCE_URL, timeout=180.0) as client:
    body = {
      "run_id": run_id,
      "model_id": model_id,
      "gen_mode": gen_mode,
      "categories": [category],
      "samples_per_category": 1,
      "fast": True,
    }
    resp = client.post("/ablate/verify", json=body)
    if resp.status_code != 200:
      log_entry({"category": category, "factor_a": factor_a, "factor_b": factor_b,
                 "status": "error", "error": f"http {resp.status_code}"})
      return {"status": "error", "response": ""}

    for line in resp.text.strip().split("\n"):
      if not line.strip():
        continue
      event = json.loads(line)
      if event.get("type") == "prompt":
        prompt_text = event.get("prompt_text", "")
        before_text = event.get("response_before", "")
        after_text = event.get("response_after", "")
        degraded, reason = is_degraded(after_text)

        if degraded:
          status = f"degraded:{reason}"
        elif looks_like_refusal(after_text):
          status = "refused"
        else:
          status = "complied"

        log_entry({
          "category": category,
          "factor_a": factor_a,
          "factor_b": factor_b,
          "status": status,
          "prompt": prompt_text,
          "response_before": before_text,
          "response_after": after_text,
        })

        return {"status": status, "response": after_text[:200]}

  return {"status": "error", "response": ""}


def find_optimal_factors(run_id: str, model_id: str, gen_mode: str, category: str,
                         onset: int, split: int, min_a: float, max_a: float,
                         min_b: float, max_b: float, step: float) -> dict:
  print(f"\n{'='*60}")
  print(f"Category: {category}")
  print(f"{'='*60}")

  best = None
  factor_a = min_a

  while factor_a <= max_a:
    factor_b = min_b
    while factor_b <= max_b:
      result = test_category(run_id, model_id, gen_mode, category,
                             onset, split, factor_a, factor_b)

      status = result["status"]
      icon = "✓" if status == "complied" else "✗" if "degraded" in status else "○"
      print(f"  {icon} a={factor_a:.2f} b={factor_b:.2f} → {status}")

      if status == "complied":
        best = {"factor_a": factor_a, "factor_b": factor_b, "response": result["response"]}
        print(f"    Found compliance! Stopping search for this category.")
        return best

      if "degraded" in status:
        print(f"    Degradation detected, trying next factor_a")
        break

      factor_b += step
    factor_a += step

  if best is None:
    print(f"  No compliant factors found (max tested: a={max_a} b={max_b})")
    return {"factor_a": None, "factor_b": None, "response": ""}

  return best


def main():
  global _log_file

  parser = argparse.ArgumentParser()
  parser.add_argument("run_id")
  parser.add_argument("--onset", type=int, default=30)
  parser.add_argument("--split", type=int, default=39)
  parser.add_argument("--min-a", type=float, default=0.5)
  parser.add_argument("--max-a", type=float, default=3.0)
  parser.add_argument("--min-b", type=float, default=1.0)
  parser.add_argument("--max-b", type=float, default=3.0)
  parser.add_argument("--step", type=float, default=0.25)
  parser.add_argument("--categories", type=str, default="", help="comma-separated list, or empty for all")
  args = parser.parse_args()

  log_path = RUNS_DIR / f"{args.run_id}.sweep_log.jsonl"
  _log_file = open(log_path, "a")
  log_entry({"event": "sweep_start", "args": vars(args)})
  print(f"Logging to {log_path}")

  run = json.loads((RUNS_DIR / f"{args.run_id}.json").read_text())
  step_data = run["sequence"][0]
  model_id, gen_mode = step_data["model"], step_data["mode"]

  with httpx.Client(base_url=INFERENCE_URL, timeout=300.0) as client:
    status = client.get("/status").json()
    if status.get("loaded_model") != model_id:
      print(f"Loading model {model_id}...")
      client.post("/load", json={"model_id": model_id, "api_model_id": model_id})
      print("Model loaded.")

  if args.categories:
    categories = [c.strip() for c in args.categories.split(",")]
  else:
    categories = get_categories(run)

  print(f"Testing {len(categories)} categories")
  print(f"Factor range: a=[{args.min_a}, {args.max_a}] b=[{args.min_b}, {args.max_b}] step={args.step}")

  results = {}
  for category in sorted(categories):
    optimal = find_optimal_factors(
      args.run_id, model_id, gen_mode, category,
      args.onset, args.split,
      args.min_a, args.max_a, args.min_b, args.max_b, args.step
    )
    results[category] = optimal

  print(f"\n{'='*60}")
  print("SUMMARY: Optimal factors per category")
  print(f"{'='*60}")

  for cat, data in sorted(results.items()):
    if data["factor_a"] is not None:
      print(f"  {cat}: a={data['factor_a']:.2f} b={data['factor_b']:.2f}")
    else:
      print(f"  {cat}: NO COMPLIANCE FOUND")

  out_path = RUNS_DIR / f"{args.run_id}.category_factors.json"
  out_path.write_text(json.dumps(results, indent=2))
  print(f"\nSaved to {out_path}")

  log_entry({"event": "sweep_end", "results": results})
  _log_file.close()


if __name__ == "__main__":
  main()
