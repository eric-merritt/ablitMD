"""Sweep through ablation factors, testing 1 random prompt per category.

Usage:
  uv run python scripts/sweep_ablation.py <run_id> --onset 30 --split 39 \
    --factor-a 1.5 --factor-b 1.5

Prints per-category compliance summary.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import httpx

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from backend.inference.verify import looks_like_refusal

RUNS_DIR = _root / "data" / "runs"
INFERENCE_URL = "http://localhost:8238"


def sample_one_per_category(run: dict) -> list[dict]:
  by_cat: dict[str, list[dict]] = defaultdict(list)
  for prompt in run["prompts"]:
    by_cat[prompt["category"]].append(prompt)
  return [random.choice(prompts) for prompts in by_cat.values()]


def build_recipe(run_id: str, onset: int, split: int, factor_a: float, factor_b: float):
  import subprocess
  subprocess.run([
    "uv", "run", "python", "scripts/build_recipe.py", run_id,
    "--onset", str(onset), "--split", str(split),
    "--factor-a", str(factor_a), "--factor-b", str(factor_b),
  ], cwd=_root, check=True, capture_output=True)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("run_id")
  parser.add_argument("--onset", type=int, required=True)
  parser.add_argument("--split", type=int, required=True)
  parser.add_argument("--factor-a", type=float, required=True)
  parser.add_argument("--factor-b", type=float, required=True)
  args = parser.parse_args()

  run = json.loads((RUNS_DIR / f"{args.run_id}.json").read_text())
  step = run["sequence"][0]
  model_id, gen_mode = step["model"], step["mode"]

  print(f"Building recipe: onset={args.onset} split={args.split} a={args.factor_a} b={args.factor_b}")
  build_recipe(args.run_id, args.onset, args.split, args.factor_a, args.factor_b)

  prompts = sample_one_per_category(run)
  print(f"Sampled {len(prompts)} prompts (1 per category)\n")

  results = {}
  with httpx.Client(base_url=INFERENCE_URL, timeout=180.0) as client:
    for prompt in prompts:
      body = {
        "run_id": args.run_id,
        "model_id": model_id,
        "gen_mode": gen_mode,
        "categories": [prompt["category"]],
        "samples_per_category": 1,
      }
      resp = client.post("/ablate/verify", json=body)
      if resp.status_code != 200:
        print(f"[ERROR] {prompt['category']}: {resp.status_code}")
        continue

      for line in resp.text.strip().split("\n"):
        if not line.strip():
          continue
        event = json.loads(line)
        if event.get("type") == "prompt":
          after_text = event.get("response_after", "")
          refused = looks_like_refusal(after_text)
          results[prompt["category"]] = {
            "prompt": event.get("prompt_text", "")[:80],
            "refused": refused,
            "before": event.get("response_before", "")[:100],
            "after": after_text[:100],
          }

  complied = sum(1 for r in results.values() if not r["refused"])
  total = len(results)
  print(f"\n=== Results: {complied}/{total} complied ({100*complied/total:.0f}%) ===\n")

  for cat, data in sorted(results.items()):
    status = "COMPLIED" if not data["refused"] else "REFUSED"
    print(f"[{status}] {cat}")
    print(f"  Q: {data['prompt']}")
    print(f"  Before: {data['before']}...")
    print(f"  After:  {data['after']}...")
    print()


if __name__ == "__main__":
  main()
