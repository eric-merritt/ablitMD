"""Auto-tune ablation factors by sweeping through values and detecting quality.

Usage:
  uv run python scripts/sweep_ablation.py <run_id> --onset 30 --split 39

Starts at conservative factors, ramps up until degradation detected,
backs off to find optimal compliance without quality loss.
"""

import argparse
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import httpx

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from backend.inference.verify import looks_like_refusal

RUNS_DIR = _root / "data" / "runs"
INFERENCE_URL = "http://localhost:8238"

DEGRADATION_WORDS = {"inage", "dubious", "enforcement", "sizzle", "sin"}
DEGRADATION_CJK_RE = re.compile(r'[一-鿿㐀-䶿]{3,}')


def is_degraded(text: str) -> tuple[bool, str]:
  lower = text.lower()
  if lower.count("aid") > 3:
    return True, "aid_spam"
  for word in DEGRADATION_WORDS:
    if word in lower:
      return True, f"degradation_word:{word}"
  if DEGRADATION_CJK_RE.search(text):
    return True, "cjk_spam"
  repeated = re.search(r'(.{10,}?)\1{2,}', text)
  if repeated:
    return True, "repetition_loop"
  return False, ""


def sample_one_per_category(run: dict) -> list[dict]:
  by_cat: dict[str, list[dict]] = defaultdict(list)
  for prompt in run["prompts"]:
    by_cat[prompt["category"]].append(prompt)
  return [random.choice(prompts) for prompts in by_cat.values()]


def build_recipe(run_id: str, onset: int, split: int, factor_a: float, factor_b: float):
  subprocess.run([
    "uv", "run", "python", "scripts/build_recipe.py", run_id,
    "--onset", str(onset), "--split", str(split),
    "--factor-a", str(factor_a), "--factor-b", str(factor_b),
  ], cwd=_root, check=True, capture_output=True)


def test_factors(run_id: str, model_id: str, gen_mode: str, prompts: list[dict],
                 onset: int, split: int, factor_a: float, factor_b: float) -> dict:
  build_recipe(run_id, onset, split, factor_a, factor_b)

  results = {"complied": 0, "refused": 0, "degraded": 0, "details": []}

  with httpx.Client(base_url=INFERENCE_URL, timeout=180.0) as client:
    for prompt in prompts:
      body = {
        "run_id": run_id,
        "model_id": model_id,
        "gen_mode": gen_mode,
        "categories": [prompt["category"]],
        "samples_per_category": 1,
      }
      resp = client.post("/ablate/verify", json=body)
      if resp.status_code != 200:
        continue

      for line in resp.text.strip().split("\n"):
        if not line.strip():
          continue
        event = json.loads(line)
        if event.get("type") == "prompt":
          after_text = event.get("response_after", "")
          degraded, reason = is_degraded(after_text)
          refused = looks_like_refusal(after_text) if not degraded else False

          if degraded:
            results["degraded"] += 1
            status = f"DEGRADED({reason})"
          elif refused:
            results["refused"] += 1
            status = "REFUSED"
          else:
            results["complied"] += 1
            status = "COMPLIED"

          results["details"].append({
            "category": prompt["category"],
            "status": status,
            "response": after_text[:150],
          })

  return results


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("run_id")
  parser.add_argument("--onset", type=int, default=30)
  parser.add_argument("--split", type=int, default=39)
  parser.add_argument("--min-a", type=float, default=0.5)
  parser.add_argument("--max-a", type=float, default=2.5)
  parser.add_argument("--min-b", type=float, default=1.0)
  parser.add_argument("--max-b", type=float, default=2.5)
  parser.add_argument("--step", type=float, default=0.25)
  args = parser.parse_args()

  run = json.loads((RUNS_DIR / f"{args.run_id}.json").read_text())
  step = run["sequence"][0]
  model_id, gen_mode = step["model"], step["mode"]

  prompts = sample_one_per_category(run)
  print(f"Sampled {len(prompts)} prompts (1 per category)\n")

  best = {"factor_a": args.min_a, "factor_b": args.min_b, "complied": 0, "degraded": 0}
  all_results = []

  factor_a = args.min_a
  while factor_a <= args.max_a:
    factor_b = args.min_b
    while factor_b <= args.max_b:
      print(f"\n--- Testing a={factor_a:.2f} b={factor_b:.2f} ---")
      results = test_factors(args.run_id, model_id, gen_mode, prompts,
                             args.onset, args.split, factor_a, factor_b)

      total = results["complied"] + results["refused"] + results["degraded"]
      print(f"  Complied: {results['complied']}/{total}")
      print(f"  Refused:  {results['refused']}/{total}")
      print(f"  Degraded: {results['degraded']}/{total}")

      all_results.append({
        "factor_a": factor_a,
        "factor_b": factor_b,
        **results,
      })

      if results["degraded"] == 0 and results["complied"] > best["complied"]:
        best = {"factor_a": factor_a, "factor_b": factor_b, **results}
        print(f"  ** New best! **")

      if results["degraded"] > total * 0.3:
        print(f"  High degradation, skipping higher factor_b values")
        break

      factor_b += args.step
    factor_a += args.step

  print(f"\n{'='*60}")
  print(f"BEST FACTORS: a={best['factor_a']:.2f} b={best['factor_b']:.2f}")
  print(f"  Complied: {best['complied']}")
  print(f"  Degraded: {best['degraded']}")
  print(f"{'='*60}\n")

  out_path = RUNS_DIR / f"{args.run_id}.sweep_results.json"
  out_path.write_text(json.dumps(all_results, indent=2))
  print(f"Full results saved to {out_path}")


if __name__ == "__main__":
  main()
