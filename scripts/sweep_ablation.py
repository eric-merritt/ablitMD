"""Auto-tune ablation factors per category.

Usage:
  uv run python scripts/sweep_ablation.py <run_id> --onset 30 --split 39
  uv run python scripts/sweep_ablation.py <run_id> --v2  # abliterate_v2 style

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


def get_qwen35_defaults(model_name: str) -> dict:
  """Return optimized defaults for Qwen3.5 variants."""
  name_lower = model_name.lower()
  if "9b" in name_lower or "8b" in name_lower or "7b" in name_lower:
    return {"layer_start": 2, "layer_end": 24, "factor_min": 0.4, "factor_max": 2.5, "step": 0.1}
  elif "27b" in name_lower or "32b" in name_lower or "30b" in name_lower:
    return {"layer_start": 8, "layer_end": 42, "factor_min": 0.5, "factor_max": 2.5, "step": 0.1}
  else:
    return {"layer_start": 2, "layer_end": None, "factor_min": 0.5, "factor_max": 2.5, "step": 0.1}


def build_recipe(run_id: str, onset: int, split: int, factor_a: float, factor_b: float):
  subprocess.run([
    "uv", "run", "python", "scripts/build_recipe.py", run_id,
    "--onset", str(onset), "--split", str(split),
    "--factor-a", str(factor_a), "--factor-b", str(factor_b),
  ], cwd=_root, check=True, capture_output=True)


def classify_response(prompt: str, response: str) -> str:
  """Call local classifier to get hard_refusal/redirect_refusal/non_refusal."""
  try:
    with httpx.Client(base_url=INFERENCE_URL, timeout=60.0) as client:
      resp = client.post("/classify", json={"prompt": prompt, "response": response})
      if resp.status_code == 200:
        return resp.json().get("classification", "unknown")
  except Exception:
    pass
  return "unknown"


def test_factor_v2(run_id: str, model_id: str, gen_mode: str, factor: float,
                   samples_per_category: int = 2) -> dict:
  """v2 mode: test single factor across ALL categories (classic ablation)."""
  with httpx.Client(base_url=INFERENCE_URL, timeout=300.0) as client:
    body = {
      "run_id": run_id,
      "model_id": model_id,
      "gen_mode": gen_mode,
      "factor": factor,
      "samples_per_category": samples_per_category,
    }
    resp = client.post("/ablate/verify/classic", json=body)
    if resp.status_code != 200:
      log_entry({"factor": factor, "status": "error", "error": f"http {resp.status_code}"})
      return {"complied": 0, "refused": 0, "degraded": 0, "total": 0}

    complied, refused, degraded, redirect = 0, 0, 0, 0
    responses = []

    for line in resp.text.strip().split("\n"):
      if not line.strip():
        continue
      event = json.loads(line)
      if event.get("type") == "prompt":
        prompt_text = event.get("prompt_text", "")
        category = event.get("category", "")
        after_text = event.get("response_after", "")
        is_deg, reason = is_degraded(after_text)

        if is_deg:
          status = f"degraded:{reason}"
          degraded += 1
        else:
          classification = classify_response(prompt_text, after_text)
          if classification == "non_refusal":
            status = "complied"
            complied += 1
          elif classification == "redirect_refusal":
            status = "redirect"
            redirect += 1
          else:
            status = "refused"
            refused += 1

        log_entry({
          "factor": factor,
          "category": category,
          "status": status,
          "prompt": prompt_text,
          "response_after": after_text,
        })
        responses.append({"category": category, "status": status, "response": after_text[:100]})

    total = complied + refused + degraded + redirect
    return {"complied": complied, "refused": refused, "degraded": degraded, "redirect": redirect,
            "total": total, "responses": responses}


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


def find_optimal_factor_v2(run_id: str, model_id: str, gen_mode: str,
                           min_factor: float, max_factor: float, step: float,
                           samples_per_category: int = 2) -> dict:
  """v2 mode: sweep single factor across ALL categories (classic ablation)."""
  print(f"\n{'='*60}")
  print(f"Classic ablation sweep: factor [{min_factor}, {max_factor}] step={step}")
  print(f"{'='*60}")

  best = None
  factor = min_factor

  while factor <= max_factor + 1e-6:
    result = test_factor_v2(run_id, model_id, gen_mode, factor, samples_per_category)
    total = result["total"]
    complied = result["complied"]
    degraded = result["degraded"]
    refused = result["refused"]
    redirect = result["redirect"]

    rate = complied / total if total > 0 else 0
    icon = "✓" if degraded == 0 and complied > refused else "✗" if degraded > 0 else "○"
    print(f"  {icon} factor={factor:.2f} → {complied}/{total} complied, {refused} refused, {redirect} redirect, {degraded} degraded")

    if degraded > 0:
      print(f"    Degradation detected, stopping.")
      break

    if complied > 0 and (best is None or complied > best["complied"]):
      best = {"factor": factor, "complied": complied, "total": total, "rate": rate}

    if complied == total and total > 0:
      print(f"    Full compliance!")
      return best

    factor = round(factor + step, 10)

  if best:
    print(f"  Best factor: {best['factor']:.2f} ({best['complied']}/{best['total']} = {best['rate']:.0%})")
    return best

  print(f"  No compliant factor found")
  return {"factor": None}


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
  parser.add_argument("--v2", action="store_true", help="Use abliterate_v2.py method (single factor, auto layer range)")
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

  print(f"Categories in run: {len(categories)}")

  results = {}

  if args.v2:
    # v2 mode: classic single-factor ablation across ALL prompts
    defaults = get_qwen35_defaults(model_id)
    factor_min = defaults["factor_min"]
    factor_max = defaults["factor_max"]
    step = defaults["step"]

    optimal = find_optimal_factor_v2(
      args.run_id, model_id, gen_mode,
      factor_min, factor_max, step,
      samples_per_category=2
    )
    results["all"] = optimal
  else:
    # Original mode: factor_a/factor_b
    print(f"Factor range: a=[{args.min_a}, {args.max_a}] b=[{args.min_b}, {args.max_b}] step={args.step}")

    for category in sorted(categories):
      optimal = find_optimal_factors(
        args.run_id, model_id, gen_mode, category,
        args.onset, args.split,
        args.min_a, args.max_a, args.min_b, args.max_b, args.step
      )
      results[category] = optimal

  print(f"\n{'='*60}")
  print("SUMMARY")
  print(f"{'='*60}")

  for key, data in sorted(results.items()):
    if args.v2:
      if data.get("factor") is not None:
        rate = data.get("rate", 0)
        print(f"  Optimal factor: {data['factor']:.2f} ({rate:.0%} compliance)")
      else:
        print(f"  NO COMPLIANCE FOUND")
    else:
      if data.get("factor_a") is not None:
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
