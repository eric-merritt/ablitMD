"""Local one-pass recompute of direction_results for a run — no GPU, no inference service.

Reads the run JSON + .npy hidden states under data/runs/<run_id>/, computes per-category
refusal directions, then writes:
  - data/runs/<run_id>.directions.json   slim chart data (raw direction vectors stripped)
  - frontend/public/direction_pca.json   2D PCA of the per-category direction vectors

Pure numpy on local files — this is the whole "calculation" step; it does not need the
model loaded or a vast.ai instance.

Usage: uv run python scripts/recompute_directions.py <run_id>
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "scripts"))

from backend.inference.direction import compute_run_directions
import compute_direction_pca as pca

RUNS_DIR = _root / "data" / "runs"
PCA_RANGE = (38, 63)  # layer range averaged for the PCA, matching the prior direction_pca.json


def compute_step(run, model_id, mode, state_dir):
  """Replicates the inference service /compute endpoint, fully in-process."""
  all_hidden, all_meta = {}, {}
  for prompt in run["prompts"]:
    result = prompt.get("model_results", {}).get(model_id, {}).get(mode)
    if not result:
      continue
    key = result["hidden_states_key"]
    npy_path = state_dir / f"{key}.npy"
    if not npy_path.exists():
      continue
    all_hidden[key] = np.load(str(npy_path))
    all_meta[key] = {
      "category": prompt["category"],
      "triggers": prompt.get("triggers", []),
      "refused": result["refused"],
      "refusal_mode": result.get("refusal_mode", "hard" if result["refused"] else "none"),
    }

  per_category_meta = {}
  for key, meta in all_meta.items():
    per_category_meta.setdefault(meta["category"], []).append({
      "hidden_states_key": key,
      "refused": meta["refused"],
      "refusal_mode": meta["refusal_mode"],
    })

  per_category = {}
  for category, classifications in per_category_meta.items():
    cat_hidden = {c["hidden_states_key"]: all_hidden[c["hidden_states_key"]] for c in classifications}
    visitors = {
      key: all_hidden[key]
      for key, meta in all_meta.items()
      if meta["category"] != category and category in meta["triggers"]
    }
    trigger_meta = {
      key: {
        "source_category": all_meta[key]["category"],
        "refused": all_meta[key]["refused"],
        "refusal_mode": all_meta[key]["refusal_mode"],
      }
      for key in visitors
    }
    result = compute_run_directions(cat_hidden, classifications, category=category, visitors=visitors or None)
    if result:
      result["trigger_meta"] = trigger_meta
      per_category[category] = result
  return per_category


def write_pca(per_category, out_path):
  """2D PCA of per-category hard/redirect direction vectors — needs the raw vectors."""
  meta_by_mode = {"hard": [], "redirect": []}
  for category_id, cat_result in per_category.items():
    by_mode = (cat_result or {}).get("by_mode", {})
    for mode_name in ("hard", "redirect"):
      mode_entry = by_mode.get(mode_name)
      if not mode_entry:
        continue
      peak = pca.extract_direction(mode_entry, PCA_RANGE)
      if not peak:
        continue
      meta_by_mode[mode_name].append({
        "id": category_id,
        "name": pca.CATEGORY_NAMES.get(category_id, category_id),
        "peak_layer": peak["peak_layer"],
        "magnitude": peak["magnitude"],
        "vector": peak["vector"],
      })
  if not meta_by_mode["hard"]:
    print("[pca] no hard directions — skipped")
    return 0
  hard_matrix = np.stack([entry["vector"] for entry in meta_by_mode["hard"]], axis=0)
  payload = {
    "centered": pca.build_projection(hard_matrix, meta_by_mode, centered=True),
    "uncentered": pca.build_projection(hard_matrix, meta_by_mode, centered=False),
    "n_categories": int(hard_matrix.shape[0]),
    "hidden_dim": int(hard_matrix.shape[1]),
    "extraction": {"mode": "mean", "range": list(PCA_RANGE)},
  }
  out_path.write_text(json.dumps(payload, indent=2))
  return hard_matrix.shape[0]


def strip_vectors(per_category):
  """Drop direction_per_layer — large, and read by no chart."""
  for cat_result in per_category.values():
    for mode_data in cat_result.get("by_mode", {}).values():
      mode_data.pop("direction_per_layer", None)


def main():
  run_id = sys.argv[1] if len(sys.argv) > 1 else None
  if not run_id:
    print("usage: recompute_directions.py <run_id>")
    sys.exit(1)

  run = json.loads((RUNS_DIR / f"{run_id}.json").read_text())
  state_dir = RUNS_DIR / run_id

  direction_results = {}
  pca_written = False
  for step in run["sequence"]:
    model_id, mode = step["model"], step["mode"]
    per_category = compute_step(run, model_id, mode, state_dir)
    print(f"[recompute] {model_id}/{mode}: {len(per_category)} categories with a direction")

    if not pca_written:
      count = write_pca(per_category, _root / "frontend" / "public" / "direction_pca.json")
      print(f"[pca]       frontend/public/direction_pca.json: {count} categories")
      pca_written = True

    strip_vectors(per_category)
    direction_results.setdefault(model_id, {})[mode] = {
      "computed_at": datetime.now(timezone.utc).isoformat(),
      "per_category": per_category,
    }

  sidecar = RUNS_DIR / f"{run_id}.directions.json"
  sidecar.write_text(json.dumps(direction_results, indent=2))
  print(f"[recompute] wrote {sidecar.name}: {sidecar.stat().st_size / 1048576:.2f} MB")


if __name__ == "__main__":
  main()
