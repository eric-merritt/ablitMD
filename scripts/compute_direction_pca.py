"""Compute 2D embedding of per-category direction vectors (hard + redirect).

Produces TWO projections side by side:
  - centered:   PCA on (M - mean), arrows from centroid, shows deviations
  - uncentered: SVD on raw M, arrows from origin, shows cone-from-origin geometry

Default uses each category's PEAK-LAYER direction. With --range L_START L_END,
averages direction vectors across the inclusive layer range.

Usage:
  uv run python scripts/compute_direction_pca.py <run_json_path> [--range 38 63] [--out path]
"""

import json
import sys
from pathlib import Path

import numpy as np


CATEGORY_NAMES = {
  "academic_dishonesty":      "Academic Dishonesty",
  "cbrn_uplift":              "CBRN Uplift",
  "child_grooming":           "Child Grooming",
  "confidential_data":        "Confidential Data",
  "copyright_ip":             "Copyright / IP",
  "crime_assistance":         "Crime Assistance",
  "csam":                     "CSAM",
  "dangerous_activity":       "Dangerous Activity",
  "disordered_eating":        "Disordered Eating",
  "distress_signals":         "Distress Signals",
  "excessive_requests":       "Excessive Requests",
  "explicit_nsfw":            "Explicit NSFW",
  "financial_advice":         "Financial Advice",
  "fraud_facilitation":       "Fraud Facilitation",
}


def peak_direction(mode_entry):
  direction_per_layer = mode_entry.get("direction_per_layer")
  magnitude_per_layer = mode_entry.get("magnitude_per_layer")
  if not direction_per_layer or not magnitude_per_layer:
    return None
  magnitudes = np.array(magnitude_per_layer, dtype=np.float32)
  peak_layer = int(np.argmax(magnitudes))
  return {
    "vector":     np.array(direction_per_layer[peak_layer], dtype=np.float32),
    "magnitude":  float(magnitudes[peak_layer]),
    "peak_layer": peak_layer,
  }


def mean_direction(mode_entry, layer_start, layer_end):
  direction_per_layer = mode_entry.get("direction_per_layer")
  magnitude_per_layer = mode_entry.get("magnitude_per_layer")
  if not direction_per_layer or not magnitude_per_layer:
    return None
  n_layers = len(direction_per_layer)
  start = max(0, layer_start)
  end = min(n_layers - 1, layer_end)
  if end < start:
    return None
  vectors = np.array(direction_per_layer[start:end + 1], dtype=np.float32)
  mean_vector = vectors.mean(axis=0)
  return {
    "vector":     mean_vector,
    "magnitude":  float(np.linalg.norm(mean_vector)),
    "peak_layer": (start + end) // 2,
    "range":      [start, end],
  }


def extract_direction(mode_entry, layer_range):
  return mean_direction(mode_entry, *layer_range) if layer_range else peak_direction(mode_entry)


def pin_signs(U, Vt):
  """sklearn-style svd_flip — make each PC's max-abs entry in U positive."""
  max_abs_rows = np.argmax(np.abs(U[:, :2]), axis=0)
  signs = np.sign(U[max_abs_rows, np.arange(2)])
  signs[signs == 0] = 1
  return Vt[:2, :] * signs[:, None]


def build_basis(matrix, centered):
  """Return (basis_2x_dim, variance_explained, mean_or_None)."""
  if centered:
    mean = matrix.mean(axis=0)
    working = matrix - mean
  else:
    mean = None
    working = matrix

  U, S, Vt = np.linalg.svd(working, full_matrices=False)
  basis = pin_signs(U, Vt)
  total_variance = float((S ** 2).sum())
  variance_explained = [float((S[idx] ** 2) / total_variance) for idx in range(min(2, len(S)))]
  return basis, variance_explained, mean


def project_arrow(vector, basis, mean, magnitude):
  reference = vector - mean if mean is not None else vector
  coords = basis @ reference
  norm = float(np.linalg.norm(coords))
  if norm < 1e-9:
    return coords
  return coords * (magnitude / norm)


def parse_args(argv):
  if len(argv) < 2:
    print("usage: compute_direction_pca.py <run_json_path> [--range L_START L_END] [--out path]")
    sys.exit(1)

  args = {"run_path": Path(argv[1]), "range": None, "out_path": Path("frontend/public/direction_pca.json")}
  idx = 2
  while idx < len(argv):
    token = argv[idx]
    if token == "--range" and idx + 2 < len(argv):
      args["range"] = (int(argv[idx + 1]), int(argv[idx + 2]))
      idx += 3
    elif token == "--out" and idx + 1 < len(argv):
      args["out_path"] = Path(argv[idx + 1])
      idx += 2
    else:
      args["out_path"] = Path(token)
      idx += 1
  return args


def build_projection(matrix, meta_by_mode, centered):
  basis, variance_explained, mean = build_basis(matrix, centered=centered)
  arrows_by_mode = {}
  for mode_name, entries in meta_by_mode.items():
    arrows = []
    for entry in entries:
      coords = project_arrow(entry["vector"], basis, mean, entry["magnitude"])
      arrows.append({
        "id":         entry["id"],
        "name":       entry["name"],
        "peak_layer": entry["peak_layer"],
        "magnitude":  entry["magnitude"],
        "x":          float(coords[0]),
        "y":          float(coords[1]),
      })
    arrows_by_mode[mode_name] = arrows
  return {**arrows_by_mode, "variance_explained": variance_explained}


def main():
  args = parse_args(sys.argv)
  run_path = args["run_path"]
  out_path = args["out_path"]
  layer_range = args["range"]

  run = json.loads(run_path.read_text())
  direction_results = run.get("direction_results") or {}

  meta_by_mode: dict[str, list[dict]] = {"hard": [], "redirect": []}

  for category_id, cat_result in direction_results.items():
    by_mode = (cat_result or {}).get("by_mode", {})
    for mode_name in ("hard", "redirect"):
      mode_entry = by_mode.get(mode_name)
      if not mode_entry: continue
      peak = extract_direction(mode_entry, layer_range)
      if not peak: continue
      meta_by_mode[mode_name].append({
        "id":         category_id,
        "name":       CATEGORY_NAMES.get(category_id, category_id),
        "peak_layer": peak["peak_layer"],
        "magnitude":  peak["magnitude"],
        "vector":     peak["vector"],
      })

  if not meta_by_mode["hard"]:
    print("[pca] no hard directions found")
    sys.exit(1)

  hard_matrix = np.stack([entry["vector"] for entry in meta_by_mode["hard"]], axis=0)

  centered_payload   = build_projection(hard_matrix, meta_by_mode, centered=True)
  uncentered_payload = build_projection(hard_matrix, meta_by_mode, centered=False)

  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps({
    "centered":     centered_payload,
    "uncentered":   uncentered_payload,
    "n_categories": hard_matrix.shape[0],
    "hidden_dim":   hard_matrix.shape[1],
    "extraction":   {"mode": "mean", "range": list(layer_range)} if layer_range else {"mode": "peak"},
  }, indent=2))

  mode_label = f"mean L{layer_range[0]}–L{layer_range[1]}" if layer_range else "peak"
  centered_var   = centered_payload["variance_explained"]
  uncentered_var = uncentered_payload["variance_explained"]
  print(f"[pca] wrote {out_path} — {mode_label}")
  print(f"      centered   PC1={centered_var[0]:.3f} PC2={centered_var[1]:.3f}")
  print(f"      uncentered PC1={uncentered_var[0]:.3f} PC2={uncentered_var[1]:.3f}")


if __name__ == "__main__":
  main()
