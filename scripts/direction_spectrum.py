"""Dump full SVD spectrum of per-category hard directions.

Shows the effective rank of the polyhedral cone — how many independent
directions the categories actually span.

Default uses each category's peak-layer direction. With --range L_START L_END,
averages directions across the inclusive layer window.

Usage:
  uv run python scripts/direction_spectrum.py <run_json_path> [--range 38 63]
"""

import json
import sys
from pathlib import Path

import numpy as np


def main():
  if len(sys.argv) < 2:
    print("usage: direction_spectrum.py <run_json_path> [--range L_START L_END]")
    sys.exit(1)

  run_path = Path(sys.argv[1])
  layer_range = None
  if "--range" in sys.argv:
    idx = sys.argv.index("--range")
    layer_range = (int(sys.argv[idx + 1]), int(sys.argv[idx + 2]))

  run = json.loads(run_path.read_text())
  direction_results = run.get("direction_results") or {}

  rows = []
  category_ids = []
  for category_id, cat_result in direction_results.items():
    hard = (cat_result or {}).get("by_mode", {}).get("hard")
    if not hard or not hard.get("direction_per_layer") or not hard.get("magnitude_per_layer"):
      continue
    direction_per_layer = hard["direction_per_layer"]
    if layer_range:
      start = max(0, layer_range[0])
      end = min(len(direction_per_layer) - 1, layer_range[1])
      vectors = np.array(direction_per_layer[start:end + 1], dtype=np.float32)
      rows.append(vectors.mean(axis=0))
    else:
      magnitudes = np.array(hard["magnitude_per_layer"], dtype=np.float32)
      peak_layer = int(np.argmax(magnitudes))
      rows.append(np.array(direction_per_layer[peak_layer], dtype=np.float32))
    category_ids.append(category_id)

  matrix = np.stack(rows, axis=0)
  centered = matrix - matrix.mean(axis=0, keepdims=True)
  _U, singular_values, _Vt = np.linalg.svd(centered, full_matrices=False)

  variances = (singular_values ** 2)
  total = variances.sum()
  ratios = variances / total
  cumulative = np.cumsum(ratios)

  mode_label = f"mean L{layer_range[0]}–L{layer_range[1]}" if layer_range else "peak-layer"
  print(f"matrix: {matrix.shape[0]} categories × {matrix.shape[1]} dims · extraction: {mode_label}")
  print()
  print(f"{'PC':>4} {'σ':>14} {'variance %':>12} {'cumulative %':>14}")
  print("-" * 50)
  for idx, (sigma, ratio, cum) in enumerate(zip(singular_values, ratios, cumulative)):
    bar = "█" * max(1, int(ratio * 60))
    print(f"{idx + 1:>4} {sigma:>14.4f} {ratio * 100:>11.2f}% {cum * 100:>13.2f}%  {bar}")

  print()
  for threshold in (0.50, 0.80, 0.90, 0.95, 0.99):
    rank = int(np.searchsorted(cumulative, threshold)) + 1
    print(f"PCs needed to reach {int(threshold * 100)}% variance: {rank}")


if __name__ == "__main__":
  main()
