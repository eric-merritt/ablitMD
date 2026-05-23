"""Brainstorming analysis: how clustered ARE the per-category direction vectors?
Informs the 'recommended abliteration directions' design — pure inspection, writes nothing.
"""

import json
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "scripts"))

import recompute_directions as rd
import compute_direction_pca as pca

RUN_ID = "run_2026-05-17T19-57-17-480Z_8696f34a"

run = json.loads((rd.RUNS_DIR / f"{RUN_ID}.json").read_text())
step = run["sequence"][0]
per_category = rd.compute_step(run, step["model"], step["mode"], rd.RUNS_DIR / RUN_ID)

# one representative vector per category: hard direction averaged over layers 38-63
cats, vecs = [], []
for cat_id, result in sorted(per_category.items()):
  hard = result.get("by_mode", {}).get("hard")
  if not hard:
    continue
  extracted = pca.extract_direction(hard, (38, 63))
  if not extracted:
    continue
  vector = extracted["vector"].astype(np.float64)
  cats.append(cat_id)
  vecs.append(vector / (np.linalg.norm(vector) or 1.0))

n = len(cats)
matrix = np.stack(vecs)
sim = np.clip(matrix @ matrix.T, -1.0, 1.0)
dist = 1.0 - sim
off_diagonal = sim[np.triu_indices(n, k=1)]

print(f"=== {n} categories with a hard direction (mean L38-63, unit vectors) ===\n")
print("pairwise cosine similarity distribution:")
for low in np.arange(-0.2, 1.0, 0.1):
  count = int(((off_diagonal >= low) & (off_diagonal < low + 0.1)).sum())
  print(f"  [{low:+.1f}, {low + 0.1:+.1f})  {'#' * count}  {count}")
print(f"  min={off_diagonal.min():.2f}  median={np.median(off_diagonal):.2f}  "
      f"mean={off_diagonal.mean():.2f}  max={off_diagonal.max():.2f}")


def agglomerate(stop_at):
  """Average-linkage agglomerative clustering; returns (merge_log, groups_at_stop)."""
  groups = {idx: [idx] for idx in range(n)}
  next_id = n
  merge_log = []
  while len(groups) > stop_at:
    keys = list(groups)
    best = None
    for i in range(len(keys)):
      for j in range(i + 1, len(keys)):
        gap = np.mean([dist[x, y] for x in groups[keys[i]] for y in groups[keys[j]]])
        if best is None or gap < best[0]:
          best = (gap, keys[i], keys[j])
    gap, a, b = best
    merge_log.append((len(groups), gap))
    groups[next_id] = groups[a] + groups[b]
    del groups[a], groups[b]
    next_id += 1
  return merge_log, [sorted(cats[i] for i in g) for g in groups.values()]

merge_log, _ = agglomerate(1)
print("\nmerge heights — cosine distance at which the cluster count drops:")
for count, gap in merge_log:
  print(f"  {count:2d} -> {count - 1:2d}   merge distance {gap:.3f}")

# biggest jump between consecutive merges = natural cluster count
jumps = [(merge_log[i][0], merge_log[i + 1][1] - merge_log[i][1])
         for i in range(len(merge_log) - 1)]
natural_k = max(jumps, key=lambda t: t[1])[0]
print(f"\nlargest jump suggests a natural K = {natural_k}")
_, groups = agglomerate(natural_k)
for i, group in enumerate(sorted(groups, key=len, reverse=True), 1):
  print(f"  cluster {i} ({len(group)}): {', '.join(group)}")
