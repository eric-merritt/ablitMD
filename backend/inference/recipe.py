from datetime import datetime, timezone

import numpy as np

from backend.inference.run_loader import rebuild_directions


def _renormalize(vector: np.ndarray) -> np.ndarray:
  norm = float(np.linalg.norm(vector))
  return vector / norm if norm > 1e-8 else vector


def phase_a_direction(directions: list[np.ndarray], onset: int, split: int) -> np.ndarray:
  """Shared direction: mean of every (category x layer) unit vector in [onset..split],
  renormalized. directions: list of (n_layers, dim) per-category arrays."""
  stack = np.stack(directions, axis=0)            # (n_cat, n_layers, dim)
  window = stack[:, onset:split + 1, :]            # (n_cat, w, dim)
  flat = window.reshape(-1, window.shape[-1])      # (n_cat*w, dim)
  return _renormalize(flat.mean(axis=0)).astype(np.float32)



def directions_for_layer(recipe: dict, hidden_index: int) -> list[tuple[np.ndarray, float]]:
  """Directions active at a hidden-state index: phase-A mean if onset<=L<=split,
  every phase-B category direction if split<L<=last_layer. Returns [(vector, factor), ...]."""
  onset, split, last = recipe["onset"], recipe["split"], recipe["last_layer"]
  factor_a, factor_b = recipe["factor_a"], recipe["factor_b"]
  result: list[tuple[np.ndarray, float]] = []

  for mode_data in recipe["modes"].values():
    if onset <= hidden_index <= split:
      result.append((np.array(mode_data["phase_a"]["direction"], dtype=np.float32), factor_a))
    elif split < hidden_index <= last:
      layer_offset = hidden_index - split - 1
      for per_layer in mode_data["phase_b"]["per_category"].values():
        arr = np.array(per_layer, dtype=np.float32)
        # arr is either (n_layers, dim) per-layer or a single (dim,) averaged vector
        vec = arr[layer_offset] if arr.ndim == 2 else arr
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
          result.append((vec / norm, factor_b))
  return result


def build_recipe(run: dict, model_id: str, gen_mode: str, onset: int, split: int,
                 factor_a: float, factor_b: float, state_dir) -> dict:
  """Stage 2 — build the two-phase abliteration recipe (hard + redirect)."""
  per_category = rebuild_directions(run, model_id, gen_mode, state_dir)
  if not per_category:
    raise ValueError("no categories with directions")

  sample_mode = next(iter(next(iter(per_category.values()))["by_mode"].values()))
  last_layer = len(sample_mode["direction_per_layer"]) - 1

  modes: dict[str, dict] = {}
  for refusal_mode in ("hard", "redirect"):
    category_directions: dict[str, np.ndarray] = {}
    for category_id, cat_result in per_category.items():
      entry = cat_result["by_mode"].get(refusal_mode)
      if entry:
        category_directions[category_id] = np.array(entry["direction_per_layer"], dtype=np.float32)
    if not category_directions:
      continue

    shared = phase_a_direction(list(category_directions.values()), onset, split)
    modes[refusal_mode] = {
      "phase_a": {"layers": [onset, split], "direction": shared.tolist()},
      "phase_b": {
        "layers": [split, last_layer],
        "per_category": {
          category_id: dpl[split:last_layer + 1, :].tolist()
          for category_id, dpl in category_directions.items()
        },
      },
    }

  return {
    "run_id": run["run_id"], "model_id": model_id, "gen_mode": gen_mode,
    "onset": onset, "split": split, "last_layer": last_layer,
    "factor_a": factor_a, "factor_b": factor_b,
    "modes": modes, "built_at": datetime.now(timezone.utc).isoformat(),
  }
