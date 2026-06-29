import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.inference.run_loader import rebuild_directions


def _sidecar_has_vectors(data: dict) -> bool:
  """The directions sidecar / Mongo docs are normally stripped of direction_per_layer
  (the ~150MB per-layer vectors) since no chart reads them. Recipe building DOES need
  them, so only trust the sidecar when those vectors are actually present."""
  for cat_result in data.values():
    if not isinstance(cat_result, dict):
      continue
    by_mode = cat_result.get("by_mode")
    if not isinstance(by_mode, dict):
      continue
    for mode_data in by_mode.values():
      if isinstance(mode_data, dict) and mode_data.get("direction_per_layer"):
        return True
  return False


def load_directions(run: dict, model_id: str, gen_mode: str, state_dir: Path) -> dict:
  """Per-category directions for recipe building. Use the directions sidecar only when
  it carries the full per-layer vectors (a future full-directions pull); otherwise
  recompute from the raw .npy hidden states, which always have them."""
  sidecar = state_dir.parent / f"{run['run_id']}.directions.json"
  if sidecar.exists():
    data = json.loads(sidecar.read_text())
    if data and _sidecar_has_vectors(data):
      return {k: v for k, v in data.items() if isinstance(v, dict) and "by_mode" in v}
  return rebuild_directions(run, model_id, gen_mode, state_dir)


def recipe_filename(run_id: str) -> str:
  """Timestamped recipe filename: <run_id>.recipe.<YYYY-MM-DD_HHMMSS>.json.
  Every build writes a fresh file so each baked model is traceable to its recipe."""
  return f"{run_id}.recipe.{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"


def latest_recipe_path(runs_dir: Path, run_id: str):
  """Newest recipe for a run: most-recently-written of the timestamped files,
  falling back to a legacy <run_id>.recipe.json. Returns None if none exist."""
  candidates = list(runs_dir.glob(f"{run_id}.recipe.*.json"))
  legacy = runs_dir / f"{run_id}.recipe.json"
  if legacy.exists():
    candidates.append(legacy)
  if not candidates:
    return None
  return max(candidates, key=lambda path: path.stat().st_mtime)


def _renormalize(vector: np.ndarray) -> np.ndarray:
  norm = float(np.linalg.norm(vector))
  return vector / norm if norm > 1e-8 else vector


def shared_direction(directions: list[np.ndarray], start: int, end: int) -> np.ndarray:
  """Single unit vector: mean of every (category x layer) direction in [start..end],
  renormalized. directions: list of (n_layers, dim) per-category arrays."""
  stack = np.stack(directions, axis=0)            # (n_cat, n_layers, dim)
  window = stack[:, start:end + 1, :]             # (n_cat, w, dim)
  flat = window.reshape(-1, window.shape[-1])     # (n_cat*w, dim)
  return _renormalize(flat.mean(axis=0)).astype(np.float32)


def dedup_overlap(pairs: list[tuple[np.ndarray, float]]) -> list[tuple[np.ndarray, float]]:
  """Ordered Gram-Schmidt so a subspace shared by several categories is ablated
  once, at the greatest factor among the categories that share it.

  Process directions highest-factor first; each lower-factor direction keeps only
  the component orthogonal to the directions already taken, ablated at its own
  factor. The shared part is therefore ablated solely by the highest-factor
  direction. pairs: [(unit_vector, factor), ...]; returns [(residual_unit, factor), ...]."""
  ordered = sorted(pairs, key=lambda pair: -pair[1])
  basis: list[np.ndarray] = []
  result: list[tuple[np.ndarray, float]] = []
  for vector, factor in ordered:
    residual = vector.astype(np.float32).copy()
    for prior in basis:
      residual = residual - float(residual @ prior) * prior
    norm = float(np.linalg.norm(residual))
    if norm <= 1e-6:
      continue  # fully covered by a higher-factor direction
    unit = (residual / norm).astype(np.float32)
    result.append((unit, factor))
    basis.append(unit)
  return result


def directions_for_layer(recipe: dict, hidden_index: int) -> list[tuple[np.ndarray, float]]:
  """Directions active at a hidden-state index:
  phase-A (onset..split): per-category direction for this layer, each at its own
    factor (factor_a_per_category, falling back to factor_a), overlap-deduplicated.
  phase-B (split..last):  single shared direction.
  Returns [(vector, factor), ...]."""
  onset, split, last = recipe["onset"], recipe["split"], recipe["last_layer"]
  factor_a, factor_b = recipe["factor_a"], recipe["factor_b"]
  per_category_factor = recipe.get("factor_a_per_category") or {}
  result: list[tuple[np.ndarray, float]] = []

  for mode_data in recipe["modes"].values():
    if onset <= hidden_index <= split:
      layer_offset = hidden_index - onset
      pairs: list[tuple[np.ndarray, float]] = []
      for category_id, per_layer in mode_data["phase_a"]["per_category"].items():
        arr = np.array(per_layer, dtype=np.float32)
        vec = arr[layer_offset] if arr.ndim == 2 else arr
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
          factor = float(per_category_factor.get(category_id, factor_a))
          pairs.append((vec / norm, factor))
      result.extend(dedup_overlap(pairs))
    elif split < hidden_index <= last:
      result.append((np.array(mode_data["phase_b"]["direction"], dtype=np.float32), factor_b))
  return result


def build_recipe(run: dict, model_id: str, gen_mode: str, onset: int, split: int,
                 factor_a: float, factor_b: float, state_dir,
                 factor_a_per_category: dict[str, float] | None = None) -> dict:
  """Build the two-phase abliteration recipe.
  Phase A (onset..split): one merged direction per category per layer
    (mean of hard + redirect unit vectors, renormalized).
  Phase B (split..last):  single shared direction (mean over all merged directions)."""
  per_category = load_directions(run, model_id, gen_mode, state_dir)
  if not per_category:
    raise ValueError("no categories with directions")

  sample_mode = next(iter(next(iter(per_category.values()))["by_mode"].values()))
  last_layer = len(sample_mode["direction_per_layer"]) - 1

  merged_per_category: dict[str, np.ndarray] = {}
  for category_id, cat_result in per_category.items():
    by_mode = cat_result["by_mode"]
    hard  = by_mode.get("hard")
    redir = by_mode.get("redirect")

    if hard and redir:
      raw = (
        np.array(hard["direction_per_layer"],  dtype=np.float32) +
        np.array(redir["direction_per_layer"], dtype=np.float32)
      )
    elif hard:
      raw = np.array(hard["direction_per_layer"],  dtype=np.float32)
    elif redir:
      raw = np.array(redir["direction_per_layer"], dtype=np.float32)
    else:
      continue

    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    merged_per_category[category_id] = (raw / norms).astype(np.float32)

  if not merged_per_category:
    raise ValueError("no categories produced merged directions")

  shared = shared_direction(list(merged_per_category.values()), split, last_layer)

  return {
    "run_id": run["run_id"], "model_id": model_id, "gen_mode": gen_mode,
    "onset": onset, "split": split, "last_layer": last_layer,
    "factor_a": factor_a, "factor_b": factor_b,
    "factor_a_per_category": {
      cat_id: float(factor)
      for cat_id, factor in (factor_a_per_category or {}).items()
      if cat_id in merged_per_category
    },
    "modes": {
      "merged": {
        "phase_a": {
          "layers": [onset, split],
          "per_category": {
            cat_id: dpl[onset:split + 1, :].tolist()
            for cat_id, dpl in merged_per_category.items()
          },
        },
        "phase_b": {"layers": [split, last_layer], "direction": shared.tolist()},
      }
    },
    "built_at": datetime.now(timezone.utc).isoformat(),
  }
