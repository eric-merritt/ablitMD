import numpy as np


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


def phase_b_direction(direction_per_layer: np.ndarray, split: int, last: int) -> np.ndarray:
  """One category's direction: mean of its unit vectors in [split..last], renormalized."""
  window = direction_per_layer[split:last + 1, :]
  return _renormalize(window.mean(axis=0)).astype(np.float32)


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
      for vector in mode_data["phase_b"]["per_category"].values():
        result.append((np.array(vector, dtype=np.float32), factor_b))
  return result
