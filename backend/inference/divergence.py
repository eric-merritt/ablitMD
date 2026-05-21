import numpy as np


def clumping_curve(directions: list[np.ndarray]) -> np.ndarray:
  """directions: list of (n_layers, dim) per-category unit-direction arrays.
  Returns (n_layers,) clumping = squared length of the per-layer centroid.
  1.0 = every category points the same way; 0.0 = fully fanned out."""
  stack = np.stack(directions, axis=0)        # (n_cat, n_layers, dim)
  centroid = stack.mean(axis=0)               # (n_layers, dim)
  return (centroid ** 2).sum(axis=1).astype(np.float32)


def mean_magnitude_curve(magnitudes: list[np.ndarray]) -> np.ndarray:
  """magnitudes: list of (n_layers,) per-category magnitude arrays. Returns (n_layers,) mean."""
  return np.stack(magnitudes, axis=0).mean(axis=0).astype(np.float32)


def detect_onset(magnitude: np.ndarray, frac: float = 0.25) -> int:
  """First layer where mean direction magnitude reaches frac * peak."""
  peak = float(magnitude.max())
  if peak <= 0:
    return 0
  above = np.where(magnitude >= frac * peak)[0]
  return int(above[0]) if len(above) else 0


def detect_divergence(clumping: np.ndarray, onset: int, retain: float = 0.90) -> int:
  """First layer >= onset where clumping drops below retain * (post-onset peak).
  Falls back to the last layer index if the curve never drops that far."""
  tail = clumping[onset:]
  if len(tail) == 0:
    return onset
  threshold = retain * float(tail.max())
  for layer in range(onset, len(clumping)):
    if clumping[layer] < threshold:
      return layer
  return len(clumping) - 1
