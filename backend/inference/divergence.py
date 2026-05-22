import numpy as np

from backend.inference.run_loader import rebuild_directions


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


def detect_onset(magnitude: np.ndarray, frac: float = 0.10) -> int:
  """First layer where mean direction magnitude reaches frac * peak."""
  peak = float(magnitude.max())
  if peak <= 0:
    return 0
  above = np.where(magnitude >= frac * peak)[0]
  return int(above[0]) if len(above) else 0


def detect_split_by_mode(magnitude: np.ndarray, onset: int, mode_val: int) -> int:
  """First layer after onset+1 where rounded magnitude equals mode_val.
  Falls back to argmax if mode_val never appears."""
  tail = np.round(magnitude[onset:]).astype(int)
  matches = np.where(tail == mode_val)[0]
  if len(matches):
    return onset + int(matches[0]) + 1
  return onset + int(np.argmax(magnitude[onset:])) + 1


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


def analyze_run(run: dict, model_id: str, gen_mode: str, state_dir) -> dict:
  """Per refusal mode (hard, redirect): clumping curve, magnitude curve, and
  suggested onset/divergence layers. Returns {mode: {...}} for modes with data."""
  per_category = rebuild_directions(run, model_id, gen_mode, state_dir)
  out: dict[str, dict] = {}

  for refusal_mode in ("hard", "redirect"):
    directions, magnitudes, category_ids = [], [], []
    for category_id, cat_result in per_category.items():
      entry = cat_result["by_mode"].get(refusal_mode)
      if not entry:
        continue
      directions.append(np.array(entry["direction_per_layer"], dtype=np.float32))
      magnitudes.append(np.array(entry["magnitude_per_layer"], dtype=np.float32))
      category_ids.append(category_id)
    if not directions:
      continue

    clumping = clumping_curve(directions)
    magnitude = mean_magnitude_curve(magnitudes)
    onset = detect_onset(magnitude)
    divergence = detect_divergence(clumping, onset)
    all_rounded = np.round(np.concatenate(magnitudes)).astype(int)
    values, counts = np.unique(all_rounded, return_counts=True)
    mode_val = int(values[np.argmax(counts)])
    per_cat_splits = [detect_split_by_mode(m, onset, mode_val) for m in magnitudes]
    split = int(round(sum(per_cat_splits) / len(per_cat_splits)))
    out[refusal_mode] = {
      "clumping": clumping.tolist(),
      "magnitude": magnitude.tolist(),
      "category_ids": category_ids,
      "suggested_onset": onset,
      "suggested_divergence": divergence,
      "suggested_split": split,
    }
  return out
