import numpy as np
from datetime import timezone, datetime


def compute_direction(
  refusal_states: np.ndarray,
  non_refusal_states: np.ndarray
) -> np.ndarray:
  refusal_mean = refusal_states.mean(axis=0)
  non_refusal_mean = non_refusal_states.mean(axis=0)
  diff = refusal_mean - non_refusal_mean
  norms = np.linalg.norm(diff, axis=1, keepdims=True)
  safe_norms = np.where(norms < 1e-8, 1.0, norms)
  direction = np.where(norms < 1e-8, np.zeros_like(diff), diff / safe_norms)
  return direction.astype(np.float32)


def compute_similarity(
  hidden_state: np.ndarray,
  direction: np.ndarray
) -> np.ndarray:
  hs_norms = np.linalg.norm(hidden_state, axis=1, keepdims=True)
  dir_norms = np.linalg.norm(direction, axis=1, keepdims=True)
  safe_hs = np.where(hs_norms < 1e-8, 1.0, hs_norms)
  safe_dir = np.where(dir_norms < 1e-8, 1.0, dir_norms)
  hs_norm = hidden_state / safe_hs
  dir_norm = direction / safe_dir
  return np.clip((hs_norm * dir_norm).sum(axis=1), -1.0, 1.0).astype(np.float32)


def compute_run_directions(
  hidden_states: dict[str, np.ndarray],
  classifications: list[dict],
  category: str
) -> dict | None:
  refused_keys = [c['hidden_states_key'] for c in classifications if c['refused']]
  non_refused_keys = [c['hidden_states_key'] for c in classifications if not c['refused']]

  if len(refused_keys) == 0 or len(non_refused_keys) == 0:
    return None

  refused_stack = np.stack([hidden_states[key] for key in refused_keys if key in hidden_states])
  non_refused_stack = np.stack([hidden_states[key] for key in non_refused_keys if key in hidden_states])

  if refused_stack.shape[0] == 0 or non_refused_stack.shape[0] == 0:
    return None

  direction = compute_direction(refused_stack, non_refused_stack)

  similarity_per_prompt = {}
  for classification in classifications:
    key = classification['hidden_states_key']
    if key in hidden_states:
      sims = compute_similarity(hidden_states[key], direction)
      similarity_per_prompt[key] = sims.tolist()

  return {
    'computed_at': datetime.now(timezone.utc).isoformat(),
    'direction_per_layer': direction.tolist(),
    'similarity_per_prompt': similarity_per_prompt,
  }
