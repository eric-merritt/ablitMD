import numpy as np
from datetime import timezone, datetime


def compute_direction(
  refusal_states: np.ndarray,
  non_refusal_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  refusal_mean = refusal_states.mean(axis=0)
  non_refusal_mean = non_refusal_states.mean(axis=0)
  diff = refusal_mean - non_refusal_mean
  norms = np.linalg.norm(diff, axis=1, keepdims=True)
  safe_norms = np.where(norms < 1e-8, 1.0, norms)
  direction = np.where(norms < 1e-8, np.zeros_like(diff), diff / safe_norms)
  magnitude = norms.squeeze(axis=1)
  return direction.astype(np.float32), magnitude.astype(np.float32)


def compute_similarity(
  hidden_state: np.ndarray,
  direction: np.ndarray,
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
  category: str,
  visitors: dict[str, np.ndarray] | None = None,
) -> dict | None:
  groups: dict[str, list[str]] = {'none': [], 'hard': [], 'redirect': [], 'disclaimer': []}
  for c in classifications:
    key = c['hidden_states_key']
    if key not in hidden_states:
      continue
    mode = c.get('refusal_mode', 'none')
    if mode in groups:
      groups[mode].append(key)

  none_keys = groups['none']
  if not none_keys:
    return None

  none_stack = np.stack([hidden_states[k] for k in none_keys])

  by_mode: dict = {}
  for mode in ('hard', 'redirect', 'disclaimer'):
    mode_keys = groups[mode]
    if not mode_keys:
      continue

    mode_stack = np.stack([hidden_states[k] for k in mode_keys])
    direction, magnitude = compute_direction(mode_stack, none_stack)

    sims_per_prompt = {
      c['hidden_states_key']: compute_similarity(
        hidden_states[c['hidden_states_key']], direction
      ).tolist()
      for c in classifications
      if c['hidden_states_key'] in hidden_states
    }

    by_mode[mode] = {
      'direction_per_layer': direction.tolist(),
      'magnitude_per_layer': magnitude.tolist(),
      'sample_count': len(mode_keys),
      'similarity_per_prompt': sims_per_prompt,
    }

  if not by_mode:
    return None

  none_mean = none_stack.mean(axis=0)
  none_norms = np.linalg.norm(none_mean, axis=1, keepdims=True)
  safe_none_norms = np.where(none_norms < 1e-8, 1.0, none_norms)
  none_direction = np.where(none_norms < 1e-8, np.zeros_like(none_mean), none_mean / safe_none_norms).astype(np.float32)
  by_mode['none'] = {
    'direction_per_layer': none_direction.tolist(),
    'magnitude_per_layer': none_norms.squeeze(axis=1).astype(np.float32).tolist(),
    'sample_count': len(none_keys),
    'similarity_per_prompt': {
      c['hidden_states_key']: compute_similarity(
        hidden_states[c['hidden_states_key']], none_direction
      ).tolist()
      for c in classifications
      if c['hidden_states_key'] in hidden_states
    },
  }

  alignment: dict[str, list] = {}
  computed = list(by_mode.keys())
  for i, m1 in enumerate(computed):
    d1 = np.array(by_mode[m1]['direction_per_layer'])
    for m2 in computed[i + 1:]:
      d2 = np.array(by_mode[m2]['direction_per_layer'])
      alignment[f'{m1}_vs_{m2}'] = (d1 * d2).sum(axis=1).tolist()

  if visitors:
    for mode_data in by_mode.values():
      direction = np.array(mode_data['direction_per_layer'])
      mode_data['trigger_similarity_per_prompt'] = {
        key: compute_similarity(hs, direction).tolist()
        for key, hs in visitors.items()
      }

  return {
    'computed_at': datetime.now(timezone.utc).isoformat(),
    'by_mode': by_mode,
    'alignment': alignment,
  }
