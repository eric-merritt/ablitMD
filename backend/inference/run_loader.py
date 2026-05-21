from pathlib import Path

import numpy as np

from backend.inference.direction import compute_run_directions


def rebuild_directions(run: dict, model_id: str, gen_mode: str, state_dir: Path) -> dict:
  """Rebuild per-category direction_per_layer / magnitude_per_layer from .npy files.

  Returns {category_id: <compute_run_directions result>} — the result keeps the raw
  direction_per_layer vectors (unlike the stripped data on disk).
  """
  hidden: dict[str, np.ndarray] = {}
  by_category: dict[str, list[dict]] = {}

  for prompt in run["prompts"]:
    result = prompt.get("model_results", {}).get(model_id, {}).get(gen_mode)
    if not result:
      continue
    key = result["hidden_states_key"]
    npy_path = state_dir / f"{key}.npy"
    if not npy_path.exists():
      continue
    hidden[key] = np.load(str(npy_path))
    by_category.setdefault(prompt["category"], []).append({
      "hidden_states_key": key,
      "refused": result["refused"],
      "refusal_mode": result.get("refusal_mode", "hard" if result["refused"] else "none"),
    })

  per_category: dict[str, dict] = {}
  for category, classifications in by_category.items():
    cat_hidden = {c["hidden_states_key"]: hidden[c["hidden_states_key"]] for c in classifications}
    cat_result = compute_run_directions(cat_hidden, classifications, category=category)
    if cat_result:
      per_category[category] = cat_result
  return per_category
