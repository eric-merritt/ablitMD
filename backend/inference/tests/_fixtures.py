import json
from pathlib import Path

import numpy as np

HIDDEN = 16
N_LAYERS = 12  # hidden-state indices 0..11


def make_mini_run(root: Path, run_id: str = "run_test") -> dict:
  """Write a run JSON + .npy hidden states under root/data/runs/.
     Two categories, each with none/hard/redirect prompts. Returns path + meta."""
  runs_dir = root / "data" / "runs"
  state_dir = runs_dir / run_id
  state_dir.mkdir(parents=True, exist_ok=True)
  model_id, gen_mode = "Test/Model", "non_thinking"
  rng = np.random.default_rng(0)

  specs = [
    ("cat_a", "none"), ("cat_a", "none"), ("cat_a", "hard"), ("cat_a", "hard"),
    ("cat_a", "redirect"), ("cat_a", "redirect"),
    ("cat_b", "none"), ("cat_b", "none"), ("cat_b", "hard"), ("cat_b", "hard"),
    ("cat_b", "redirect"), ("cat_b", "redirect"),
  ]
  prompts = []
  for idx, (category, refusal_mode) in enumerate(specs):
    key = f"p{idx}__Test__Model__{gen_mode}"
    np.save(str(state_dir / f"{key}.npy"),
            rng.standard_normal((N_LAYERS, HIDDEN)).astype(np.float32))
    prompts.append({
      "prompt_id": f"p{idx}", "text": f"prompt {idx}",
      "category": category, "category_group": category, "type": "harmful",
      "triggers": [],
      "model_results": {model_id: {gen_mode: {
        "response": "...", "hidden_states_key": key,
        "refused": refusal_mode != "none", "refusal_mode": refusal_mode,
      }}},
    })

  run = {
    "run_id": run_id, "models": [model_id], "mode_selection": gen_mode,
    "sequence": [{"model": model_id, "mode": gen_mode}], "prompts": prompts,
  }
  (runs_dir / f"{run_id}.json").write_text(json.dumps(run))
  return {"runs_dir": runs_dir, "state_dir": state_dir, "run": run,
          "run_id": run_id, "model_id": model_id, "gen_mode": gen_mode}
