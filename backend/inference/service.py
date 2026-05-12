import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `backend.*` imports resolve when the
# script is invoked directly (e.g. `uv run python backend/inference/service.py`).
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
  sys.path.insert(0, str(_project_root))

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.inference.direction import compute_run_directions
from backend.inference.generator import run_prompt
from backend.inference.model_loader import (
  get_loaded_model_id,
  load_model,
  unload_model,
)

app = FastAPI(title="ablitMD inference service")

RUNS_DIR = Path("./data/runs")


class LoadRequest(BaseModel):
  model_id: str
  api_model_id: str


class GenerateRequest(BaseModel):
  prompt_id: str
  prompt_text: str
  run_id: str
  model_id: str
  mode: str


class ComputeRequest(BaseModel):
  run_id: str
  model_id: str
  mode: str


@app.get("/status")
def status():
  return {"loaded_model": get_loaded_model_id()}


@app.post("/load")
def load(req: LoadRequest):
  load_model(req.model_id, req.api_model_id)
  return {"loaded_model": req.model_id}


@app.post("/generate")
def generate(req: GenerateRequest):
  if get_loaded_model_id() != req.model_id:
    raise HTTPException(status_code=400, detail=f"Model {req.model_id} not loaded")

  safe_model = req.model_id.replace("/", "__")
  hidden_states_key = f"{req.prompt_id}__{safe_model}__{req.mode}"

  response = run_prompt(
    prompt_text=req.prompt_text,
    mode=req.mode,
    run_id=req.run_id,
    hidden_states_key=hidden_states_key,
    runs_dir=RUNS_DIR,
  )

  return {"response": response, "hidden_states_key": hidden_states_key}


@app.post("/compute")
def compute(req: ComputeRequest):
  run_file = RUNS_DIR / f"{req.run_id}.json"
  if not run_file.exists():
    raise HTTPException(status_code=404, detail="Run not found")

  run_data = json.loads(run_file.read_text())
  state_dir = RUNS_DIR / req.run_id

  per_category: dict[str, list] = {}

  for prompt in run_data["prompts"]:
    result = prompt.get("model_results", {}).get(req.model_id, {}).get(req.mode)
    if not result:
      continue

    key = result["hidden_states_key"]
    npy_path = state_dir / f"{key}.npy"
    if not npy_path.exists():
      continue

    category = prompt["category"]
    per_category.setdefault(category, []).append({
      "hidden_states_key": key,
      "refused": result["refused"],
      "hidden_state": np.load(str(npy_path)),
    })

  direction_results: dict[str, dict] = {}

  for category, entries in per_category.items():
    hidden_states_map = {e["hidden_states_key"]: e["hidden_state"] for e in entries}
    classifications = [
      {"hidden_states_key": e["hidden_states_key"], "refused": e["refused"]}
      for e in entries
    ]
    result = compute_run_directions(hidden_states_map, classifications, category=category)
    if result:
      direction_results[category] = result

  return direction_results


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8200, reload=False)
