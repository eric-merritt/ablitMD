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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.inference.direction import compute_run_directions
from backend.inference.generator import stream_prompt
from backend.inference.model_loader import (
  get_loaded_model_id,
  get_model,
  load_model,
  unload_model,
)
from backend.inference.ablation import set_ablation, clear_ablation, ablation_status
from backend.inference.verify import looks_like_refusal, projection_strength
from backend.inference.generator import run_prompt

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


class AblateRequest(BaseModel):
  run_id: str


class VerifyRequest(BaseModel):
  run_id: str
  model_id: str
  gen_mode: str
  categories: list[str] | None = None


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

  def ndjson_events():
    for event in stream_prompt(
      prompt_text=req.prompt_text,
      mode=req.mode,
      run_id=req.run_id,
      hidden_states_key=hidden_states_key,
      runs_dir=RUNS_DIR,
    ):
      yield json.dumps(event) + "\n"

  return StreamingResponse(ndjson_events(), media_type="application/x-ndjson")


@app.post("/compute")
def compute(req: ComputeRequest):
  run_file = RUNS_DIR / f"{req.run_id}.json"
  if not run_file.exists():
    raise HTTPException(status_code=404, detail="Run not found")

  run_data = json.loads(run_file.read_text())
  state_dir = RUNS_DIR / req.run_id

  all_hidden: dict[str, np.ndarray] = {}
  all_meta: dict[str, dict] = {}

  for prompt in run_data["prompts"]:
    result = prompt.get("model_results", {}).get(req.model_id, {}).get(req.mode)
    if not result:
      continue
    key = result["hidden_states_key"]
    npy_path = state_dir / f"{key}.npy"
    if not npy_path.exists():
      continue
    all_hidden[key] = np.load(str(npy_path))
    all_meta[key] = {
      "category": prompt["category"],
      "triggers": prompt.get("triggers", []),
      "refused": result["refused"],
      "refusal_mode": result.get("refusal_mode", "hard" if result["refused"] else "none"),
    }

  per_category: dict[str, list] = {}
  for key, meta in all_meta.items():
    per_category.setdefault(meta["category"], []).append({
      "hidden_states_key": key,
      "refused": meta["refused"],
      "refusal_mode": meta["refusal_mode"],
    })

  direction_results: dict[str, dict] = {}

  for category, classifications in per_category.items():
    cat_hidden = {c["hidden_states_key"]: all_hidden[c["hidden_states_key"]] for c in classifications}

    visitors = {
      key: all_hidden[key]
      for key, meta in all_meta.items()
      if meta["category"] != category and category in meta["triggers"]
    }
    trigger_meta = {
      key: {
        "source_category": all_meta[key]["category"],
        "refused": all_meta[key]["refused"],
        "refusal_mode": all_meta[key]["refusal_mode"],
      }
      for key in visitors
    }

    result = compute_run_directions(cat_hidden, classifications, category=category, visitors=visitors or None)
    if result:
      result["trigger_meta"] = trigger_meta
      direction_results[category] = result

  # raw per-layer direction vectors are large (~150 MB+) and read by no chart;
  # drop them so the response stays small enough to PATCH back into the run.
  for cat_result in direction_results.values():
    for mode_data in cat_result.get("by_mode", {}).values():
      mode_data.pop("direction_per_layer", None)

  return direction_results


@app.post("/ablate")
def ablate(req: AblateRequest):
  if get_loaded_model_id() is None:
    raise HTTPException(status_code=400, detail="No model loaded")
  recipe_path = RUNS_DIR / f"{req.run_id}.recipe.json"
  if not recipe_path.exists():
    raise HTTPException(status_code=404, detail="Recipe not found — build it first")
  set_ablation(json.loads(recipe_path.read_text()), get_model())
  return ablation_status()


@app.post("/ablate/clear")
def ablate_clear():
  clear_ablation()
  return ablation_status()


@app.get("/ablate/status")
def ablate_status_endpoint():
  return ablation_status()


@app.post("/ablate/verify")
def ablate_verify(req: VerifyRequest):
  if get_loaded_model_id() is None:
    raise HTTPException(status_code=400, detail="No model loaded")
  recipe_path = RUNS_DIR / f"{req.run_id}.recipe.json"
  if not recipe_path.exists():
    raise HTTPException(status_code=404, detail="Recipe not found")
  recipe = json.loads(recipe_path.read_text())

  run_data = json.loads((RUNS_DIR / f"{req.run_id}.json").read_text())
  state_dir = RUNS_DIR / req.run_id
  phase_b_range = tuple(next(iter(recipe["modes"].values()))["phase_b"]["layers"])

  by_category: dict[str, list[dict]] = {}
  for prompt in run_data["prompts"]:
    if req.categories and prompt["category"] not in req.categories:
      continue
    result = prompt.get("model_results", {}).get(req.model_id, {}).get(req.gen_mode)
    if result:
      by_category.setdefault(prompt["category"], []).append({"prompt": prompt, "result": result})

  def events():
    for category, items in by_category.items():
      hard_phase_b = recipe["modes"].get("hard", {}).get("phase_b", {}).get("per_category", {})
      direction = np.array(hard_phase_b.get(category, [0.0]), dtype=np.float32)

      before_proj, after_proj, refused_before, refused_after, sample = [], [], 0, 0, ""
      for item in items:
        key = item["result"]["hidden_states_key"]
        before_npy = state_dir / f"{key}.npy"
        if before_npy.exists() and direction.shape[0] > 1:
          before_proj.append(projection_strength(np.load(str(before_npy)), direction, phase_b_range))
        refused_before += 1 if item["result"].get("refused") else 0

        response = run_prompt(item["prompt"]["text"], req.gen_mode, req.run_id,
                              f"verify__{key}", RUNS_DIR)
        after_npy = RUNS_DIR / req.run_id / f"verify__{key}.npy"
        if after_npy.exists() and direction.shape[0] > 1:
          after_proj.append(projection_strength(np.load(str(after_npy)), direction, phase_b_range))
        if looks_like_refusal(response):
          refused_after += 1
        sample = sample or response

      count = max(len(items), 1)
      yield json.dumps({
        "category": category,
        "projection_before": float(np.mean(before_proj)) if before_proj else 0.0,
        "projection_after": float(np.mean(after_proj)) if after_proj else 0.0,
        "refusal_rate_before": refused_before / count,
        "refusal_rate_after": refused_after / count,
        "sample_response": sample[:400],
      }) + "\n"

  return StreamingResponse(events(), media_type="application/x-ndjson")


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8238, reload=False)
