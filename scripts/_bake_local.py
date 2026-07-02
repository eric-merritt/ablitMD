"""Local bake driver — memory-lean variant of the /ablate/bake (ablitmd) path.

Loads the base model with device_map="auto" (GPU when available), applies the latest
recipe in place WITHOUT retaining f32 snapshot clones (a bake never needs to restore),
then saves the baked variant to ~/models/<family>/<name>-Ablit.

Usage:
  uv run python scripts/_bake_local.py [run_id]
"""

import json
import sys
from pathlib import Path

import transformers
import torch
from transformers import AutoConfig, AutoTokenizer

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parents[1]))
sys.path.insert(0, str(_here.parent))

from _run_picker import resolve_run_id
from backend.inference.ablation import (
  orthogonalize_weight, _decoder_layers, _projection_modules,
)
from backend.inference.recipe import latest_recipe_path, directions_for_layer

RUNS_DIR = _here.parents[1] / "data" / "runs"
MODELS_ROOT = Path.home() / "models"


def model_dirs(model_id: str) -> tuple[str, str]:
  """Map an org/name model id to its on-disk (base, abliterated) dirs:
  ~/models/<family>/<name>-BaseModel and ~/models/<family>/<name>-Ablit."""
  family, _, name = model_id.rpartition("/")
  family_dir = MODELS_ROOT / family if family else MODELS_ROOT
  return str(family_dir / f"{name}-BaseModel"), str(family_dir / f"{name}-Ablit")


def ablate(recipe: dict, model) -> None:
  """Mirror apply_ablation_in_place's math without snapshot clones.
  Decoder layer i is hidden_index i+1 (layer 0 / MTP head is left alone)."""
  layers = _decoder_layers(model)
  logged = False
  with torch.no_grad():
    for decoder_idx in range(len(layers)):
      raw = directions_for_layer(recipe, hidden_index=decoder_idx + 1)
      if not raw:
        continue
      for proj in _projection_modules(layers[decoder_idx]):
        device = proj.weight.device
        weight_f32 = proj.weight.data.to(torch.float32)
        for vector, factor in raw:
          direction = torch.tensor(vector, dtype=torch.float32, device=device)
          orthogonalize_weight(weight_f32, direction, float(factor))
        proj.weight.data = weight_f32.to(torch.bfloat16)
        if not logged:
          logged = True
          print(f"[ablation] first edit layer={decoder_idx + 1} proj={type(proj).__name__} "
                f"shape={tuple(proj.weight.shape)}", flush=True)


def main():
  run_id = resolve_run_id(sys.argv[1] if len(sys.argv) > 1 else None)
  run_data = json.loads((RUNS_DIR / f"{run_id}.json").read_text())
  model_id = run_data["models"][0]
  base_dir, ablit_dir = model_dirs(model_id)
  source = sys.argv[2] if len(sys.argv) > 2 else base_dir
  ablit_dir = sys.argv[3] if len(sys.argv) > 3 else ablit_dir

  ablit_path = Path(ablit_dir)
  if ablit_path.exists() and any(ablit_path.iterdir()):
    raise SystemExit(f"[bake] abort: target already exists and is non-empty: {ablit_dir}")

  cfg = AutoConfig.from_pretrained(source)
  arch = (cfg.architectures or [None])[0]
  model_class = getattr(transformers, arch) if arch else None
  if model_class is None:
    raise SystemExit(f"[bake] cannot resolve model class for architecture: {arch}")
  print(f"[bake] loading {source} as {arch} (bf16, eager attn)", flush=True)
  model = model_class.from_pretrained(
    source, dtype=torch.bfloat16, device_map="auto",
    low_cpu_mem_usage=True, attn_implementation="eager",
  )
  model.eval()
  tokenizer = AutoTokenizer.from_pretrained(source)
  declared_vocab = (getattr(cfg, "text_config", None) or cfg).vocab_size
  if len(tokenizer) < declared_vocab // 2:
    raise SystemExit(
      f"[bake] abort: tokenizer loaded only {len(tokenizer)} tokens but config declares "
      f"{declared_vocab} — tokenizer.json is missing or corrupt in {source}"
    )

  recipe_path = latest_recipe_path(RUNS_DIR, run_id)
  if recipe_path is None:
    raise SystemExit("no recipe found for run")
  recipe = json.loads(recipe_path.read_text())
  print(f"[bake] recipe {recipe_path.name}: onset={recipe['onset']} split={recipe['split']} "
        f"last={recipe['last_layer']} factor_a={recipe['factor_a']} factor_b={recipe['factor_b']}",
        flush=True)

  ablate(recipe, model)
  print(f"[bake] saving to {ablit_dir}", flush=True)
  model.save_pretrained(ablit_dir, max_shard_size="3GB")
  tokenizer.save_pretrained(ablit_dir)
  print(f"[bake] saved_to {ablit_dir}", flush=True)


if __name__ == "__main__":
  main()
