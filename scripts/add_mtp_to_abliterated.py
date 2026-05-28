"""Copy MTP projection weights from the original model into the abliterated model.

Creates a new shard (model-mtp.safetensors) in the abliterated model directory
and updates its model.safetensors.index.json to include those keys.

Usage:
  python scripts/add_mtp_to_abliterated.py \
    /home/ermer/models/Qwen/Qwen3.6-27B \
    /home/ermer/models/Qwen/Qwen3.6-27B-Abliterated/v0.1.0
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PROJ_SUFFIXES = (".self_attn.o_proj.weight", ".linear_attn.out_proj.weight", ".mlp.down_proj.weight")

def missing_proj_keys(orig_path: Path, ablit_path: Path) -> list[str]:
  orig_idx  = json.loads((orig_path  / "model.safetensors.index.json").read_text())["weight_map"]
  ablit_idx = json.loads((ablit_path / "model.safetensors.index.json").read_text())["weight_map"]
  orig_proj  = {k for k in orig_idx  if any(k.endswith(s) for s in PROJ_SUFFIXES)}
  ablit_proj = {k for k in ablit_idx if any(k.endswith(s) for s in PROJ_SUFFIXES)}
  return sorted(orig_proj - ablit_proj)

def load_tensor(orig_path: Path, shard_map: dict, key: str) -> torch.Tensor:
  with safe_open(str(orig_path / shard_map[key]), framework="pt") as f:
    return f.get_tensor(key)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("orig_path")
  parser.add_argument("ablit_path")
  args = parser.parse_args()

  orig_path  = Path(args.orig_path)
  ablit_path = Path(args.ablit_path)

  missing = missing_proj_keys(orig_path, ablit_path)
  if not missing:
    print("No missing projection keys — nothing to do.")
    sys.exit(0)

  print(f"Missing keys: {missing}")

  orig_idx  = json.loads((orig_path / "model.safetensors.index.json").read_text())["weight_map"]
  tensors = { k: load_tensor(orig_path, orig_idx, k) for k in missing }

  new_shard = ablit_path / "model-mtp.safetensors"
  save_file(tensors, str(new_shard))
  print(f"Wrote {new_shard}")

  ablit_index_path = ablit_path / "model.safetensors.index.json"
  ablit_index = json.loads(ablit_index_path.read_text())
  for k in missing:
    ablit_index["weight_map"][k] = "model-mtp.safetensors"
  ablit_index_path.write_text(json.dumps(ablit_index, indent=2))
  print(f"Updated {ablit_index_path}")

if __name__ == "__main__":
  main()
