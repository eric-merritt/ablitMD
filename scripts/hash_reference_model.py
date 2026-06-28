"""Hash all projection weights of the reference (unabliterated) model.

Writes data/layerHash.json. Run this once against the clean model before
any ablation so the bake endpoint can detect dirty pre-bake state.

Usage (on VastAI, from the ablitMD project root):
  uv run python scripts/hash_reference_model.py os.path.join(os.path.expanduser("~"),"models")/Qwen3.6-27B
"""

import argparse
import hashlib
import json
import sys
import torch
from datetime import datetime, timezone
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

OUT_PATH = _root / "data" / "layerHash_ablit.json"

PROJ_SUFFIXES = (
  ".self_attn.o_proj.weight",
  ".linear_attn.out_proj.weight",
  ".mlp.down_proj.weight",
)

def proj_keys_from_index(index_path: Path) -> list[str]:
  idx = json.loads(index_path.read_text())
  return [
    k for k in idx["weight_map"]
    if any(k.endswith(s) for s in PROJ_SUFFIXES)
  ]

def hash_tensor_bytes(model_path: Path, shard_map: dict, key: str) -> str:
  from safetensors import safe_open
  shard = shard_map[key]
  with safe_open(str(model_path / shard), framework="pt") as f:
    tensor = f.get_tensor(key)
  return hashlib.sha256(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model_path", help="path to clean reference model directory")
  args = parser.parse_args()

  model_path = Path(args.model_path)
  index_path = model_path / "model.safetensors.index.json"
  if not index_path.exists():
    print(f"No model.safetensors.index.json found at {model_path}")
    sys.exit(1)

  idx = json.loads(index_path.read_text())
  shard_map = idx["weight_map"]
  keys = [k for k in shard_map if any(k.endswith(s) for s in PROJ_SUFFIXES)]
  print(f"Hashing {len(keys)} projection tensors from {model_path}")

  hashes: dict[str, str] = {}
  for i, key in enumerate(sorted(keys)):
    hashes[key] = hash_tensor_bytes(model_path, shard_map, key)
    if (i + 1) % 20 == 0:
      print(f"  {i + 1}/{len(keys)}...", flush=True)

  out = {
    "model_path": str(model_path),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "hashes": hashes,
  }
  OUT_PATH.write_text(json.dumps(out, indent=2))
  print(f"Wrote {OUT_PATH} ({len(hashes)} hashes)")

if __name__ == "__main__":
  main()
