"""Splice tensors the abliteration bake dropped (MTP head, vision tower) back into an
abliterated model from the original. Copies every key the abliterated index is missing
(optionally filtered by --prefix) into a new shard and registers it, so a re-convert to
GGUF is whole again.

Usage:
  uv run python scripts/splice_mtp_head.py \
    /home/ermer/models/Qwen/Qwen3.6-27B \
    /home/ermer/models/Qwen/Qwen3.6-27B-Ablit-v1.0 \
    --shard-name model-restored.safetensors          # all missing keys
  ... --prefix mtp.                                   # just the MTP head
  ... --prefix model.visual.                          # just the vision tower
"""

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def load_index(model_path: Path) -> dict:
  return json.loads((model_path / "model.safetensors.index.json").read_text())


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("orig_path", help="complete source model")
  parser.add_argument("ablit_path", help="abliterated model missing tensors")
  parser.add_argument("--shard-name", default="model-restored.safetensors")
  parser.add_argument("--prefix", default="", help="only splice keys with this prefix")
  args = parser.parse_args()

  orig_path = Path(args.orig_path)
  ablit_path = Path(args.ablit_path)

  orig_map = load_index(orig_path)["weight_map"]
  ablit_index = load_index(ablit_path)
  ablit_map = ablit_index["weight_map"]

  missing = sorted(
    key for key in orig_map
    if key not in ablit_map and key.startswith(args.prefix)
  )
  if not missing:
    print("No missing keys — nothing to do.")
    return

  print(f"Splicing {len(missing)} tensors from {orig_path.name} -> {args.shard_name}")
  tensors = {}
  for key in missing:
    with safe_open(str(orig_path / orig_map[key]), framework="pt") as handle:
      tensors[key] = handle.get_tensor(key)

  save_file(tensors, str(ablit_path / args.shard_name))
  print(f"Wrote {ablit_path / args.shard_name}")

  for key in missing:
    ablit_map[key] = args.shard_name
  metadata = ablit_index.setdefault("metadata", {})
  if "total_size" in metadata:
    metadata["total_size"] += sum(t.numel() * t.element_size() for t in tensors.values())
  (ablit_path / "model.safetensors.index.json").write_text(json.dumps(ablit_index, indent=2))
  print("Updated model.safetensors.index.json")


if __name__ == "__main__":
  main()
