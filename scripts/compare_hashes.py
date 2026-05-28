"""Compare projection weight hashes between original and abliterated model.

Reads data/layerHash.json (original) and data/layerHash_ablit.json (abliterated)
and prints which layers changed, which are identical, and which are missing.

Usage:
  python scripts/compare_hashes.py
"""

import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]

orig_path  = _root / "data" / "layerHash.json"
ablit_path = _root / "data" / "layerHash_ablit.json"

if not orig_path.exists():
  print(f"Missing {orig_path} — run hash_reference_model.py on the original first")
  sys.exit(1)
if not ablit_path.exists():
  print(f"Missing {ablit_path} — run hash_reference_model.py on the abliterated model first")
  sys.exit(1)

orig_hashes  = json.loads(orig_path.read_text())["hashes"]
ablit_hashes = json.loads(ablit_path.read_text())["hashes"]

all_keys = sorted(set(orig_hashes) | set(ablit_hashes))

changed, identical, only_orig, only_ablit = [], [], [], []

for key in all_keys:
  in_orig  = key in orig_hashes
  in_ablit = key in ablit_hashes
  if in_orig and in_ablit:
    if orig_hashes[key] == ablit_hashes[key]:
      identical.append(key)
    else:
      changed.append(key)
  elif in_orig:
    only_orig.append(key)
  else:
    only_ablit.append(key)

print(f"\n=== HASH COMPARISON ===")
print(f"  identical : {len(identical)}")
print(f"  changed   : {len(changed)}")
print(f"  only orig : {len(only_orig)}")
print(f"  only ablit: {len(only_ablit)}")

if changed:
  print(f"\nCHANGED layers:")
  for k in changed:
    print(f"  {k}")

if only_orig:
  print(f"\nOnly in original (not in abliterated):")
  for k in only_orig:
    print(f"  {k}")

if only_ablit:
  print(f"\nOnly in abliterated (not in original):")
  for k in only_ablit:
    print(f"  {k}")
