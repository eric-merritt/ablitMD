"""Stage 2 — build an abliteration recipe.

Writes data/runs/<run_id>.recipe.json. CPU only.

Usage:
  uv run python scripts/build_recipe.py <run_id> --onset 38 --split 50 \
    --factor-a 0.15 --factor-b 0.15
"""

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from backend.inference.recipe import build_recipe, recipe_filename

RUNS_DIR = _root / "data" / "runs"


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("run_id")
  parser.add_argument("--onset", type=int, required=True)
  parser.add_argument("--split", type=int, required=True)
  parser.add_argument("--factor-a", type=float, default=0.15)
  parser.add_argument("--factor-b", type=float, default=0.15)
  parser.add_argument("--factor-a-per-category", type=json.loads, default=None,
                      help='JSON map of category_id -> factor, e.g. \'{"cbrn":1.5}\'')
  args = parser.parse_args()

  run = json.loads((RUNS_DIR / f"{args.run_id}.json").read_text())
  state_dir = RUNS_DIR / args.run_id
  step = run["sequence"][0]  # v1: single-model runs — use the first sequence step

  recipe = build_recipe(run, step["model"], step["mode"], args.onset, args.split,
                        args.factor_a, args.factor_b, state_dir,
                        factor_a_per_category=args.factor_a_per_category)
  out_path = RUNS_DIR / recipe_filename(args.run_id)
  out_path.write_text(json.dumps(recipe))
  print(f"[recipe] wrote {out_path.name} ({out_path.stat().st_size / 1048576:.2f} MB)")


if __name__ == "__main__":
  main()
