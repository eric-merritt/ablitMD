"""Stage 2 — build an abliteration recipe.

Writes data/runs/<run_id>.recipe.json. CPU only.

Two methods:
  classic (default): two-phase per-category directions.
    uv run python scripts/build_recipe.py <run_id> --onset 38 --split 50 \
      --factor-a 0.15 --factor-b 0.15

  som-md: SOM-based multi-directional ablation (SOM-MD paper).
    uv run python scripts/build_recipe.py <run_id> --method som-md \
      --k 7 --grid 4,4 --factor 1.0
"""

import argparse
import json
import sys
from pathlib import Path

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parents[1]))
sys.path.insert(0, str(_here.parent))

from _run_picker import resolve_run_id
from backend.inference.recipe import build_recipe, recipe_filename, build_som_md_recipe

RUNS_DIR = _here.parents[1] / "data" / "runs"


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("run_id", nargs="?")
  parser.add_argument("--method", choices=["classic", "som-md"], default="classic",
                      help="Recipe method: classic (two-phase) or som-md (SOM multi-directional)")
  # Classic args
  parser.add_argument("--onset", type=int, default=None)
  parser.add_argument("--split", type=int, default=None)
  parser.add_argument("--last-layer", type=int, default=None,
                      help="Override last ablation layer (default: all layers in model)")
  parser.add_argument("--factor-a", type=float, default=0.15)
  parser.add_argument("--factor-b", type=float, default=0.15)
  parser.add_argument("--factor-a-per-category", type=json.loads, default=None,
                      help='JSON map of category_id -> factor, e.g. \'{"cbrn":1.5}\'')
  # SOM-MD args
  parser.add_argument("--k", type=int, default=7,
                      help="Number of SOM directions (default: 7)")
  parser.add_argument("--grid", type=str, default="4,4",
                      help="SOM grid shape as rows,cols (default: 4,4)")
  parser.add_argument("--factor", type=float, default=1.95,
                      help="Ablation factor for SOM-MD directions (default: 1.95)")
  args = parser.parse_args()
  run_id = resolve_run_id(args.run_id)

  run = json.loads((RUNS_DIR / f"{run_id}.json").read_text())
  state_dir = RUNS_DIR / run_id
  step = run["sequence"][0]  # v1: single-model runs — use the first sequence step

  if args.method == "som-md":
    rows, cols = (int(x) for x in args.grid.split(","))
    recipe = build_som_md_recipe(
        run, step["model"], step["mode"],
        k=args.k, grid_shape=(rows, cols),
        factor=args.factor, state_dir=state_dir,
    )
  else:
    if args.onset is None or args.split is None:
      parser.error("--onset and --split are required for classic method")
    recipe = build_recipe(run, step["model"], step["mode"], args.onset, args.split,
                          args.factor_a, args.factor_b, state_dir,
                          factor_a_per_category=args.factor_a_per_category,
                          last_layer=args.last_layer)

  out_path = RUNS_DIR / recipe_filename(run_id)
  out_path.write_text(json.dumps(recipe))
  print(f"[recipe] wrote {out_path.name} ({out_path.stat().st_size / 1048576:.2f} MB)")


if __name__ == "__main__":
  main()
