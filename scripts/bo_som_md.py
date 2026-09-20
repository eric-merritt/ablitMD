#!/usr/bin/env python3
"""Closed-loop SOM-MD recipe builder.

Trains a hexagonal SOM on the run's harmful hidden states, runs Bayesian
Optimization over direction subsets + ablation factor scored by measured
compliance rate on held-out refused prompts, and writes the winning recipe in
the existing `som_md` schema (plus a "bo" block) so bake/verify work unchanged.

Usage:
    uv run python scripts/bo_som_md.py <run_id> [--grid 7,12] [--trials 50] \
        [--factor-range 0.5,1.5] [--seed 42]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parents[1]))

from _run_picker import resolve_run_id  # noqa: E402
from backend.inference.recipe import (  # noqa: E402
    build_som_md_recipe_bo,
    recipe_filename,
)
from backend.inference.model_loader import load_model, get_model  # noqa: E402

RUNS_DIR = _here.parents[1] / "data" / "runs"


def main():
    parser = argparse.ArgumentParser(description="Closed-loop SOM-MD recipe builder")
    parser.add_argument("run_id", help="Run ID (or prefix) to build the recipe from")
    parser.add_argument("--grid", default="7,12", help="SOM grid rows,cols (default 7,12)")
    parser.add_argument("--trials", type=int, default=50, help="BO budget (default 50)")
    parser.add_argument(
        "--factor-range", default="0.5,1.5",
        help="Global factor bounds lo,hi (default 0.5,1.5)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    run_id = resolve_run_id(args.run_id)
    run_path = RUNS_DIR / f"{run_id}.json"
    if not run_path.exists():
        print(f"error: no run file at {run_path}", file=sys.stderr)
        sys.exit(1)

    run = json.loads(run_path.read_text())
    state_dir = RUNS_DIR / run_id

    # Determine model_id and gen_mode from the run.
    model_id = None
    gen_mode = None
    for prompt in run["prompts"]:
        results = prompt.get("model_results", {})
        for mid, modes in results.items():
            if model_id is None:
                model_id = mid
            for gm in modes:
                if gen_mode is None:
                    gen_mode = gm
            break
        if model_id and gen_mode:
            break

    if not model_id or not gen_mode:
        print("error: could not determine model_id / gen_mode from run", file=sys.stderr)
        sys.exit(1)

    rows, cols = (int(x) for x in args.grid.split(","))
    lo, hi = (float(x) for x in args.factor_range.split(","))

    print(f"[bo_som_md] run={run_id} model={model_id} mode={gen_mode}")
    print(f"[bo_som_md] grid=({rows},{cols}) trials={args.trials} factor=[{lo},{hi}]")

    # Load the model into the resident slot.
    load_model(model_id, model_id)
    model = get_model()

    log_path = RUNS_DIR / f"{run_id}_bo_trials.jsonl"

    recipe = build_som_md_recipe_bo(
        run=run,
        model_id=model_id,
        gen_mode=gen_mode,
        model=model,
        grid_shape=(rows, cols),
        n_trials=args.trials,
        factor_range=(lo, hi),
        state_dir=state_dir,
        seed=args.seed,
        log_path=log_path,
    )

    out_path = RUNS_DIR / recipe_filename(run_id)
    out_path.write_text(json.dumps(recipe, indent=2))
    print(f"\n[bo_som_md] wrote {out_path}")
    bo_block = recipe.get("bo", {})
    print(
        f"[bo_som_md] best compliance={bo_block.get('best_score')} "
        f"factor={bo_block.get('best_factor')} subset_size={bo_block.get('best_subset_size')}"
    )


if __name__ == "__main__":
    main()
