#!/usr/bin/env python3
"""Per-category, per-layer refusal directions, orthogonalized across categories.

For every category and every layer l:
    dir_cat_l = normalize( refused_mean_l - none_mean_l )
where refused/none are that CATEGORY's own prompts (the non-SOM convention from
direction.compute_direction). Then, at each layer, Gram-Schmidt the 44 category
directions (center-out order) so no shared component is ever ablated twice.

Pure numpy over the run's .npy hidden states — no model load.
"""

import json
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parents[1]))

from backend.inference.direction import compute_direction  # noqa: E402
from _run_picker import resolve_run_id  # noqa: E402

RUNS_DIR = _here.parents[1] / "data" / "runs"


def main():
    run_id = resolve_run_id(sys.argv[1])
    run_path = RUNS_DIR / f"{run_id}.json"
    run = json.loads(run_path.read_text())
    state_dir = RUNS_DIR / run_id

    model_id, gen_mode = None, None
    for p in run["prompts"]:
        for mid, modes in p.get("model_results", {}).items():
            if model_id is None:
                model_id = mid
            for gm in modes:
                if gen_mode is None:
                    gen_mode = gm
        if model_id and gen_mode:
            break

    # Group prompts by category, load hidden states once.
    by_cat: dict[str, list[dict]] = {}
    hidden: dict[str, np.ndarray] = {}
    for p in run["prompts"]:
        r = p.get("model_results", {}).get(model_id, {}).get(gen_mode)
        if not r:
            continue
        key = r["hidden_states_key"]
        npy = state_dir / f"{key}.npy"
        if not npy.exists():
            continue
        hidden[key] = np.load(str(npy))
        by_cat.setdefault(p["category"], []).append({
            "key": key,
            "refused": bool(r.get("refused")),
            "mode": r.get("refusal_mode"),
        })

    # Per category: refused stack vs none stack -> per-layer direction + magnitude.
    cat_dirs = {}   # cat -> (n_layers, dim) unit dir
    cat_mag = {}    # cat -> (n_layers,) magnitude
    for cat, cls in by_cat.items():
        ref_keys = [c["key"] for c in cls if c["refused"]]
        none_keys = [c["key"] for c in cls if c["mode"] == "none"]
        if not ref_keys or not none_keys:
            print(f"  skip {cat}: refused={len(ref_keys)} none={len(none_keys)}")
            continue
        ref_stack = np.stack([hidden[k] for k in ref_keys])
        none_stack = np.stack([hidden[k] for k in none_keys])
        direction, magnitude = compute_direction(ref_stack, none_stack)
        cat_dirs[cat] = direction
        cat_mag[cat] = magnitude

    n_layers = next(iter(cat_dirs.values())).shape[0]
    print(f"categories with directions: {len(cat_dirs)}  layers: {n_layers}")

    # Order is arbitrary — all 44 are applied at equal factor and the output set
    # is orthonormal regardless of Gram-Schmidt order. Use plain category order.
    order = list(cat_dirs.keys())

    # Apply the same Gram-Schmidt the rest of the codebase uses (dedup_overlap):
    # each direction keeps only the component orthogonal to what's already been
    # ablated, at its own factor; a shared subspace is carried once. All factors
    # equal here, so processing order is just `order`.
    from backend.inference.recipe import dedup_overlap

    out_layers = {}
    kept_counts = []
    for l in range(n_layers):
        pairs = []
        for c in order:
            v = cat_dirs[c][l].astype(np.float32)
            nrm = float(np.linalg.norm(v))
            if nrm < 1e-8:
                continue
            pairs.append((v / nrm, 1.0))
        kept = dedup_overlap(pairs)
        kept_counts.append(len(kept))
        out_layers[l] = [u.tolist() for u, _ in kept]

    print(f"dedup_overlap applied per layer; kept {min(kept_counts)}..{max(kept_counts)} "
          f"directions/layer (dropped fully-covered ones).")

    # Save a sidecar so we can inspect / feed a recipe builder.
    out = RUNS_DIR / f"{run_id}.per_category_orth.json"
    out.write_text(json.dumps({
        "run_id": run_id,
        "model_id": model_id,
        "gen_mode": gen_mode,
        "n_layers": n_layers,
        "order": order,  # category order used for Gram-Schmidt (center-out)
        "per_layer_directions": {str(l): out_layers[l] for l in range(n_layers)},
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
