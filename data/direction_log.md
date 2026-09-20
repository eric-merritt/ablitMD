# Direction-Combination Log

Tracks every ablation recipe I bake, which categories it flipped, and the per-category
directional effect — so we can later compare overlapping categories, subtract their common
directionality, and find uncovered space.

## How to read this

Each SOM-MD recipe emits **k unit directions** (here k=7) toward the harmless centroid,
applied uniformly to every decoder layer. A direction is a 5120-d vector; I don't log the
raw vectors here (they're in the recipe JSON), but I log:

- **recipe id** → which `data/runs/*.recipe.*.json` file
- **k / grid / best_layer / factor** → the knobs
- **flipped categories** → which of the 3 refusal categories went REFUSE→COMPLY after bake
- **per-category Δ** → from `host_check.py --compare` (base→ablit comply counts)

The subtraction idea: if two recipes both flip `cbrn_uplift`, their shared directions are
candidates for the "common directionality" of that category. Categories flipped by *different*
direction subsets mark uncovered space worth a targeted recipe.

## Log

| # | date (UTC)   | recipe file (short)              | k | grid | best_layer | factor | flipped categories            | per-cat Δ (base→ablit) | notes |
|---|--------------|----------------------------------|---|------|-----------|--------|-------------------------------|------------------------|-------|
| 1 | 2026-09-19   | run_...bfb44921.recipe.233521    | 7 | 4,4  | 63        | 1.0    | (pending bake)               |                        | first SOM-MD test ablation; not yet baked/verified |

## Pending verification

- [ ] Bake recipe #1 on vast.ai → `Qwen3.8-27B-Ablit`
- [ ] Run `host_check.py --compare base ablit` → fill flipped + Δ columns for row 1
- [ ] Update `data/checklist.md` Ablit column from the same run
