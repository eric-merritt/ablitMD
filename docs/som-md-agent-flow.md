# SOM-MD Agent Flow

End-to-end, agent-driven walkthrough of the **SOM-MD** (Self-Organizing Map
Multi-Directional) ablation pipeline. Follow these steps in order; each one is
a discrete action with a concrete command or UI path and a "done when" check so
an agent can verify progress without guessing.

The whole point: take a finished capture run, learn the refusal manifold with a
hexagonal SOM, and emit a recipe that makes the model **comply** on prompts it
previously refused — then prove it with a held-out verify pass.

---

## Prerequisites (one-time)

| Check | How | Done when |
|---|---|---|
| Deps installed | `uv sync && npm i` | no errors |
| Flash-attn wheel present | see `README.md` quickstart | `import flash_attn` works in the venv |
| Inference service up | `npm run dev` (FastAPI :8238 + Express :8237) | `curl localhost:8238/health` returns ok |
| A capture run exists | see **Step 0** | a `<run_id>.json` is on disk in `data/runs/` |

> The SOM-MD math (SOM training, direction extraction, best-layer selection)
> lives in [`backend/inference/som.py`](../backend/inference/som.py). The recipe
> builders live in [`backend/inference/recipe.py`](../backend/inference/recipe.py).
> You don't edit either for a normal run — you drive them.

---

## Step 0 — Capture hidden states (the data)

You need a completed capture run: prompts with model responses **and** the
per-prompt hidden-state `.npy` files on disk.

- Run ID format: `run_<UTC-timestamp>_<8-hex>` e.g. `run_2026-09-19T10-01-32-556Z_bfb44921`.
- On-disk layout (see [`data-template.md`](./data-template.md) for the full schema):

```
data/runs/
  <run_id>.json                       # run manifest + prompts + model_results
  <run_id>/
    <prompt_id>__<org>__<model>__<gen_mode>.npy   # (n_layers, dim) float32
```

**Done when:** `data/runs/<run_id>.json` exists **and** its sibling directory
contains `.npy` files. Verify:

```bash
RUN=run_2026-09-19T10-01-32-556Z_bfb44921
ls data/runs/$RUN.json && ls data/runs/$RUN | head
```

If the run is `incomplete: true` or `.npy` files are missing, finish capturing
responses first (the normal app flow) before continuing. SOM-MD needs **both**
refused and complied hidden states — a run where everything refused or nothing
refused won't produce directions.

---

## Step 1 — Build the recipe

Two paths. Pick based on how much compute you want to spend.

### Path A — Fast, deterministic (no model in memory)

Trains the SOM, picks the best layer by the Arditi refusal metric, emits `k`
directions. No GPU inference needed.

```bash
uv run python scripts/build_recipe.py <run_id> \
  --method som-md \
  --k 7 \
  --grid 4,4 \
  --factor 1.0
```

Or from the UI: open the run → **SOM-MD panel** → set `k`, grid, factor →
**Build**. This POSTs to `/ablate/:runId/recipe/som-md`.

### Path B — Closed-loop Bayesian Optimization (needs the model loaded)

Runs BO over direction subsets + ablation factor, scored by *measured* compliance
on held-out refused prompts. Slower (loads the model, runs inference per trial)
but finds a better factor/subset than a fixed `k`.

```bash
uv run python scripts/bo_som_md.py <run_id> \
  --grid 7,12 \
  --trials 50 \
  --factor-range 0.5,1.5 \
  --seed 42
```

Writes a `<run_id>.recipe.<timestamp>.json` **and** a trial log at
`data/runs/<run_id>_bo_trials.jsonl`.

**Done when:** the newest `data/runs/<run_id>.recipe.*.json` has
`"method": "som_md"`. Confirm:

```bash
ls -t data/runs/$RUN.recipe.*.json | head -1 | xargs python3 -c \
  'import json,sys; r=json.load(open(sys.argv[1])); print(r["method"], r["best_layer"], r["k"])'
```

---

## Step 2 — Bake the ablation into the model

Applies the recipe's per-layer directions in place on the resident model.

- UI: **Bake** button (POST `/ablate/:runId/bake`).
- The Express route waits for any in-flight recipe rebuild before baking, so you
  never bake a stale recipe.

**Done when:** the service logs a successful bake and the model is resident with
the ablation applied.

---

## Step 3 — Verify on held-out prompts

Re-runs the refused prompts against the baked model and streams per-prompt
`refused_after` results as NDJSON. The proxy tees the stream and records the
per-category complied/refused outcome against the recipe's master-factor identity,
so a later attempt at the same config surfaces a "verified before" warning.

- UI: **Verify** button (POST `/ablate/:runId/verify`).
- Watch the streamed lines; each `prompt` event carries `category` and
  `refused_after`. A working recipe flips most `refused_after: true` → `false`.

**Done when:** compliance rate on the held-out refused set is meaningfully above
the pre-ablation baseline. If it's ~0, the factor is too small or the best layer
was wrong — go back to Step 1 (Path B) and let BO search for a better factor.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `state_dir required for SOM-MD recipe` | Ran the builder without pointing at the run's state dir | Use the script (it resolves `data/runs/<run_id>`) or pass `state_dir` |
| `need both refused and complied hidden states` | Run has no complied prompts, or `.npy` files missing | Finish capturing; confirm `.npy` count ≈ prompt count |
| Recipe method isn't `som_md` after build | A legacy two-phase recipe got written instead | Re-run with `--method som-md`; check you didn't hit the plain `/recipe` route |
| Verify compliance ~0% | Factor too low or wrong layer | Use Path B (BO) to search factor/subset; or raise `--factor` |
| BO is slow / OOMs | Model not resident, or grid too big | Load model first (`load_model`), drop `--grid` to `4,4`, lower `--trials` |

---

## Where things live

- SOM math: [`backend/inference/som.py`](../backend/inference/som.py)
  - `train_som`, `compute_som_directions`, `select_best_layer`, `hex_grid_neighbors`
- Recipe builders: [`backend/inference/recipe.py`](../backend/inference/recipe.py)
  - `build_som_md_recipe` (fast), `build_som_md_recipe_bo` (closed-loop)
- CLI entrypoints: [`scripts/build_recipe.py`](../scripts/build_recipe.py),
  [`scripts/bo_som_md.py`](../scripts/bo_som_md.py)
- API routes: [`backend/routes/api/ablation.js`](../backend/routes/api/ablation.js)
  - `POST /:runId/recipe/som-md`, `POST /:runId/bake`, `POST /:runId/verify`
- UI panel: [`frontend/src/components/molecules/SomMdPanel.tsx`](../frontend/src/components/molecules/SomMdPanel.tsx)
- Data schema reference: [`data-template.md`](./data-template.md)
