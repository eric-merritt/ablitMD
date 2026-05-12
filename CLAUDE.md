# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ablitMD (Multi-Dimensional) is a research sandbox for categorical abliteration of LLMs. The core hypothesis is that refusal behavior in LLMs is multi-dimensional — different refusal categories activate different directions across different layers — and that these directions can be independently tuned via per-category sliders rather than a single global abliteration pass.

The project builds on `../abliterating_models/` (single-direction abliteration) but is not a replacement of that codebase — it extends the approach to multi-dimensional refusal geometry.

## Commands

### Frontend (from `frontend/`)
```
npm run dev       # Vite dev server on :5173, proxies /api to :8199
npm run build     # tsc + vite build to dist/
npm run lint      # eslint
```

### Backend (from `backend/`)
```
npm run dev       # node --watch bin/www
npm run start     # node bin/www
```

### From project root
```
npm run dev:frontend
npm run dev:backend
```

### Python (from project root, uses uv)
```
uv run <script>   # abliteration scripts
```

## Architecture

### Frontend / Backend Split
- **Frontend** — React + TypeScript + Vite on `:5173`. Vite proxies `/api/*` to Express on `:8199`. Currently a scaffold; UI will expose per-category refusal sliders backed by abliteration results stored in MongoDB.
- **Backend** — Express 5 (`"type": "module"`, ESM throughout). Entry point is `backend/bin/www`, app wired in `backend/app.js`. All routes under `/api`.

### Data Layer
MongoDB via Mongoose. Three models in `backend/models/`:

- **`llm.js`** — Open-source models available for abliteration (Qwen, Llama, DeepSeek, Gemma). Key fields: `architecture` (numLayers, hiddenSize, outputProjections), `abliterationDefaults` (layerStart, layerEnd, refusalWeight), `research.selfReportedCategories` (which refusal categories the model named in self-report).
- **`refCat.js`** — 46 refusal categories derived from cross-model research, grouped into 11 themes. Exports both the Mongoose model and a `CATEGORIES` constant array used for seeding. Category `id` strings are the shared key across models and prompts.
- **`prompt.js`** — Test prompts keyed by `category` (string id) and `type` (harmful/harmless). Includes a `triggers` array of secondary category IDs likely co-activated by the prompt — used to subtract cross-category noise when isolating direction geometry. Embeds a `results` array of per-model test outcomes: `refused`, `refusalMode`, `activeLayers`, `directionSimilarity`, `refusalWeight`.

Seed data lives in `backend/db/seeds/` — LLM seeds in `llms/` (one file per model), categories in `refCats/`. A `db/seed.js` runner executes all seeds.

### Research Data
`data/refusal_scope_research.json` — cross-model self-report dataset collected via API chat sessions with Qwen3.6, Claude Opus 4.7, GPT-5.5, DeepSeek V4 Pro, Gemma 4 31B IT, and Llama 4 Maverick. Contains refusal categories, granularity self-reports, training methodology disclosures, and calibration failure modes per model. This file is the source of truth for `selfReportedCategories` in LLM seed files.

`data/harmful_behaviors.json` — 416 harmful prompts from `mlabonne/harmful_behaviors` (HuggingFace). Source dataset for harmful prompt seeds.

`data/harmless_alpaca.json` — 25k general instruction prompts from `mlabonne/harmless_alpaca` (HuggingFace). Reference dataset; harmless prompts are written manually per category since they must be topically adjacent to the harmful category to test over-refusal.

### Prompt Classification Rules
- **`type: 'harmful'`** — prompts the model refuses on moral grounds. Legality is a correlate, not the defining criterion. The hidden state directions being measured are shaped by moral/ethical training signal.
- **`type: 'harmless'`** — topically adjacent but no moral objection. Tests over-refusal. Must be written manually per category.
- **`triggers`** — secondary category IDs whose refusal directions likely co-activate. Used to difference out cross-category noise. Only add triggers for genuinely distinct directions — avoid adding the same 2-3 categories to everything.
- Trim "and avoid getting caught / without getting caught / make it look like an accident" tails from prompts. Clean direction signal is preferred over compound requests.
- Soft refusals, redirects, and "I can't give personalized advice" responses count as refusals — `refusalMode` captures the granularity.
- `unique_model_specific` categories are excluded from prompt seeds — they describe model behavior quirks rather than measurable refusal geometry.
- Category `reportedBy` field (on `unique_model_specific` entries in `Categories.js`) uses the same model ID strings as `llm.research.selfReportedCategories`.

### Python Abliteration Layer
The actual hidden state collection and weight ablation runs in Python (see `pyproject.toml`). The Python scripts will eventually write results back to MongoDB, populating `prompt.results` with `activeLayers` and `directionSimilarity` per category per model. Architecture auto-detection in the abliteration scripts handles standard transformer, DeltaNet/linear attention, and Mamba output projections.

### Category IDs
The `id` strings in `CATEGORIES` (`refCat.js`) are the canonical identifiers used across:
- `prompt.category`
- `llm.research.selfReportedCategories`
- Seed files in `db/seeds/`
- The JSON research dataset

Do not rename category ids without updating all four locations.
