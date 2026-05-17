# Prompt Testing UI — Design Spec
**Date:** 2026-05-12
**Status:** Approved

---

## 1. Overview

A single-page research tool for manually testing LLM prompts from the ablitMD seed database
against self-hosted models loaded via HuggingFace Transformers (bfloat16, no quantization).
The user walks through prompts one at a time, classifies each model response as refused or not
refused, and after all prompts are classified the tool computes per-category refusal directions
from raw hidden state vectors and renders a WandB-style chart grid for visual analysis.

---

## 2. System Architecture

Three tiers, all running on the VastAI RTX 6000 Pro instance:

```
Browser (local) → Vite :5173 → Express :8199 → Python FastAPI :8200
                                     ↓
                              MongoDB (models, prompts)
                                     ↓
                         ./data/runs/*.json + *.npz
```

**Startup sequence:**

Single command from project root: `npm run dev`

Uses `concurrently` in the root `package.json` `dev` script to fire all three processes
in one terminal with labeled, colored output:
- `[backend]` — Express on :8199, connects to MongoDB, mounts API routes
- `[inference]` — Python FastAPI on :8200 via `uv run python backend/inference/service.py`, model slot empty
- `[frontend]` — Vite on :5173, proxies `/api` to Express

`concurrently` added as a root devDependency. All three processes killed together on Ctrl+C.

**Model load lifecycle:** One model loaded at a time. Stays resident in VRAM for the session.
`device_map="cuda:0"` used — model loaded directly onto the single GPU.
Model unloaded only on explicit unload request or service restart.

Express proxies `/api/inference/*` to :8200 internally. The browser never talks to :8200.

---

## 3. Data Layer

### 3a. Prompt Schema Addition

`category_group` added as an indexed field on the `Prompt` Mongoose schema alongside
the existing `category` (category_id) field. Both are denormalized onto the document.
Seed runner populates `category_group` from `CATEGORIES` constant at seed time.

### 3b. Refusal Mode Enum

Schema enum revised from `['hard', 'soft', 'redirect', 'partial', 'clarification', 'none']`
to:

```
['hard', 'redirect', 'disclaimer', 'none']
```

- `hard` — flat refusal, no engagement, no alternatives
- `redirect` — refuses specific request, offers safe alternative or educational reframe
- `disclaimer` — completes the request but with explicit professional/safety disclaimers
- `none` — not refused (stored value; UI label is "Not Refused")

Derived from cross-model self-report data in `data/refusal_scope_research.json`.
`educational` folds into `redirect` per Gemma's "soft refusal + education" pattern.
`clarification` and `partial` removed — not applicable to the designed prompt set.

### 3c. Run File Format

Two files per run, same stem, stored in `./data/runs/`:

**`run_<ISO-timestamp>_<uuid-short>.json`**

```json
{
  "run_id": "abc123",
  "started_at": "2026-05-12T14:00:00Z",
  "completed_at": null,
  "incomplete": true,
  "models": ["Qwen/Qwen3.6-27B", "google/gemma-4-31b-it"],
  "mode_selection": "both",
  "prompt_scope": {
    "categories": ["dangerous_activity", "hate_speech"]
  },
  "sequence": [
    { "model": "Qwen/Qwen3.6-27B", "mode": "non_thinking" },
    { "model": "Qwen/Qwen3.6-27B", "mode": "thinking" },
    { "model": "google/gemma-4-31b-it", "mode": "non_thinking" }
  ],
  "current_sequence_index": 2,
  "prompts": [
    {
      "prompt_id": "<mongo_id>",
      "text": "...",
      "category": "dangerous_activity",
      "category_group": "violence_physical_harm",
      "type": "harmful",
      "triggers": [],
      "model_results": {
        "Qwen/Qwen3.6-27B": {
          "non_thinking": {
            "response": "...",
            "refused": true,
            "refusal_mode": "hard",
            "classified_at": "2026-05-12T14:01:22Z",
            "hidden_states_key": "<prompt_id>__Qwen__Qwen3.6-27B__non_thinking"
          },
          "thinking": {
            "response": "...",
            "refused": false,
            "refusal_mode": "none",
            "classified_at": "2026-05-12T14:22:11Z",
            "hidden_states_key": "<prompt_id>__Qwen__Qwen3.6-27B__thinking"
          }
        }
      }
    }
  ],
  "direction_results": null
}
```

`direction_results` is `null` until post-run compute. After compute:

```json
"direction_results": {
  "Qwen/Qwen3.6-27B": {
    "non_thinking": {
      "computed_at": "2026-05-12T16:00:00Z",
      "per_category": {
        "dangerous_activity": {
          "direction_per_layer": [[0.12, ...], ...],
          "similarity_per_prompt": {
            "<prompt_id>": [0.87, 0.43, ...]
          }
        }
      }
    },
    "thinking": {
      "computed_at": "2026-05-12T16:01:00Z",
      "per_category": {
        "dangerous_activity": {
          "direction_per_layer": [[0.09, ...], ...],
          "similarity_per_prompt": {
            "<prompt_id>": [0.71, 0.38, ...]
          }
        }
      }
    }
  }
}
```

**`run_<ISO-timestamp>_<uuid-short>.npz`**

Numpy archive. Each key is the `hidden_states_key` from the JSON manifest.
Each value is a float32 array of shape `(n_layers + 1, hidden_size)` — one vector per
layer at the last-token position.

### 3d. Direction Computation Logic

Classification outcome, not `prompt.type`, determines direction pool membership — per
`(model, mode)` pair independently:

- **Refusal states** — prompts where `refused: true` (hard / redirect / disclaimer),
  regardless of prompt `type`
- **Non-refusal states** — prompts where `refused: false` (none), regardless of prompt `type`

A harmful prompt classified as `none` contributes to the non-refusal mean.
A harmless prompt classified with any refusal mode contributes to the refusal mean
(over-refusal signal — visually identifiable in charts).

Direction = normalize(mean(refusal hidden states) − mean(non-refusal hidden states)) per layer.
Computed independently per `(model_id, mode)` pair.

### 3e. Resumption

No automatic detection. The config phase has two entry points:

- **New run** — configure models, modes, categories, click Start
- **Open existing run** — select from a list of previous runs (populated from `./data/runs/`)

On open:
- **Complete run** (`incomplete: false`) → skips config and running phases entirely,
  goes directly to results/charts view
- **Incomplete run** (`incomplete: true`) → skips config, loads run state, enters running
  phase at the first unclassified prompt for the current sequence position

---

## 4. API Routes

### Express (all under `/api`)

```
GET  /api/models
GET  /api/prompts/all
POST /api/prompts/selected          { categories: string[], groups: string[] }
GET  /api/runs
POST /api/runs                      { models: string[], modes: string[], prompt_scope: { categories: string[] } }
GET  /api/runs/:runId
PATCH /api/runs/:runId              top-level fields: current_sequence_index, incomplete, completed_at
PATCH /api/runs/:runId/prompts/:promptId   { model_id: string, mode: string, result: ModelResult }
POST /api/runs/:runId/compute       triggers direction computation for completed sequence entries

POST /api/inference/load            { model_id: string, api_model_id: string }
POST /api/inference/generate        { prompt_id: string, prompt_text: string, run_id: string, model_id: string, mode: 'thinking' | 'non_thinking' }
POST /api/inference/compute         { run_id: string, model_id: string, mode: string }
GET  /api/inference/status
```

### Python FastAPI (internal, :8200)

```
POST /load      { model_id, api_model_id }
POST /generate  { prompt_id, prompt_text, run_id, model_id, mode }
                → saves hidden states to .npz, returns { response: string }
POST /compute   { run_id, model_id, mode }
                → reads .npz + JSON classifications, computes directions + similarities
                → returns direction_results blob for that (model_id, mode) pair
GET  /status    → { loaded_model: string | null }
```

---

## 5. Frontend Components

Single page (`App.tsx`) with phase state: `'config' | 'running' | 'results'`.
No routing.

```
App.tsx
  components/
    organisms/
      RunConfigPanel.tsx         phase: config
      PromptWalkthrough.tsx      phase: running
      ResultsGrid.tsx            phase: results
    molecules/
      ModelCheckboxList.tsx      3-level expandable tree: model → group → category
      ModeRadioGroup.tsx         non_thinking (default) | thinking | both
      PromptCard.tsx             prompt text + type badge + category badge
      ResponsePanel.tsx          generated response text + loading spinner
      RefusalClassifier.tsx      Not Refused button + Refused radio group
      RunProgress.tsx            current model/mode, prompt index / total
      ModelResultsRow.tsx        tabbed thinking/non-thinking + expand toggle
      CategoryDirectionChart.tsx layer similarity lines, colored by group
      GroupDrillChart.tsx        single group expanded, colored per category
    atoms/
      ModelCheckbox.tsx          used at all three levels of the tree
      ModeRadio.tsx
      RefusalModeRadio.tsx       hard | redirect | disclaimer
      GroupToggle.tsx            enable/disable group in results view
  hooks/
    useRun.ts
    useInference.ts
    useModels.ts
    usePrompts.ts
  api/
    runs.ts
    models.ts
    prompts.ts
    inference.ts
  types/
    run.ts
    prompt.ts
    model.ts
```

### 5a. Run Config Phase

`RunConfigPanel` contains:
- `ModeRadioGroup` — three mutually exclusive options, `non_thinking` selected by default:
  ```
  ○ Non-Thinking  ○ Thinking  ○ Both
  ```
  "Both" populates `sequence` with non-thinking pass first, then thinking pass for each model.
- `ModelCheckboxList` — three-level expandable tree. Default state: model rows only visible.
  Thin rounded rectangle rows. `+` button vertically centered at right edge, `px-1` padding.
  ```
  [x] Qwen3.6 27B                              [+]
      [x] Violence & Physical Harm             [+]
          [x] Dangerous Activity Instructions
          [x] Violence & Terrorism
          [x] Mass-Casualty Violence
      [x] Weapons & WMD                        [+]
          ...
  [x] Gemma 4 31B IT                           [+]
      ...
  ```
  - Model level checkbox: selects/deselects that model from the run
  - Group level checkbox: selects/deselects all categories in that group (globally — affects all models)
  - Category level checkbox: selects/deselects that category (globally — affects all models)
  - Same groups/categories appear under each model; selection is shared across models
- Start button (disabled until at least one model, one category selected)
- Open existing run button → list of runs from `./data/runs/`, sorted by date desc;
  complete runs labeled as such, incomplete runs labeled as such

### 5b. Running Phase

`PromptWalkthrough` contains:
- `RunProgress` — "Model: Qwen3.6 27B | Mode: Non-Thinking | Prompt 12 / 47"
- `PromptCard` — prompt text, `type` badge (harmful / harmless), category badge
- `ResponsePanel` — loading spinner during generation, response text when done
- `RefusalClassifier`:

  ```
  [ Not Refused ]

  ─── Refused ──────────────────
    ○ Hard
    ○ Redirect
    ○ Disclaimer
  ```

  "Not Refused" is a standalone action button. Selecting any Refused radio is the
  complete classification — immediate write to run file, advance to next prompt.
  No secondary confirm step.

### 5c. Results Phase

`ResultsGrid` contains:
- Per-`(model, mode)` group toggle row at top: checkboxes to show/hide groups
  for refusal and non-refusal sets independently
- One `ModelResultsRow` per model:
  - **Collapsed (default):** tabbed — `Non-Thinking | Thinking` — one chart row visible
  - **Expanded:** two chart rows side by side, category columns aligned for comparison
  - Each row: one `CategoryDirectionChart` per category
    - X axis: layer index
    - Y axis: cosine similarity with category refusal direction
    - Individual prompt lines: solid, colored by group
    - Group mean lines (mean of refused pool, mean of non-refused pool): dotted, same color scheme
    - `prompt.type` (harmful/harmless) annotated as a marker on each line endpoint
  - Drill-down: `GroupDrillChart` replaces row when a single group is selected —
    per-category colors instead of per-group colors, one group at a time

Charting library: **Recharts** (React-native, TypeScript-friendly, no canvas footprint).

---

## 6. Python Inference Service

Located at `backend/inference/service.py`. Depends on `torch`, `transformers`, `numpy`,
`fastapi`, `uvicorn`. Uses the same `AutoModelForCausalLM.from_pretrained` pattern as
`abliterate_v2.py` with `torch_dtype=torch.bfloat16` and `device_map="cuda:0"`.

**Thinking mode:** passed as a generation flag. For Qwen3 models, `enable_thinking=True`
in the generate call. Other models receive the `non_thinking` path only unless they
support an equivalent parameter.

**Hidden state capture:** `output_hidden_states=True` on the forward pass. Last-token
position extracted per layer: `output.hidden_states[layer_idx][0, -1, :]`. Saved
immediately to the run's `.npz` file after each prompt, keyed by `hidden_states_key`.

**Direction computation:** reads all classified prompt entries for the given `(run_id,
model_key)`, loads their hidden state arrays from `.npz`, groups into refusal/non-refusal
pools by classification outcome (not `prompt.type`), computes per-layer direction vectors,
then computes per-prompt cosine similarity against the direction. Returns the full
`direction_results` blob for that model_key.

---

## 7. Constraints and Notes

- `unique_model_specific` category group excluded from prompt scope by default
  (per CLAUDE.md conventions — describes model quirks, not measurable direction geometry)
- `ambiguous_context` category excluded from testable scope (clarification-type responses
  are not refusal-direction signals)
- Run files accumulate across sessions; `./data/runs/` should be gitignored
- `.npz` files are large (depends on model hidden size × layers × prompt count); not
  committed to git
- Cross-model direction comparison is valid only when the same prompt set and
  classification criteria are used — enforced by global (not per-model) category selection
- `prompt.type` (harmful/harmless design intent) is annotation metadata only;
  it does not affect direction computation, chart grouping, or coloring
