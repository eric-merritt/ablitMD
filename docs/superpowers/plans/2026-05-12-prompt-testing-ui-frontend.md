# Prompt Testing UI — Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the React/TypeScript frontend for the ablitMD prompt testing tool — a 3-phase SPA
(config → running → results) that walks through LLM prompts, captures refusal classifications,
and renders per-category direction geometry charts.

**Architecture:** Single-page app with phase state in `App.tsx` (`'config' | 'running' | 'results'`).
No routing. Each phase is an organism component. API calls use native `fetch` against Express on
`:8199` (already proxied by Vite in `vite.config.ts`). Recharts renders direction similarity charts
in the results phase. All styling uses CSS custom properties matching the dark mockup — no CSS
framework needed.

**Tech Stack:** React 19, TypeScript, Vite, Recharts, native fetch

---

## File Map

**Created:**
- `frontend/src/types/categories.ts` — CATEGORIES constant (44 entries, excludes `unique_model_specific` + `ambiguous_context`), GROUP_LABELS, GROUP_COLORS, GROUPS, catsByGroup helper
- `frontend/src/types/model.ts` — LLM interface
- `frontend/src/types/run.ts` — Run, RunPrompt, ModelResult, SequenceStep, CategoryDirectionResult, ModeDirectionResult, RunSummary interfaces; RefusalMode + RunMode types
- `frontend/src/api/models.ts` — `fetchModels()`
- `frontend/src/api/prompts.ts` — `fetchSelectedPrompts()`
- `frontend/src/api/runs.ts` — `fetchRuns()`, `createRun()`, `fetchRun()`, `patchRun()`, `patchRunPrompt()`
- `frontend/src/api/inference.ts` — `inferenceStatus()`, `inferenceLoad()`, `inferenceGenerate()`, `inferenceCompute()`
- `frontend/src/hooks/useModels.ts` — loads models on mount
- `frontend/src/hooks/useRun.ts` — thin run CRUD wrapper with optional initial value
- `frontend/src/hooks/useInference.ts` — loaded-model state, generate/compute wrappers
- `frontend/src/components/atoms/ModelCheckbox.tsx` — checkbox row used at all 3 tree levels
- `frontend/src/components/atoms/ModeRadio.tsx` — radio button for mode selection
- `frontend/src/components/atoms/RefusalModeRadio.tsx` — hard/redirect/disclaimer radio
- `frontend/src/components/molecules/ModeRadioGroup.tsx` — non_thinking / thinking / both
- `frontend/src/components/molecules/ModelCheckboxList.tsx` — 3-level expandable tree
- `frontend/src/components/molecules/PromptCard.tsx` — prompt text + type/category badges
- `frontend/src/components/molecules/ResponsePanel.tsx` — spinner + response text
- `frontend/src/components/molecules/RefusalClassifier.tsx` — Not Refused button + Refused radios
- `frontend/src/components/molecules/RunProgress.tsx` — model/mode/prompt-index header
- `frontend/src/components/molecules/CategoryDirectionChart.tsx` — Recharts line chart per category
- `frontend/src/components/molecules/GroupDrillChart.tsx` — per-category colors for one group
- `frontend/src/components/molecules/ModelResultsRow.tsx` — tabbed non-thinking/thinking + expand
- `frontend/src/components/organisms/RunConfigPanel.tsx` — full config phase
- `frontend/src/components/organisms/PromptWalkthrough.tsx` — full running phase
- `frontend/src/components/organisms/ResultsGrid.tsx` — full results phase

**Modified:**
- `frontend/package.json` — add recharts dependency
- `frontend/src/App.tsx` — replace scaffold with phase machine
- `frontend/src/index.css` — replace light theme with dark theme CSS custom properties
- `frontend/src/App.css` — clear to empty

---

## Task 1: Dependencies & Dark Theme

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/src/index.css`
- Modify: `frontend/src/App.css`

- [ ] **Step 1: Add recharts to `frontend/package.json` dependencies**

Edit the `"dependencies"` block:

```json
{
  "dependencies": {
    "react": "^19.2.6",
    "react-dom": "^19.2.6",
    "recharts": "^3.8.1"
  }
}
```

- [ ] **Step 2: Install the new dependency**

Run from project root:
```bash
npm install --workspace=frontend
```
Expected: `recharts` and its transitive deps appear in `frontend/node_modules/`.

- [ ] **Step 3: Rewrite `frontend/src/index.css`**

Replace the entire file:

```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:         #111111;
  --surface:    #1a1a1a;
  --surface-2:  #161616;
  --surface-3:  #131313;
  --border:     #2e2e2e;
  --border-2:   #252525;
  --text:       #e0e0e0;
  --text-dim:   #aaaaaa;
  --text-muted: #888888;
  --accent:     #7c6aff;
  --accent-dim: rgba(124, 106, 255, 0.2);
  --radius:     8px;
  --font:       system-ui, 'Segoe UI', Roboto, sans-serif;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

#root { min-height: 100vh; }

button { cursor: pointer; font-family: inherit; font-size: inherit; }
input[type="checkbox"] { accent-color: var(--accent); }
input[type="radio"] { accent-color: var(--accent); }
```

- [ ] **Step 4: Clear `frontend/src/App.css`**

Replace with an empty file. The import in `App.tsx` can stay; the file just needs to exist.

```
(empty — delete all content)
```

- [ ] **Step 5: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/src/index.css frontend/src/App.css
git commit -m "chore: add recharts, replace scaffold CSS with dark theme

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 2: TypeScript Types

**Files:**
- Create: `frontend/src/types/categories.ts`
- Create: `frontend/src/types/model.ts`
- Create: `frontend/src/types/run.ts`

- [ ] **Step 1: Create `frontend/src/types/categories.ts`**

```ts
export interface Category {
  id: string
  name: string
  group: string
}

export const CATEGORIES: Category[] = [
  { id: 'violence_terrorism',         name: 'Violence & Terrorism',                                    group: 'violence_physical_harm' },
  { id: 'mass_casualty_violence',     name: 'Mass-Casualty / Targeted Violence',                       group: 'violence_physical_harm' },
  { id: 'dangerous_activity',         name: 'Dangerous Activity Instructions',                          group: 'violence_physical_harm' },
  { id: 'cbrn_uplift',                name: 'CBRN Weapons Uplift',                                     group: 'weapons_wmd' },
  { id: 'weapons_explosives',         name: 'Weapons & Explosives Instructions',                        group: 'weapons_wmd' },
  { id: 'hazardous_materials',        name: 'Unsafe Chemical / Drug / Hazardous Material Instructions', group: 'weapons_wmd' },
  { id: 'csam',                       name: 'CSAM',                                                    group: 'child_safety' },
  { id: 'child_grooming',             name: 'Child Grooming & Non-Sexual Exploitation',                group: 'child_safety' },
  { id: 'nonconsensual_sexual',       name: 'Non-Consensual Sexual Content / Deepfakes',               group: 'sexual_content' },
  { id: 'explicit_nsfw',              name: 'Explicit / NSFW Content',                                 group: 'sexual_content' },
  { id: 'malicious_code',             name: 'Functional Malicious Code / Cyberweapons',                group: 'cyber_technical_harm' },
  { id: 'hacking_unauthorized',       name: 'Hacking / Credential Theft / Unauthorized Access',        group: 'cyber_technical_harm' },
  { id: 'self_harm_methods',          name: 'Self-Harm / Suicide Method Specifics',                    group: 'self_harm_mental_health' },
  { id: 'disordered_eating',          name: 'Disordered Eating Signals',                               group: 'self_harm_mental_health' },
  { id: 'distress_signals',           name: 'Personal Distress Signals',                               group: 'self_harm_mental_health' },
  { id: 'hate_speech',                name: 'Hate Speech / Dehumanization',                            group: 'hate_discrimination' },
  { id: 'harassment_bullying',        name: 'Harassment & Bullying',                                   group: 'hate_discrimination' },
  { id: 'identity_discrimination',    name: 'Identity-Based Discrimination',                           group: 'hate_discrimination' },
  { id: 'pii_doxxing',               name: 'PII Requests / Doxxing',                                  group: 'privacy' },
  { id: 'confidential_data',          name: 'Confidential or Proprietary Data',                        group: 'privacy' },
  { id: 'nonconsensual_surveillance', name: 'Non-Consensual Surveillance',                             group: 'privacy' },
  { id: 'social_engineering',         name: 'Social Engineering / Manipulation Tactics',               group: 'deception_manipulation' },
  { id: 'impersonation',              name: 'Impersonation of Real Persons or Authorities',            group: 'deception_manipulation' },
  { id: 'fraud_facilitation',         name: 'Fraud Facilitation',                                      group: 'deception_manipulation' },
  { id: 'high_stakes_falsehoods',     name: 'High-Stakes False Information',                           group: 'misinformation' },
  { id: 'political_influence',        name: 'Political Persuasion / Targeted Influence',               group: 'misinformation' },
  { id: 'crime_assistance',           name: 'Crime Assistance',                                        group: 'illegal_activities' },
  { id: 'regulated_goods',            name: 'Regulated Goods & Services',                              group: 'illegal_activities' },
  { id: 'hide_wrongdoing',            name: 'Hiding Wrongdoing',                                       group: 'illegal_activities' },
  { id: 'medical_advice',             name: 'Medical Advice Beyond Safe Limits',                       group: 'professional_advice' },
  { id: 'legal_advice',               name: 'Legal Advice Beyond Safe Limits',                         group: 'professional_advice' },
  { id: 'financial_advice',           name: 'Financial Advice Beyond Safe Limits',                     group: 'professional_advice' },
  { id: 'copyright_ip',               name: 'Copyright / IP Infringement',                             group: 'content_policy' },
  { id: 'academic_dishonesty',        name: 'Academic Dishonesty',                                     group: 'content_policy' },
  { id: 'spam_platform_abuse',        name: 'Spam / Platform Abuse',                                   group: 'content_policy' },
  { id: 'jurisdictional',             name: 'Jurisdictional / Regulatory Restrictions',                group: 'content_policy' },
  { id: 'unfair_advantage',           name: 'Unfair Advantage',                                        group: 'content_policy' },
  { id: 'prompt_injection',           name: 'Prompt Injection / Jailbreak Attempts',                   group: 'system_integrity' },
  { id: 'reveal_system_prompt',       name: 'Requests to Reveal System Prompts',                       group: 'system_integrity' },
  { id: 'realtime_information',       name: 'Real-Time / Current Information',                         group: 'capability_limits' },
  { id: 'physical_world_actions',     name: 'Physical World Actions',                                  group: 'capability_limits' },
  { id: 'private_access',             name: 'Private Database / Account Access',                       group: 'capability_limits' },
  { id: 'physical_sensation',         name: 'Real-Time Physical Sensation',                            group: 'capability_limits' },
  { id: 'excessive_requests',         name: 'Excessive / Unbounded Requests',                          group: 'capability_limits' },
  // ambiguous_context excluded per spec — clarification responses are not refusal-direction signals
  // unique_model_specific group excluded per spec — describes model quirks, not measurable geometry
]

export const GROUP_LABELS: Record<string, string> = {
  violence_physical_harm:  'Violence & Physical Harm',
  weapons_wmd:             'Weapons & WMD',
  child_safety:            'Child Safety',
  sexual_content:          'Sexual Content',
  cyber_technical_harm:    'Cyber & Technical Harm',
  self_harm_mental_health: 'Self-Harm & Mental Health',
  hate_discrimination:     'Hate & Discrimination',
  privacy:                 'Privacy',
  deception_manipulation:  'Deception & Manipulation',
  misinformation:          'Misinformation',
  illegal_activities:      'Illegal Activities',
  professional_advice:     'Professional Advice Limits',
  content_policy:          'Content Policy',
  system_integrity:        'System Integrity',
  capability_limits:       'Capability Limits',
}

export const GROUP_COLORS: Record<string, string> = {
  violence_physical_harm:  '#ef4444',
  weapons_wmd:             '#f97316',
  child_safety:            '#ec4899',
  sexual_content:          '#f43f5e',
  cyber_technical_harm:    '#06b6d4',
  self_harm_mental_health: '#eab308',
  hate_discrimination:     '#a855f7',
  privacy:                 '#3b82f6',
  deception_manipulation:  '#d97706',
  misinformation:          '#84cc16',
  illegal_activities:      '#dc2626',
  professional_advice:     '#14b8a6',
  content_policy:          '#6366f1',
  system_integrity:        '#10b981',
  capability_limits:       '#0ea5e9',
}

export const GROUPS = Object.keys(GROUP_LABELS)

export const catsByGroup = (groupId: string) =>
  CATEGORIES.filter(cat => cat.group === groupId)
```

- [ ] **Step 2: Create `frontend/src/types/model.ts`**

```ts
export interface LLM {
  _id: string
  name: string
  modelId: string
  apiModelId: string
  vendor: string
  architecture: {
    numLayers: number
    hiddenSize: number
  }
}
```

- [ ] **Step 3: Create `frontend/src/types/run.ts`**

```ts
export type RefusalMode = 'hard' | 'redirect' | 'disclaimer' | 'none'
export type RunMode = 'non_thinking' | 'thinking' | 'both'

export interface ModelResult {
  response: string
  refused: boolean
  refusal_mode: RefusalMode
  classified_at: string
  hidden_states_key: string
}

export interface RunPrompt {
  prompt_id: string
  text: string
  category: string
  category_group: string
  type: 'harmful' | 'harmless'
  triggers: string[]
  model_results: Record<string, Record<string, ModelResult>>
}

export interface SequenceStep {
  model: string
  mode: string
}

export interface CategoryDirectionResult {
  computed_at: string
  direction_per_layer: number[][]
  similarity_per_prompt: Record<string, number[]>
}

export interface ModeDirectionResult {
  computed_at: string
  per_category: Record<string, CategoryDirectionResult>
}

export interface Run {
  run_id: string
  started_at: string
  completed_at: string | null
  incomplete: boolean
  models: string[]
  mode_selection: RunMode
  prompt_scope: { categories: string[] }
  sequence: SequenceStep[]
  current_sequence_index: number
  prompts: RunPrompt[]
  direction_results: Record<string, Record<string, ModeDirectionResult>> | null
}

export interface RunSummary {
  run_id: string
  started_at: string
  completed_at: string | null
  incomplete: boolean
  models: string[]
  mode_selection: RunMode
  prompt_count: number
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/types/
git commit -m "feat: frontend TypeScript types — categories, model, run

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 3: API Layer

**Files:**
- Create: `frontend/src/api/models.ts`
- Create: `frontend/src/api/prompts.ts`
- Create: `frontend/src/api/runs.ts`
- Create: `frontend/src/api/inference.ts`

- [ ] **Step 1: Create `frontend/src/api/models.ts`**

```ts
import type { LLM } from '../types/model'

export const fetchModels = async (): Promise<LLM[]> => {
  const res = await fetch('/api/models')
  if (!res.ok) throw new Error(`fetchModels failed: ${res.status}`)
  return res.json()
}
```

- [ ] **Step 2: Create `frontend/src/api/prompts.ts`**

```ts
export const fetchSelectedPrompts = async (
  categories: string[]
): Promise<{ _id: string; category: string; category_group: string; type: string; text: string; triggers: string[] }[]> => {
  const res = await fetch('/api/prompts/selected', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ categories }),
  })
  if (!res.ok) throw new Error(`fetchSelectedPrompts failed: ${res.status}`)
  return res.json()
}
```

- [ ] **Step 3: Create `frontend/src/api/runs.ts`**

```ts
import type { Run, RunSummary } from '../types/run'

export const fetchRuns = async (): Promise<RunSummary[]> => {
  const res = await fetch('/api/runs')
  if (!res.ok) throw new Error(`fetchRuns failed: ${res.status}`)
  return res.json()
}

export const createRun = async (body: {
  models: string[]
  mode_selection: string
  prompt_scope: { categories: string[] }
}): Promise<Run> => {
  const res = await fetch('/api/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`createRun failed: ${res.status}`)
  return res.json()
}

export const fetchRun = async (runId: string): Promise<Run> => {
  const res = await fetch(`/api/runs/${runId}`)
  if (!res.ok) throw new Error(`fetchRun failed: ${res.status}`)
  return res.json()
}

export const patchRun = async (runId: string, fields: Partial<Run>): Promise<Run> => {
  const res = await fetch(`/api/runs/${runId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(fields),
  })
  if (!res.ok) throw new Error(`patchRun failed: ${res.status}`)
  return res.json()
}

export const patchRunPrompt = async (
  runId: string,
  promptId: string,
  model_id: string,
  mode: string,
  result: object
): Promise<Run> => {
  const res = await fetch(`/api/runs/${runId}/prompts/${promptId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id, mode, result }),
  })
  if (!res.ok) throw new Error(`patchRunPrompt failed: ${res.status}`)
  return res.json()
}
```

- [ ] **Step 4: Create `frontend/src/api/inference.ts`**

```ts
export const inferenceStatus = async (): Promise<{ loaded_model: string | null }> => {
  const res = await fetch('/api/inference/status')
  if (!res.ok) throw new Error(`inferenceStatus failed: ${res.status}`)
  return res.json()
}

export const inferenceLoad = async (body: {
  model_id: string
  api_model_id: string
}): Promise<{ loaded_model: string }> => {
  const res = await fetch('/api/inference/load', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`inferenceLoad failed: ${res.status}`)
  return res.json()
}

export const inferenceGenerate = async (body: {
  prompt_id: string
  prompt_text: string
  run_id: string
  model_id: string
  mode: string
}): Promise<{ response: string; hidden_states_key: string }> => {
  const res = await fetch('/api/inference/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`inferenceGenerate failed: ${res.status}`)
  return res.json()
}

export const inferenceCompute = async (body: {
  run_id: string
  model_id: string
  mode: string
}): Promise<Record<string, {
  computed_at: string
  direction_per_layer: number[][]
  similarity_per_prompt: Record<string, number[]>
}>> => {
  const res = await fetch('/api/inference/compute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`inferenceCompute failed: ${res.status}`)
  return res.json()
}
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api/
git commit -m "feat: frontend API layer — models, prompts, runs, inference

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 4: Hooks

**Files:**
- Create: `frontend/src/hooks/useModels.ts`
- Create: `frontend/src/hooks/useRun.ts`
- Create: `frontend/src/hooks/useInference.ts`

- [ ] **Step 1: Create `frontend/src/hooks/useModels.ts`**

```ts
import { useState, useEffect } from 'react'
import type { LLM } from '../types/model'
import { fetchModels } from '../api/models'

export const useModels = () => {
  const [models, setModels] = useState<LLM[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchModels()
      .then(setModels)
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  return { models, loading, error }
}
```

- [ ] **Step 2: Create `frontend/src/hooks/useRun.ts`**

Accepts an optional `initialRun` so `PromptWalkthrough` can seed it without a flash of null state.

```ts
import { useState } from 'react'
import type { Run } from '../types/run'
import { patchRun, patchRunPrompt } from '../api/runs'

export const useRun = (initialRun?: Run) => {
  const [run, setRun] = useState<Run | null>(initialRun ?? null)

  const updatePromptResult = async (
    promptId: string,
    model_id: string,
    mode: string,
    result: object
  ): Promise<Run | undefined> => {
    if (!run) return
    const updated = await patchRunPrompt(run.run_id, promptId, model_id, mode, result)
    setRun(updated)
    return updated
  }

  const updateRunFields = async (fields: Partial<Run>): Promise<Run | undefined> => {
    if (!run) return
    const updated = await patchRun(run.run_id, fields)
    setRun(updated)
    return updated
  }

  return { run, setRun, updatePromptResult, updateRunFields }
}
```

- [ ] **Step 3: Create `frontend/src/hooks/useInference.ts`**

```ts
import { useState } from 'react'
import { inferenceLoad, inferenceGenerate, inferenceCompute } from '../api/inference'
import type { Run, ModeDirectionResult } from '../types/run'

export const useInference = () => {
  const [loadedModel, setLoadedModel] = useState<string | null>(null)
  const [modelLoading, setModelLoading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [computing, setComputing] = useState(false)

  const ensureModelLoaded = async (modelId: string, apiModelId: string) => {
    if (loadedModel === modelId) return
    setModelLoading(true)
    await inferenceLoad({ model_id: modelId, api_model_id: apiModelId })
    setLoadedModel(modelId)
    setModelLoading(false)
  }

  const generate = async (body: {
    prompt_id: string
    prompt_text: string
    run_id: string
    model_id: string
    mode: string
  }) => {
    setGenerating(true)
    try {
      return await inferenceGenerate(body)
    } finally {
      setGenerating(false)
    }
  }

  const computeAllDirections = async (
    run: Run
  ): Promise<Record<string, Record<string, ModeDirectionResult>>> => {
    setComputing(true)
    const direction_results: Record<string, Record<string, ModeDirectionResult>> = {}

    try {
      for (const step of run.sequence) {
        const per_category = await inferenceCompute({
          run_id: run.run_id,
          model_id: step.model,
          mode: step.mode,
        })
        direction_results[step.model] ??= {}
        direction_results[step.model][step.mode] = {
          computed_at: new Date().toISOString(),
          per_category,
        }
      }
    } finally {
      setComputing(false)
    }

    return direction_results
  }

  return { loadedModel, modelLoading, generating, computing, ensureModelLoaded, generate, computeAllDirections }
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/hooks/
git commit -m "feat: frontend hooks — useModels, useRun, useInference

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 5: Atoms

**Files:**
- Create: `frontend/src/components/atoms/ModelCheckbox.tsx`
- Create: `frontend/src/components/atoms/ModeRadio.tsx`
- Create: `frontend/src/components/atoms/RefusalModeRadio.tsx`

- [ ] **Step 1: Create `frontend/src/components/atoms/ModelCheckbox.tsx`**

Used at model, group, and category levels of the config tree. `variant` controls background and border color.

```tsx
interface ModelCheckboxProps {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
  onExpand?: () => void
  expanded?: boolean
  variant?: 'model' | 'group' | 'category'
}

const BG = {
  model:    'var(--surface)',
  group:    'var(--surface-2)',
  category: 'var(--surface-3)',
}

const BORDER = {
  model:    'var(--border)',
  group:    'var(--border)',
  category: 'var(--border-2)',
}

export const ModelCheckbox = ({
  label,
  checked,
  onChange,
  onExpand,
  expanded = false,
  variant = 'model',
}: ModelCheckboxProps) => (
  <div style={{
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    border: `1px solid ${BORDER[variant]}`,
    borderRadius: 'var(--radius)',
    padding: '7px 4px 7px 10px',
    background: BG[variant],
    userSelect: 'none',
  }}>
    <input
      type="checkbox"
      checked={checked}
      onChange={evt => onChange(evt.target.checked)}
    />
    <span style={{ flex: 1, color: variant === 'model' ? 'var(--text)' : 'var(--text-dim)' }}>
      {label}
    </span>
    {onExpand && (
      <span
        onClick={onExpand}
        style={{ fontSize: '16px', lineHeight: '1', color: 'var(--text-muted)', padding: '0 4px', cursor: 'pointer', borderRadius: '4px', flexShrink: 0 }}
      >
        {expanded ? '−' : '+'}
      </span>
    )}
  </div>
)
```

- [ ] **Step 2: Create `frontend/src/components/atoms/ModeRadio.tsx`**

```tsx
interface ModeRadioProps {
  value: string
  label: string
  selected: string
  onChange: (value: string) => void
}

export const ModeRadio = ({ value, label, selected, onChange }: ModeRadioProps) => (
  <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: 'var(--text-dim)' }}>
    <input
      type="radio"
      name="mode"
      value={value}
      checked={selected === value}
      onChange={() => onChange(value)}
    />
    {label}
  </label>
)
```

- [ ] **Step 3: Create `frontend/src/components/atoms/RefusalModeRadio.tsx`**

```tsx
import type { RefusalMode } from '../../types/run'

interface RefusalModeRadioProps {
  value: RefusalMode
  label: string
  selected: RefusalMode | null
  onChange: (value: RefusalMode) => void
}

export const RefusalModeRadio = ({ value, label, selected, onChange }: RefusalModeRadioProps) => (
  <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: 'var(--text-dim)' }}>
    <input
      type="radio"
      name="refusal_mode"
      value={value}
      checked={selected === value}
      onChange={() => onChange(value)}
    />
    {label}
  </label>
)
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/atoms/
git commit -m "feat: atom components — ModelCheckbox, ModeRadio, RefusalModeRadio

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 6: Config Phase

**Files:**
- Create: `frontend/src/components/molecules/ModeRadioGroup.tsx`
- Create: `frontend/src/components/molecules/ModelCheckboxList.tsx`
- Create: `frontend/src/components/organisms/RunConfigPanel.tsx`

- [ ] **Step 1: Create `frontend/src/components/molecules/ModeRadioGroup.tsx`**

```tsx
import { ModeRadio } from '../atoms/ModeRadio'

interface ModeRadioGroupProps {
  selected: string
  onChange: (mode: string) => void
}

const MODES = [
  { value: 'non_thinking', label: 'Non-Thinking' },
  { value: 'thinking',     label: 'Thinking' },
  { value: 'both',         label: 'Both' },
]

const SectionLabel = () => (
  <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '10px' }}>
    Mode
  </div>
)

export const ModeRadioGroup = ({ selected, onChange }: ModeRadioGroupProps) => (
  <div>
    <SectionLabel />
    <div style={{ display: 'flex', gap: '20px' }}>
      {MODES.map(m => (
        <ModeRadio key={m.value} value={m.value} label={m.label} selected={selected} onChange={onChange} />
      ))}
    </div>
  </div>
)
```

- [ ] **Step 2: Create `frontend/src/components/molecules/ModelCheckboxList.tsx`**

Group and category selection is global — shared across all models. Expanding a model reveals the group tree; expanding a group reveals its categories.

```tsx
import { useState } from 'react'
import { ModelCheckbox } from '../atoms/ModelCheckbox'
import { CATEGORIES, GROUP_LABELS, GROUPS, catsByGroup } from '../../types/categories'
import type { LLM } from '../../types/model'

interface ModelCheckboxListProps {
  models: LLM[]
  selectedModels: Set<string>
  selectedCategories: Set<string>
  onModelToggle: (modelId: string, checked: boolean) => void
  onCategoryToggle: (categoryId: string, checked: boolean) => void
  onGroupToggle: (groupId: string, checked: boolean) => void
}

const SectionLabel = () => (
  <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '10px' }}>
    Models &amp; Categories
  </div>
)

export const ModelCheckboxList = ({
  models,
  selectedModels,
  selectedCategories,
  onModelToggle,
  onCategoryToggle,
  onGroupToggle,
}: ModelCheckboxListProps) => {
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set())
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set())

  const toggleModel = (modelId: string) => setExpandedModels(prev => {
    const next = new Set(prev)
    next.has(modelId) ? next.delete(modelId) : next.add(modelId)
    return next
  })

  const toggleGroup = (groupId: string) => setExpandedGroups(prev => {
    const next = new Set(prev)
    next.has(groupId) ? next.delete(groupId) : next.add(groupId)
    return next
  })

  const isGroupChecked = (groupId: string) =>
    catsByGroup(groupId).every(cat => selectedCategories.has(cat.id))

  return (
    <div>
      <SectionLabel />
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        {models.map(model => (
          <div key={model.modelId}>
            <ModelCheckbox
              label={model.name}
              checked={selectedModels.has(model.modelId)}
              onChange={checked => onModelToggle(model.modelId, checked)}
              onExpand={() => toggleModel(model.modelId)}
              expanded={expandedModels.has(model.modelId)}
              variant="model"
            />
            {expandedModels.has(model.modelId) && (
              <div style={{ marginLeft: '20px', display: 'flex', flexDirection: 'column', gap: '4px', marginTop: '4px' }}>
                {GROUPS.map(groupId => (
                  <div key={groupId}>
                    <ModelCheckbox
                      label={GROUP_LABELS[groupId]}
                      checked={isGroupChecked(groupId)}
                      onChange={checked => onGroupToggle(groupId, checked)}
                      onExpand={() => toggleGroup(groupId)}
                      expanded={expandedGroups.has(groupId)}
                      variant="group"
                    />
                    {expandedGroups.has(groupId) && (
                      <div style={{ marginLeft: '20px', display: 'flex', flexDirection: 'column', gap: '4px', marginTop: '4px' }}>
                        {catsByGroup(groupId).map(cat => (
                          <ModelCheckbox
                            key={cat.id}
                            label={cat.name}
                            checked={selectedCategories.has(cat.id)}
                            onChange={checked => onCategoryToggle(cat.id, checked)}
                            variant="category"
                          />
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Create `frontend/src/components/organisms/RunConfigPanel.tsx`**

Receives `models` from App (not fetched internally — avoids a second API call). Manages local config state and the open-existing-run list.

```tsx
import { useState } from 'react'
import { ModeRadioGroup } from '../molecules/ModeRadioGroup'
import { ModelCheckboxList } from '../molecules/ModelCheckboxList'
import { createRun, fetchRuns, fetchRun } from '../../api/runs'
import { CATEGORIES } from '../../types/categories'
import type { LLM } from '../../types/model'
import type { Run, RunSummary } from '../../types/run'

interface RunConfigPanelProps {
  models: LLM[]
  onRunStart: (run: Run) => void
  onRunOpen: (run: Run) => void
}

const PanelStyle: React.CSSProperties = {
  maxWidth: '520px',
  margin: '48px auto',
  padding: '0 24px',
  display: 'flex',
  flexDirection: 'column',
  gap: '28px',
}

const btnBase: React.CSSProperties = {
  padding: '9px 20px',
  borderRadius: 'var(--radius)',
  border: 'none',
  fontWeight: 600,
}

const PrimaryBtn = (disabled: boolean): React.CSSProperties => ({
  ...btnBase,
  background: disabled ? '#333' : 'var(--accent)',
  color: disabled ? 'var(--text-muted)' : '#fff',
  cursor: disabled ? 'not-allowed' : 'pointer',
})

const SecondaryBtn: React.CSSProperties = {
  ...btnBase,
  background: 'var(--surface)',
  color: 'var(--text-dim)',
  border: '1px solid var(--border)',
  cursor: 'pointer',
}

export const RunConfigPanel = ({ models, onRunStart, onRunOpen }: RunConfigPanelProps) => {
  const [mode, setMode] = useState('non_thinking')
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set())
  const [selectedCategories, setSelectedCategories] = useState<Set<string>>(
    new Set(CATEGORIES.map(cat => cat.id))
  )
  const [existingRuns, setExistingRuns] = useState<RunSummary[] | null>(null)
  const [loadingRuns, setLoadingRuns] = useState(false)
  const [starting, setStarting] = useState(false)

  const canStart = selectedModels.size > 0 && selectedCategories.size > 0

  const handleModelToggle = (modelId: string, checked: boolean) =>
    setSelectedModels(prev => {
      const next = new Set(prev)
      checked ? next.add(modelId) : next.delete(modelId)
      return next
    })

  const handleCategoryToggle = (categoryId: string, checked: boolean) =>
    setSelectedCategories(prev => {
      const next = new Set(prev)
      checked ? next.add(categoryId) : next.delete(categoryId)
      return next
    })

  const handleGroupToggle = (groupId: string, checked: boolean) => {
    const groupCatIds = CATEGORIES.filter(cat => cat.group === groupId).map(cat => cat.id)
    setSelectedCategories(prev => {
      const next = new Set(prev)
      groupCatIds.forEach(id => checked ? next.add(id) : next.delete(id))
      return next
    })
  }

  const handleStart = async () => {
    setStarting(true)
    try {
      const run = await createRun({
        models: [...selectedModels],
        mode_selection: mode,
        prompt_scope: { categories: [...selectedCategories] },
      })
      onRunStart(run)
    } finally {
      setStarting(false)
    }
  }

  const handleOpenExisting = async () => {
    setLoadingRuns(true)
    const runs = await fetchRuns()
    setExistingRuns(runs)
    setLoadingRuns(false)
  }

  const handleSelectRun = async (summary: RunSummary) => {
    const run = await fetchRun(summary.run_id)
    onRunOpen(run)
  }

  return (
    <div style={PanelStyle}>
      <ModeRadioGroup selected={mode} onChange={setMode} />
      <ModelCheckboxList
        models={models}
        selectedModels={selectedModels}
        selectedCategories={selectedCategories}
        onModelToggle={handleModelToggle}
        onCategoryToggle={handleCategoryToggle}
        onGroupToggle={handleGroupToggle}
      />
      <div style={{ display: 'flex', gap: '12px' }}>
        <button
          style={PrimaryBtn(!canStart || starting)}
          disabled={!canStart || starting}
          onClick={handleStart}
        >
          {starting ? 'Starting…' : 'Start Run'}
        </button>
        <button style={SecondaryBtn} onClick={handleOpenExisting} disabled={loadingRuns}>
          {loadingRuns ? 'Loading…' : 'Open Existing Run'}
        </button>
      </div>
      {existingRuns && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '6px' }}>
            Previous Runs
          </div>
          {existingRuns.length === 0 && (
            <div style={{ color: 'var(--text-muted)' }}>No runs yet.</div>
          )}
          {existingRuns.map(run => (
            <div
              key={run.run_id}
              onClick={() => handleSelectRun(run)}
              style={{ padding: '10px 12px', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', cursor: 'pointer', display: 'flex', justifyContent: 'space-between' }}
            >
              <span style={{ color: 'var(--text-dim)' }}>
                {new Date(run.started_at).toLocaleString()} — {run.prompt_count} prompts
              </span>
              <span style={{ color: run.incomplete ? '#eab308' : '#10b981', fontSize: '12px' }}>
                {run.incomplete ? 'Incomplete' : 'Complete'}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/
git commit -m "feat: config phase — ModeRadioGroup, ModelCheckboxList, RunConfigPanel

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 7: Running Phase

**Files:**
- Create: `frontend/src/components/molecules/RunProgress.tsx`
- Create: `frontend/src/components/molecules/PromptCard.tsx`
- Create: `frontend/src/components/molecules/ResponsePanel.tsx`
- Create: `frontend/src/components/molecules/RefusalClassifier.tsx`
- Create: `frontend/src/components/organisms/PromptWalkthrough.tsx`

- [ ] **Step 1: Create `frontend/src/components/molecules/RunProgress.tsx`**

```tsx
interface RunProgressProps {
  modelName: string
  mode: string
  currentIndex: number
  total: number
}

const modeLabel = (mode: string) => mode === 'non_thinking' ? 'Non-Thinking' : 'Thinking'

export const RunProgress = ({ modelName, mode, currentIndex, total }: RunProgressProps) => (
  <div style={{ padding: '12px 20px', background: 'var(--surface)', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '13px' }}>
    <span style={{ color: 'var(--text-dim)' }}>
      <strong style={{ color: 'var(--text)' }}>{modelName}</strong>
      {' | '}
      <strong style={{ color: 'var(--text)' }}>{modeLabel(mode)}</strong>
    </span>
    <span style={{ color: 'var(--text-muted)' }}>
      Prompt <strong style={{ color: 'var(--accent)' }}>{currentIndex + 1}</strong> / {total}
    </span>
  </div>
)
```

- [ ] **Step 2: Create `frontend/src/components/molecules/PromptCard.tsx`**

```tsx
import { GROUP_LABELS } from '../../types/categories'
import type { RunPrompt } from '../../types/run'

interface PromptCardProps {
  prompt: RunPrompt
}

const Badge = ({ label, color }: { label: string; color: string }) => (
  <span style={{ padding: '2px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 600, background: color, color: '#fff', textTransform: 'uppercase' as const, letterSpacing: '0.05em' }}>
    {label}
  </span>
)

export const PromptCard = ({ prompt }: PromptCardProps) => (
  <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '20px' }}>
    <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
      <Badge label={prompt.type} color={prompt.type === 'harmful' ? '#dc2626' : '#16a34a'} />
      <Badge label={GROUP_LABELS[prompt.category_group] ?? prompt.category_group} color="#4b5563" />
    </div>
    <p style={{ color: 'var(--text)', lineHeight: '1.6' }}>{prompt.text}</p>
  </div>
)
```

- [ ] **Step 3: Create `frontend/src/components/molecules/ResponsePanel.tsx`**

```tsx
interface ResponsePanelProps {
  generating: boolean
  response: string | null
  error: string | null
}

const Spinner = () => (
  <>
    <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    <div style={{ width: '20px', height: '20px', border: '2px solid var(--border)', borderTopColor: 'var(--accent)', borderRadius: '50%', animation: 'spin 0.8s linear infinite', flexShrink: 0 }} />
  </>
)

export const ResponsePanel = ({ generating, response, error }: ResponsePanelProps) => (
  <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '20px', minHeight: '80px' }}>
    {generating && (
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', color: 'var(--text-muted)' }}>
        <Spinner />
        Generating response…
      </div>
    )}
    {!generating && error && (
      <div style={{ color: '#ef4444' }}>{error}</div>
    )}
    {!generating && !error && response && (
      <p style={{ color: 'var(--text-dim)', whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>{response}</p>
    )}
  </div>
)
```

- [ ] **Step 4: Create `frontend/src/components/molecules/RefusalClassifier.tsx`**

Selecting a Refused radio is the complete classification action — no secondary confirm. Resets radio selection after each classification so the next prompt starts clean.

```tsx
import { useState } from 'react'
import { RefusalModeRadio } from '../atoms/RefusalModeRadio'
import type { RefusalMode } from '../../types/run'

interface RefusalClassifierProps {
  disabled: boolean
  onClassify: (refused: boolean, mode: RefusalMode) => void
}

const REFUSED_MODES: { value: RefusalMode; label: string }[] = [
  { value: 'hard',       label: 'Hard' },
  { value: 'redirect',   label: 'Redirect' },
  { value: 'disclaimer', label: 'Disclaimer' },
]

export const RefusalClassifier = ({ disabled, onClassify }: RefusalClassifierProps) => {
  const [selected, setSelected] = useState<RefusalMode | null>(null)

  const handleRefused = (mode: RefusalMode) => {
    setSelected(mode)
    onClassify(true, mode)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <button
        disabled={disabled}
        onClick={() => { setSelected(null); onClassify(false, 'none') }}
        style={{
          padding: '10px 20px',
          borderRadius: 'var(--radius)',
          border: '1px solid #16a34a',
          background: 'transparent',
          color: '#16a34a',
          fontWeight: 600,
          alignSelf: 'flex-start',
          opacity: disabled ? 0.4 : 1,
          cursor: disabled ? 'not-allowed' : 'pointer',
        }}
      >
        Not Refused
      </button>
      <div style={{ borderTop: '1px solid var(--border)', paddingTop: '12px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          Refused
        </div>
        {REFUSED_MODES.map(m => (
          <RefusalModeRadio
            key={m.value}
            value={m.value}
            label={m.label}
            selected={disabled ? null : selected}
            onChange={handleRefused}
          />
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 5: Create `frontend/src/components/organisms/PromptWalkthrough.tsx`**

Auto-generates a response when `currentPrompt` changes. Classification writes to the run, then advances: next prompt → next sequence step → compute + complete.

```tsx
import { useState, useEffect, useMemo } from 'react'
import { RunProgress } from '../molecules/RunProgress'
import { PromptCard } from '../molecules/PromptCard'
import { ResponsePanel } from '../molecules/ResponsePanel'
import { RefusalClassifier } from '../molecules/RefusalClassifier'
import { useInference } from '../../hooks/useInference'
import { useRun } from '../../hooks/useRun'
import { patchRun } from '../../api/runs'
import type { Run, RunPrompt, RefusalMode } from '../../types/run'

interface WalkthroughModel {
  modelId: string
  apiModelId: string
  name: string
}

interface PromptWalkthroughProps {
  initialRun: Run
  models: WalkthroughModel[]
  onComplete: (run: Run) => void
}

export const PromptWalkthrough = ({ initialRun, models, onComplete }: PromptWalkthroughProps) => {
  const { run, updatePromptResult, updateRunFields } = useRun(initialRun)
  const { modelLoading, generating, computing, ensureModelLoaded, generate, computeAllDirections } = useInference()

  const [promptIndex, setPromptIndex] = useState(0)
  const [response, setResponse] = useState<string | null>(null)
  const [genError, setGenError] = useState<string | null>(null)

  const currentStep = run ? run.sequence[run.current_sequence_index] : null

  const pendingPrompts: RunPrompt[] = useMemo(() => {
    if (!run || !currentStep) return []
    return run.prompts.filter(p => !p.model_results[currentStep.model]?.[currentStep.mode])
  }, [run, currentStep])

  const currentPrompt = pendingPrompts[promptIndex] ?? null
  const currentModel = models.find(m => m.modelId === currentStep?.model) ?? null

  useEffect(() => {
    if (!currentPrompt || !currentStep || !currentModel || !run) return

    setResponse(null)
    setGenError(null)

    const doGenerate = async () => {
      try {
        await ensureModelLoaded(currentModel.modelId, currentModel.apiModelId)
        const result = await generate({
          prompt_id: currentPrompt.prompt_id,
          prompt_text: currentPrompt.text,
          run_id: run.run_id,
          model_id: currentStep.model,
          mode: currentStep.mode,
        })
        setResponse(result.response)
      } catch (err: unknown) {
        setGenError(err instanceof Error ? err.message : 'Generation failed')
      }
    }

    doGenerate()
  }, [currentPrompt?.prompt_id, currentStep?.model, currentStep?.mode])

  const handleClassify = async (refused: boolean, mode: RefusalMode) => {
    if (!currentPrompt || !currentStep || !run) return

    const isLastPrompt = promptIndex >= pendingPrompts.length - 1
    const isLastStep = run.current_sequence_index >= run.sequence.length - 1

    const result = {
      response: response ?? '',
      refused,
      refusal_mode: mode,
      classified_at: new Date().toISOString(),
      hidden_states_key: `${currentPrompt.prompt_id}__${currentStep.model.replace('/', '__')}__${currentStep.mode}`,
    }

    const updatedRun = await updatePromptResult(
      currentPrompt.prompt_id, currentStep.model, currentStep.mode, result
    )
    if (!updatedRun) return

    if (!isLastPrompt) {
      setPromptIndex(prev => prev + 1)
      return
    }

    if (!isLastStep) {
      await updateRunFields({ current_sequence_index: run.current_sequence_index + 1 })
      setPromptIndex(0)
      return
    }

    const direction_results = await computeAllDirections(updatedRun)
    const finalRun = await patchRun(updatedRun.run_id, {
      direction_results,
      incomplete: false,
      completed_at: new Date().toISOString(),
    })
    onComplete(finalRun)
  }

  if (!run || !currentStep || !currentModel) {
    return <div style={{ padding: '48px', color: 'var(--text-muted)' }}>Loading run…</div>
  }

  if (!currentPrompt) {
    return (
      <div style={{ padding: '48px', color: 'var(--text-muted)' }}>
        {computing ? 'Computing directions…' : 'All prompts classified.'}
      </div>
    )
  }

  return (
    <div style={{ maxWidth: '720px', margin: '0 auto', display: 'flex', flexDirection: 'column' }}>
      <RunProgress
        modelName={currentModel.name}
        mode={currentStep.mode}
        currentIndex={promptIndex}
        total={pendingPrompts.length}
      />
      <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {modelLoading && (
          <div style={{ padding: '10px', background: 'var(--accent-dim)', borderRadius: 'var(--radius)', color: 'var(--accent)', fontSize: '13px' }}>
            Loading {currentModel.name} into GPU…
          </div>
        )}
        <PromptCard prompt={currentPrompt} />
        <ResponsePanel generating={generating || modelLoading} response={response} error={genError} />
        <RefusalClassifier
          disabled={generating || modelLoading || !response}
          onClassify={handleClassify}
        />
      </div>
    </div>
  )
}
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/
git commit -m "feat: running phase — PromptCard, ResponsePanel, RefusalClassifier, PromptWalkthrough

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 8: Results Phase

**Files:**
- Create: `frontend/src/components/molecules/CategoryDirectionChart.tsx`
- Create: `frontend/src/components/molecules/GroupDrillChart.tsx`
- Create: `frontend/src/components/molecules/ModelResultsRow.tsx`
- Create: `frontend/src/components/organisms/ResultsGrid.tsx`

- [ ] **Step 1: Create `frontend/src/components/molecules/CategoryDirectionChart.tsx`**

Per-prompt lines. Solid = refused. Dashed = not refused. Color = group color.

```tsx
import { LineChart, Line, XAxis, YAxis, ReferenceLine, Tooltip, ResponsiveContainer } from 'recharts'
import { GROUP_COLORS } from '../../types/categories'
import type { CategoryDirectionResult, RunPrompt } from '../../types/run'

interface CategoryDirectionChartProps {
  categoryName: string
  directionResult: CategoryDirectionResult
  prompts: RunPrompt[]
  modelId: string
  mode: string
}

export const CategoryDirectionChart = ({
  categoryName,
  directionResult,
  prompts,
  modelId,
  mode,
}: CategoryDirectionChartProps) => {
  const { similarity_per_prompt } = directionResult
  const promptKeys = Object.keys(similarity_per_prompt)
  const numLayers = promptKeys.length > 0 ? similarity_per_prompt[promptKeys[0]].length : 0

  const chartData = Array.from({ length: numLayers }, (_, layerIndex) => {
    const point: Record<string, number> = { layer: layerIndex }
    promptKeys.forEach(key => { point[key] = similarity_per_prompt[key][layerIndex] })
    return point
  })

  const promptByKey = (key: string) =>
    prompts.find(p => p.model_results[modelId]?.[mode]?.hidden_states_key === key)

  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '12px' }}>
      <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-dim)', marginBottom: '8px' }}>
        {categoryName}
      </div>
      <ResponsiveContainer width="100%" height={140}>
        <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 4, left: -20 }}>
          <XAxis dataKey="layer" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
          <YAxis domain={[-1, 1]} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
          <ReferenceLine y={0} stroke="var(--border)" />
          <Tooltip
            contentStyle={{ background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: '4px', fontSize: '11px' }}
            labelStyle={{ color: 'var(--text-muted)' }}
          />
          {promptKeys.map(key => {
            const prompt = promptByKey(key)
            const refused = prompt?.model_results[modelId]?.[mode]?.refused ?? false
            const groupColor = GROUP_COLORS[prompt?.category_group ?? ''] ?? '#888'
            return (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={groupColor}
                strokeWidth={1.5}
                strokeDasharray={refused ? undefined : '4 2'}
                dot={false}
              />
            )
          })}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
```

- [ ] **Step 2: Create `frontend/src/components/molecules/GroupDrillChart.tsx`**

All categories from a single group side by side.

```tsx
import { CATEGORIES, GROUP_LABELS } from '../../types/categories'
import { CategoryDirectionChart } from './CategoryDirectionChart'
import type { ModeDirectionResult, RunPrompt } from '../../types/run'

interface GroupDrillChartProps {
  groupId: string
  modeResult: ModeDirectionResult
  prompts: RunPrompt[]
  modelId: string
  mode: string
}

export const GroupDrillChart = ({ groupId, modeResult, prompts, modelId, mode }: GroupDrillChartProps) => {
  const groupCats = CATEGORIES.filter(cat => cat.group === groupId)

  return (
    <div>
      <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-dim)', marginBottom: '10px' }}>
        {GROUP_LABELS[groupId]}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: '8px' }}>
        {groupCats.map(cat => {
          const dirResult = modeResult.per_category[cat.id]
          return dirResult ? (
            <CategoryDirectionChart
              key={cat.id}
              categoryName={cat.name}
              directionResult={dirResult}
              prompts={prompts}
              modelId={modelId}
              mode={mode}
            />
          ) : null
        })}
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Create `frontend/src/components/molecules/ModelResultsRow.tsx`**

Collapsed (default): tabbed non-thinking / thinking. Expanded: side-by-side.

```tsx
import { useState } from 'react'
import { CategoryDirectionChart } from './CategoryDirectionChart'
import { CATEGORIES, GROUPS } from '../../types/categories'
import type { Run } from '../../types/run'

interface ModelResultsRowProps {
  modelId: string
  modelName: string
  run: Run
  visibleGroups: Set<string>
}

const tabBtnStyle = (active: boolean): React.CSSProperties => ({
  padding: '4px 12px',
  borderRadius: '4px',
  border: '1px solid var(--border)',
  background: active ? 'var(--accent)' : 'var(--surface)',
  color: active ? '#fff' : 'var(--text-dim)',
  fontSize: '12px',
  fontWeight: 600,
  cursor: 'pointer',
})

export const ModelResultsRow = ({ modelId, modelName, run, visibleGroups }: ModelResultsRowProps) => {
  const [activeTab, setActiveTab] = useState('non_thinking')
  const [expanded, setExpanded] = useState(false)

  const modelResults = run.direction_results?.[modelId] ?? {}
  const availableModes = Object.keys(modelResults)

  const TabBar = () => (
    <div style={{ display: 'flex', gap: '4px', marginBottom: '12px' }}>
      {availableModes.map(mode => (
        <button key={mode} onClick={() => setActiveTab(mode)} style={tabBtnStyle(activeTab === mode)}>
          {mode === 'non_thinking' ? 'Non-Thinking' : 'Thinking'}
        </button>
      ))}
    </div>
  )

  const ChartGrid = ({ mode }: { mode: string }) => {
    const modeResult = modelResults[mode]
    if (!modeResult) return <div style={{ color: 'var(--text-muted)', fontSize: '13px' }}>No data</div>

    const visibleCats = CATEGORIES.filter(
      cat => visibleGroups.has(cat.group) && modeResult.per_category[cat.id]
    )

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: '8px' }}>
        {visibleCats.map(cat => (
          <CategoryDirectionChart
            key={cat.id}
            categoryName={cat.name}
            directionResult={modeResult.per_category[cat.id]}
            prompts={run.prompts}
            modelId={modelId}
            mode={mode}
          />
        ))}
      </div>
    )
  }

  return (
    <div style={{ marginBottom: '32px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h3 style={{ color: 'var(--text)', fontSize: '15px', fontWeight: 600 }}>{modelName}</h3>
        <button
          onClick={() => setExpanded(prev => !prev)}
          style={{ padding: '4px 10px', borderRadius: '4px', border: '1px solid var(--border)', background: 'var(--surface)', color: 'var(--text-dim)', fontSize: '12px', cursor: 'pointer' }}
        >
          {expanded ? 'Collapse' : 'Expand'}
        </button>
      </div>
      {expanded ? (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
          {availableModes.map(mode => (
            <div key={mode}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '8px' }}>
                {mode === 'non_thinking' ? 'Non-Thinking' : 'Thinking'}
              </div>
              <ChartGrid mode={mode} />
            </div>
          ))}
        </div>
      ) : (
        <div>
          <TabBar />
          <ChartGrid mode={activeTab} />
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Create `frontend/src/components/organisms/ResultsGrid.tsx`**

```tsx
import { useState } from 'react'
import { ModelResultsRow } from '../molecules/ModelResultsRow'
import { GROUPS, GROUP_LABELS } from '../../types/categories'
import type { Run } from '../../types/run'

interface ResultsGridProps {
  run: Run
  modelNames: Record<string, string>
}

export const ResultsGrid = ({ run, modelNames }: ResultsGridProps) => {
  const [visibleGroups, setVisibleGroups] = useState<Set<string>>(new Set(GROUPS))

  const toggleGroup = (groupId: string) =>
    setVisibleGroups(prev => {
      const next = new Set(prev)
      next.has(groupId) ? next.delete(groupId) : next.add(groupId)
      return next
    })

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '10px' }}>
          Visible Groups
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px' }}>
          {GROUPS.map(groupId => (
            <label key={groupId} style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', color: 'var(--text-dim)', fontSize: '13px' }}>
              <input
                type="checkbox"
                checked={visibleGroups.has(groupId)}
                onChange={() => toggleGroup(groupId)}
              />
              {GROUP_LABELS[groupId]}
            </label>
          ))}
        </div>
      </div>
      {run.models.map(modelId => (
        <ModelResultsRow
          key={modelId}
          modelId={modelId}
          modelName={modelNames[modelId] ?? modelId}
          run={run}
          visibleGroups={visibleGroups}
        />
      ))}
    </div>
  )
}
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/
git commit -m "feat: results phase — CategoryDirectionChart, ModelResultsRow, ResultsGrid

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 9: App Wiring

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Rewrite `frontend/src/App.tsx`**

```tsx
import { useState } from 'react'
import './App.css'
import { RunConfigPanel } from './components/organisms/RunConfigPanel'
import { PromptWalkthrough } from './components/organisms/PromptWalkthrough'
import { ResultsGrid } from './components/organisms/ResultsGrid'
import { useModels } from './hooks/useModels'
import type { Run } from './types/run'

type Phase = 'config' | 'running' | 'results'

const App = () => {
  const { models } = useModels()
  const [phase, setPhase] = useState<Phase>('config')
  const [activeRun, setActiveRun] = useState<Run | null>(null)

  const modelNames = Object.fromEntries(models.map(m => [m.modelId, m.name]))
  const walkthroughModels = models.map(m => ({ modelId: m.modelId, apiModelId: m.apiModelId, name: m.name }))

  const handleRunStart = (run: Run) => { setActiveRun(run); setPhase('running') }
  const handleRunOpen  = (run: Run) => { setActiveRun(run); setPhase(run.incomplete ? 'running' : 'results') }
  const handleRunComplete = (run: Run) => { setActiveRun(run); setPhase('results') }

  return (
    <>
      {phase === 'config' && (
        <RunConfigPanel models={models} onRunStart={handleRunStart} onRunOpen={handleRunOpen} />
      )}
      {phase === 'running' && activeRun && (
        <PromptWalkthrough initialRun={activeRun} models={walkthroughModels} onComplete={handleRunComplete} />
      )}
      {phase === 'results' && activeRun && (
        <ResultsGrid run={activeRun} modelNames={modelNames} />
      )}
    </>
  )
}

export default App
```

- [ ] **Step 2: Run TypeScript type check**

From `frontend/`:
```bash
npx tsc --noEmit
```
Expected: zero errors. Fix any type errors before continuing — they indicate interface mismatches between tasks.

- [ ] **Step 3: Start dev server and verify config phase**

```bash
npm run dev --workspace=frontend
```

Open `http://localhost:5173`. Expected:
- Dark background (`#111`)
- "Mode" radio row: Non-Thinking / Thinking / Both
- "Models & Categories" tree (empty if backend not running — that's fine)
- "Start Run" button disabled, "Open Existing Run" button present

- [ ] **Step 4: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: App phase machine — config/running/results wiring

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Self-Review

**Spec coverage:**
- ✅ §5a Config: ModeRadioGroup (3 options, non_thinking default), ModelCheckboxList (model/group/category 3-level tree, `+`/`−` expand), Start button disabled until model + category selected, Open existing run button with list sorted newest-first, complete/incomplete labels
- ✅ §5a Category selection global across models (shared `selectedCategories` Set in RunConfigPanel)
- ✅ §5b Running: RunProgress (model name, mode, prompt N/total), PromptCard (text + type badge + group badge), ResponsePanel (spinner during generation), RefusalClassifier (Not Refused button + Hard/Redirect/Disclaimer radios, immediate action on radio select)
- ✅ §5c Results: group toggle checkboxes, ModelResultsRow (tabbed non-thinking/thinking collapsed, side-by-side expanded), CategoryDirectionChart (layer x-axis, cosine similarity y-axis, per-prompt lines colored by group, refused solid / non-refused dashed)
- ✅ §3e Resumption: `handleRunOpen` routes incomplete → running, complete → results
- ✅ §3d Direction computation: `computeAllDirections` iterates sequence steps, calls `inferenceCompute`, assembles `direction_results[model][mode]` structure, writes via `patchRun`
- ✅ §7 Constraints: `unique_model_specific` not in CATEGORIES constant; `ambiguous_context` explicitly excluded with comment
- ✅ Dark theme: `#111` bg, `#1a1a1a` surface, `#7c6aff` accent, matches mockup

**Placeholder scan:** None found.

**Type consistency:**
- `Run.direction_results` typed as `Record<string, Record<string, ModeDirectionResult>> | null` — matches assembly in `useInference.computeAllDirections` and consumption in `ModelResultsRow`
- `patchRunPrompt(runId, promptId, model_id, mode, result)` — signature matches between `runs.ts` and `useRun.updatePromptResult`
- `inferenceCompute` returns `Record<string, { computed_at, direction_per_layer, similarity_per_prompt }>` — assigned to `per_category` in `useInference`, matches `ModeDirectionResult.per_category` shape
- `hidden_states_key` built as `${prompt_id}__${model.replace('/', '__')}__${mode}` in `PromptWalkthrough` — matches Python service construction (`safe_model = model_id.replace("/", "__")`)
- `CategoryDirectionChart` receives `categoryName: string` (not `categoryId`) — callers in `ModelResultsRow` and `GroupDrillChart` both pass `cat.name` ✅
