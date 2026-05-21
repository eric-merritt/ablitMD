import { useMemo } from 'react'
import { CategoryDirectionChart } from '../molecules/CategoryDirectionChart'
import type { CategoryDirectionResult, RunPrompt, RefusalMode } from '../../types/run'

const NL = 28
const MODEL_ID = 'demo-model'
const MODE = 'non_thinking'

// Bell curve, clamped to [-1, 1]
const bv = (peak: number, h: number, w: number, L: number) =>
  Math.max(-1, Math.min(1, h * Math.exp(-Math.pow(L - peak, 2) / (2 * w * w))))

const magCurve  = (peak: number, h: number, w: number) =>
  Array.from({ length: NL }, (_, L) => Math.max(0, bv(peak, h, w, L)))

const simCurve = (peak: number, h: number, w: number, offset: number) =>
  Array.from({ length: NL }, (_, L) => bv(peak, h + offset, w, L))

const k = (id: string) => `${id}__demo-model__non_thinking`

type Entry = { id: string; mode: RefusalMode }
type Visitor = { id: string; sourceCat: string; refMode: RefusalMode }

function makeCatResult(entries: Entry[], visitors: Visitor[] = []): CategoryDirectionResult {
  const groups: Record<string, string[]> = { hard: [], redirect: [], disclaimer: [], none: [] }
  for (const e of entries) groups[e.mode].push(k(e.id))

  const allKeys = entries.map(e => k(e.id))

  // Sim vs a given direction depends on the prompt's own mode
  const simVsDir = (
    dirMode: string, promptMode: string, idx: number, peakShift = 0
  ): number[] => {
    const base: Record<string, Record<string, number>> = {
      hard:       { hard: 0.72, redirect: 0.38, disclaimer: 0.15, none: -0.08 },
      redirect:   { hard: 0.35, redirect: 0.65, disclaimer: 0.22, none: -0.05 },
      disclaimer: { hard: 0.20, redirect: 0.25, disclaimer: 0.58, none: -0.03 },
    }
    const h = (base[dirMode]?.[promptMode] ?? 0) + ((idx * 7) % 10 - 5) * 0.02
    const w = promptMode === 'none' ? 6.5 : 4.5
    return simCurve(17 + peakShift, h, w, 0)
  }

  const byMode: CategoryDirectionResult['by_mode'] = {}

  if (groups.hard.length > 0) {
    const sim: Record<string, number[]> = {}
    allKeys.forEach((key, idx) => {
      sim[key] = simVsDir('hard', entries[idx].mode, idx)
    })
    const tsp: Record<string, number[]> = {}
    visitors.forEach((v, vi) => {
      tsp[k(v.id)] = simCurve(17, v.refMode === 'hard' ? 0.54 : 0.28, 5, vi * 0.04)
    })
    byMode.hard = {
      direction_per_layer: [],
      magnitude_per_layer: magCurve(17, 1.42, 3.8),
      sample_count: groups.hard.length,
      similarity_per_prompt: sim,
      trigger_similarity_per_prompt: visitors.length > 0 ? tsp : undefined,
    }
  }

  if (groups.redirect.length > 0) {
    const sim: Record<string, number[]> = {}
    allKeys.forEach((key, idx) => {
      sim[key] = simVsDir('redirect', entries[idx].mode, idx, 1)
    })
    byMode.redirect = {
      direction_per_layer: [],
      magnitude_per_layer: magCurve(18, 0.98, 4.5),
      sample_count: groups.redirect.length,
      similarity_per_prompt: sim,
    }
  }

  if (groups.disclaimer.length > 0) {
    const sim: Record<string, number[]> = {}
    allKeys.forEach((key, idx) => {
      sim[key] = simVsDir('disclaimer', entries[idx].mode, idx, -1)
    })
    byMode.disclaimer = {
      direction_per_layer: [],
      magnitude_per_layer: magCurve(16, 0.66, 5.2),
      sample_count: groups.disclaimer.length,
      similarity_per_prompt: sim,
    }
  }

  const alignment: Record<string, number[]> = {}
  if (byMode.hard && byMode.redirect)
    alignment['hard_vs_redirect'] = Array.from({ length: NL }, (_, L) => bv(17, 0.76, 4.5, L))
  if (byMode.hard && byMode.disclaimer)
    alignment['hard_vs_disclaimer'] = Array.from({ length: NL }, (_, L) => bv(17, 0.60, 5, L))
  if (byMode.redirect && byMode.disclaimer)
    alignment['redirect_vs_disclaimer'] = Array.from({ length: NL }, (_, L) => bv(17, 0.70, 5, L))

  const trigger_meta: CategoryDirectionResult['trigger_meta'] = {}
  for (const v of visitors) {
    trigger_meta![k(v.id)] = { source_category: v.sourceCat, refused: v.refMode !== 'none', refusal_mode: v.refMode }
  }

  return {
    computed_at: '2025-01-01T00:00:00Z',
    by_mode: byMode,
    alignment,
    trigger_meta: visitors.length > 0 ? trigger_meta : undefined,
  }
}

function mkPrompt(id: string, cat: string, mode: RefusalMode, triggers: string[] = []): RunPrompt {
  return {
    prompt_id: id,
    text: `Mock prompt ${id}`,
    category: cat,
    category_group: cat,
    type: mode === 'none' ? 'harmless' : 'harmful',
    triggers,
    model_results: {
      [MODEL_ID]: {
        [MODE]: {
          response: mode === 'none' ? 'Sure, here is some info.' : "I can't help with that.",
          refused: mode !== 'none',
          refusal_mode: mode,
          classified_at: '2025-01-01T00:00:00Z',
          hidden_states_key: k(id),
        },
      },
    },
  }
}

// ── Category definitions ──────────────────────────────────────────────────────

const DEMO_CATS: Array<{
  id: string
  name: string
  entries: Entry[]
  visitors?: Visitor[]
  prompts: RunPrompt[]
}> = [
  {
    id: 'social_engineering',
    name: 'Social Engineering / Manipulation Tactics',
    entries: [
      { id: 'se_h1', mode: 'hard' }, { id: 'se_h2', mode: 'hard' }, { id: 'se_h3', mode: 'hard' },
      { id: 'se_r1', mode: 'redirect' }, { id: 'se_r2', mode: 'redirect' },
      { id: 'se_d1', mode: 'disclaimer' },
      { id: 'se_n1', mode: 'none' }, { id: 'se_n2', mode: 'none' }, { id: 'se_n3', mode: 'none' },
    ],
    visitors: [
      { id: 'ff_h1', sourceCat: 'fraud_facilitation', refMode: 'hard' },
      { id: 'ff_h2', sourceCat: 'fraud_facilitation', refMode: 'hard' },
      { id: 'imp_r1', sourceCat: 'impersonation', refMode: 'redirect' },
    ],
    prompts: [
      mkPrompt('se_h1', 'social_engineering', 'hard', ['fraud_facilitation']),
      mkPrompt('se_h2', 'social_engineering', 'hard'),
      mkPrompt('se_h3', 'social_engineering', 'hard'),
      mkPrompt('se_r1', 'social_engineering', 'redirect'),
      mkPrompt('se_r2', 'social_engineering', 'redirect'),
      mkPrompt('se_d1', 'social_engineering', 'disclaimer'),
      mkPrompt('se_n1', 'social_engineering', 'none'),
      mkPrompt('se_n2', 'social_engineering', 'none'),
      mkPrompt('se_n3', 'social_engineering', 'none'),
    ],
  },
  {
    id: 'impersonation',
    name: 'Impersonation of Real Persons or Authorities',
    entries: [
      { id: 'imp_h1', mode: 'hard' }, { id: 'imp_h2', mode: 'hard' },
      { id: 'imp_r1', mode: 'redirect' }, { id: 'imp_r2', mode: 'redirect' },
      { id: 'imp_d1', mode: 'disclaimer' },
      { id: 'imp_n1', mode: 'none' }, { id: 'imp_n2', mode: 'none' },
    ],
    visitors: [
      { id: 'se_h1', sourceCat: 'social_engineering', refMode: 'hard' },
      { id: 'se_h2', sourceCat: 'social_engineering', refMode: 'hard' },
    ],
    prompts: [
      mkPrompt('imp_h1', 'impersonation', 'hard', ['social_engineering']),
      mkPrompt('imp_h2', 'impersonation', 'hard'),
      mkPrompt('imp_r1', 'impersonation', 'redirect'),
      mkPrompt('imp_r2', 'impersonation', 'redirect'),
      mkPrompt('imp_d1', 'impersonation', 'disclaimer'),
      mkPrompt('imp_n1', 'impersonation', 'none'),
      mkPrompt('imp_n2', 'impersonation', 'none'),
    ],
  },
  {
    id: 'fraud_facilitation',
    name: 'Fraud Facilitation',
    entries: [
      { id: 'ff_h1', mode: 'hard' }, { id: 'ff_h2', mode: 'hard' }, { id: 'ff_h3', mode: 'hard' },
      { id: 'ff_r1', mode: 'redirect' },
      { id: 'ff_n1', mode: 'none' }, { id: 'ff_n2', mode: 'none' }, { id: 'ff_n3', mode: 'none' },
    ],
    prompts: [
      mkPrompt('ff_h1', 'fraud_facilitation', 'hard', ['social_engineering']),
      mkPrompt('ff_h2', 'fraud_facilitation', 'hard'),
      mkPrompt('ff_h3', 'fraud_facilitation', 'hard'),
      mkPrompt('ff_r1', 'fraud_facilitation', 'redirect'),
      mkPrompt('ff_n1', 'fraud_facilitation', 'none'),
      mkPrompt('ff_n2', 'fraud_facilitation', 'none'),
      mkPrompt('ff_n3', 'fraud_facilitation', 'none'),
    ],
  },
  {
    id: 'crime_assistance',
    name: 'Crime Assistance',
    entries: [
      { id: 'ca_h1', mode: 'hard' }, { id: 'ca_h2', mode: 'hard' },
      { id: 'ca_r1', mode: 'redirect' }, { id: 'ca_r2', mode: 'redirect' },
      { id: 'ca_d1', mode: 'disclaimer' },
      { id: 'ca_n1', mode: 'none' }, { id: 'ca_n2', mode: 'none' },
    ],
    prompts: [
      mkPrompt('ca_h1', 'crime_assistance', 'hard'),
      mkPrompt('ca_h2', 'crime_assistance', 'hard'),
      mkPrompt('ca_r1', 'crime_assistance', 'redirect'),
      mkPrompt('ca_r2', 'crime_assistance', 'redirect'),
      mkPrompt('ca_d1', 'crime_assistance', 'disclaimer'),
      mkPrompt('ca_n1', 'crime_assistance', 'none'),
      mkPrompt('ca_n2', 'crime_assistance', 'none'),
    ],
  },
  {
    id: 'regulated_goods',
    name: 'Regulated Goods & Services',
    entries: [
      { id: 'rg_h1', mode: 'hard' }, { id: 'rg_h2', mode: 'hard' },
      { id: 'rg_r1', mode: 'redirect' }, { id: 'rg_r2', mode: 'redirect' },
      { id: 'rg_d1', mode: 'disclaimer' },
      { id: 'rg_n1', mode: 'none' }, { id: 'rg_n2', mode: 'none' },
    ],
    prompts: [
      mkPrompt('rg_h1', 'regulated_goods', 'hard'),
      mkPrompt('rg_h2', 'regulated_goods', 'hard'),
      mkPrompt('rg_r1', 'regulated_goods', 'redirect'),
      mkPrompt('rg_r2', 'regulated_goods', 'redirect'),
      mkPrompt('rg_d1', 'regulated_goods', 'disclaimer'),
      mkPrompt('rg_n1', 'regulated_goods', 'none'),
      mkPrompt('rg_n2', 'regulated_goods', 'none'),
    ],
  },
]

export const ChartDemo = () => {
  const catResults = useMemo(() =>
    DEMO_CATS.map(cat => ({
      ...cat,
      result: makeCatResult(cat.entries, cat.visitors ?? []),
    })), []
  )

  // Merge all prompts across categories so trigger cross-references resolve
  const allPrompts = useMemo(() =>
    DEMO_CATS.flatMap(cat => cat.prompts), []
  )

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: '24px' }}>
      <div style={{ marginBottom: '20px' }}>
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '4px' }}>
          Deception &amp; Manipulation · demo-model · non_thinking
        </div>
        <div style={{ fontSize: '18px', fontWeight: 700, color: 'var(--text)' }}>
          Chart Layout Preview
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '12px' }}>
        {catResults.map(({ id, name, result }) => (
          <CategoryDirectionChart
            key={id}
            categoryName={name}
            categoryId={id}
            directionResult={result}
            prompts={allPrompts}
            modelId={MODEL_ID}
            mode={MODE}
          />
        ))}
      </div>
    </div>
  )
}
