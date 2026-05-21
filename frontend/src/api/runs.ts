import type { Run, RunSummary, RunMode, ModelResult, CategoryDirectionResult } from '../types/run'

// Legacy shape: direction_results = { [catId]: { by_mode, alignment, ... } }
// Canonical shape: direction_results = { [modelId]: { [mode]: { per_category, computed_at } } }
// Wraps legacy → canonical when detected, using the run's single sequence step.
const isFlatDirections = (directions: unknown): directions is Record<string, CategoryDirectionResult> => {
  const firstEntry = Object.values(directions ?? {})[0] as { by_mode?: unknown; per_category?: unknown } | undefined
  return Boolean( firstEntry && 'by_mode' in firstEntry && !('per_category' in firstEntry) )
}

const wrapFlatDirections = (run: Run): Run => {
  if (!run.direction_results || !isFlatDirections(run.direction_results)) return run
  const firstStep = run.sequence?.[0]
  if (!firstStep) return run
  const flatDirections = run.direction_results as unknown as Record<string, CategoryDirectionResult>
  const sampleComputedAt = Object.values(flatDirections)[0]?.computed_at ?? new Date().toISOString()
  return {
    ...run,
    direction_results: {
      [firstStep.model]: {
        [firstStep.mode]: { computed_at: sampleComputedAt, per_category: flatDirections },
      },
    },
  }
}

export const fetchRuns = async (): Promise<RunSummary[]> => {
  const res = await fetch('/api/runs')
  if (!res.ok) throw new Error(`fetchRuns failed: ${res.status}`)
  return res.json()
}

export const createRun = async (body: {
  models: string[]
  mode_selection: RunMode
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
  const run = await res.json() as Run
  return wrapFlatDirections(run)
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
  result: Partial<ModelResult>
): Promise<Run> => {
  const res = await fetch(`/api/runs/${runId}/prompts/${promptId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id, mode, result }),
  })
  if (!res.ok) throw new Error(`patchRunPrompt failed: ${res.status}`)
  return res.json()
}