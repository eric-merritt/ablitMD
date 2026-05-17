import type { Run, RunSummary, RunMode, ModelResult } from '../types/run'

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