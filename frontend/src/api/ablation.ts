import type {
  DivergencePayload, SlimRecipe, RecipeParams, VerifyEvent, DirectionCompareRow,
} from '../types/ablation'

const jsonOrThrow = async (response: Response) => {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail || `request failed (${response.status})`)
  }
  return response.json()
}

export const getDivergence = (runId: string): Promise<DivergencePayload> =>
  fetch(`/api/ablation/${runId}/divergence`).then(jsonOrThrow)

export const buildRecipe = (runId: string, params: RecipeParams): Promise<SlimRecipe> =>
  fetch(`/api/ablation/${runId}/recipe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  }).then(jsonOrThrow)

export const bakeModel = (runId: string, mode: 'ablitmd' | 'classic' = 'ablitmd', factor?: number): Promise<{ saved_to: string }> =>
  fetch(`/api/ablation/${runId}/bake`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode, ...(factor !== undefined && { factor }) }),
  }).then(jsonOrThrow)

export const compareDirections = (
  runId: string,
  body: { model_id: string; gen_mode: string }
): Promise<DirectionCompareRow[]> =>
  fetch(`/api/ablation/${runId}/directions/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(jsonOrThrow)

const readNdjsonStream = async (
  url: string,
  body: object,
  onEvent: (event: VerifyEvent) => void,
  signal?: AbortSignal,
): Promise<void> => {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!response.ok || !response.body) throw new Error(`verify failed (${response.status})`)
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      if (line.trim()) onEvent(JSON.parse(line))
    }
  }
}

export const verifyAblation = (
  runId: string,
  body: { model_id: string; gen_mode: string; categories?: string[]; samples_per_category?: number },
  onEvent: (event: VerifyEvent) => void,
  signal?: AbortSignal,
): Promise<void> => readNdjsonStream(`/api/ablation/${runId}/verify`, body, onEvent, signal)

export const verifyAblationClassic = (
  runId: string,
  body: { model_id: string; gen_mode: string; factor: number; categories?: string[]; samples_per_category?: number },
  onEvent: (event: VerifyEvent) => void,
  signal?: AbortSignal,
): Promise<void> => readNdjsonStream(`/api/ablation/${runId}/verify/classic`, body, onEvent, signal)
