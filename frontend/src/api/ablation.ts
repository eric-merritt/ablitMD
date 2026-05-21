import type {
  DivergencePayload, SlimRecipe, RecipeParams, AblationStatus, VerifyEvent,
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

export const applyAblation = (runId: string): Promise<AblationStatus> =>
  fetch(`/api/ablation/${runId}/apply`, { method: 'POST' }).then(jsonOrThrow)

export const clearAblation = (runId: string): Promise<AblationStatus> =>
  fetch(`/api/ablation/${runId}/clear`, { method: 'POST' }).then(jsonOrThrow)

export const bakeModel = (runId: string): Promise<{ saved_to: string }> =>
  fetch(`/api/ablation/${runId}/bake`, { method: 'POST' }).then(jsonOrThrow)

// Verify streams ndjson — onEvent fires per event (total, category_start, prompt, category_result).
export const verifyAblation = async (
  runId: string,
  body: { model_id: string; gen_mode: string; categories?: string[]; samples_per_category?: number },
  onEvent: (event: VerifyEvent) => void,
  signal?: AbortSignal,
): Promise<void> => {
  const response = await fetch(`/api/ablation/${runId}/verify`, {
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
