import type { CategoryDirectionResult } from '../types/run'

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

export type GenerateEvent =
  | { type: 'ready'; hidden_states_key: string }
  | { type: 'token'; text: string }
  | { type: 'done'; response: string }
  | { type: 'aborted'; response: string }
  | { type: 'error'; error: string }

export const inferenceGenerateStream = async (
  body: {
    prompt_id: string
    prompt_text: string
    run_id: string
    model_id: string
    mode: string
  },
  onEvent: (event: GenerateEvent) => void,
  signal?: AbortSignal,
): Promise<void> => {
  const res = await fetch('/api/inference/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok || !res.body) throw new Error(`inferenceGenerate failed: ${res.status}`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      const trimmed = line.trim()
      if (trimmed) onEvent(JSON.parse(trimmed) as GenerateEvent)
    }
  }
  const tail = buffer.trim()
  if (tail) onEvent(JSON.parse(tail) as GenerateEvent)
}

export const inferenceCompute = async (body: {
  run_id: string
  model_id: string
  mode: string
}): Promise<Record<string, CategoryDirectionResult>> => {
  const res = await fetch('/api/inference/compute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`inferenceCompute failed: ${res.status}`)
  return res.json()
}
