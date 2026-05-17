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
