import type { LLM } from '../types/model'

export const fetchModels = async (): Promise<LLM[]> => {
  const res = await fetch('/api/models')
  if (!res.ok) throw new Error(`fetchModels failed: ${res.status}`)
  return res.json()
}
