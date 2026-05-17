import type { Prompt } from '../types/run'

export const fetchSelectedPrompts = async (
  categories: string[]
): Promise<Prompt[]> => {
  const res = await fetch('/api/prompts/selected', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ categories }),
  })
  if (!res.ok) throw new Error(`fetchSelectedPrompts failed: ${res.status}`)
  return res.json()
}