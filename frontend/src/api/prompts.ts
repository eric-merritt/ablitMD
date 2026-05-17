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