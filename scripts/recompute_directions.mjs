// One-shot: re-run compute for a single run's sequence step(s), PATCH back into the run.
// Useful when classification is done but direction analysis crashed mid-flight.
// Usage: node scripts/recompute_directions.mjs <run_id>

const runId = process.argv[2]
if (!runId) { console.error('usage: node scripts/recompute_directions.mjs <run_id>'); process.exit(1) }

const BACKEND = 'http://localhost:8237'
const INFERENCE = 'http://localhost:8238'

const fetchJson = async (url, init) => {
  const res = await fetch(url, init)
  if (!res.ok) throw new Error(`${ init?.method || 'GET' } ${ url } → ${ res.status } ${ await res.text() }`)
  return res.json()
}

console.log(`[recompute] loading run ${ runId }...`)
const run = await fetchJson(`${ BACKEND }/api/runs/${ runId }`)
console.log(`[recompute] sequence: ${ JSON.stringify(run.sequence) }`)
console.log(`[recompute] prompts: ${ run.prompts.length }`)

const directions = {}

for (const step of run.sequence) {
  const label = `${ step.model } / ${ step.mode }`
  console.log(`[recompute] computing ${ label }...`)
  const startedAt = Date.now()
  const perCategory = await fetchJson(`${ INFERENCE }/compute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_id: runId, model_id: step.model, mode: step.mode }),
  })
  const elapsedSec = ((Date.now() - startedAt) / 1000).toFixed(1)
  const catCount = Object.keys(perCategory).length
  console.log(`[recompute] ${ label } → ${ catCount } categories in ${ elapsedSec }s`)
  directions[step.model] ??= {}
  directions[step.model][step.mode] = {
    computed_at: new Date().toISOString(),
    per_category: perCategory,
  }
}

console.log(`[recompute] PATCHing run with direction_results...`)
await fetchJson(`${ BACKEND }/api/runs/${ runId }`, {
  method: 'PATCH',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ direction_results: directions }),
})

console.log(`[recompute] done. Refresh the UI to see all categories.`)
