import { readFileSync, writeFileSync } from 'fs'

const RUN = '/home/ermer/devproj/js/react/ablitMD/data/runs/run_2026-05-17T19-57-17-480Z_8696f34a.json'
const run = JSON.parse(readFileSync(RUN, 'utf8'))
const step = run.sequence[0]               // single step: Qwen3.6-27B / non_thinking
const { model, mode } = step

const classified = []
const unclassified = []
for (const prompt of run.prompts) {
  const result = prompt.model_results?.[model]?.[mode]
  if (!result) continue
  const row = {
    prompt_id: prompt.prompt_id,
    category: prompt.category,
    type: prompt.type,
    response: result.response ?? '',
  }
  if (result.refusal_mode) {
    classified.push({ ...row, refused: result.refused, refusal_mode: result.refusal_mode })
  } else {
    unclassified.push(row)
  }
}

// --- calibration: sample existing classifications per mode ---
console.log(`classified=${classified.length} unclassified=${unclassified.length}`)
const byMode = {}
for (const row of classified) (byMode[row.refusal_mode] ??= []).push(row)
console.log('mode distribution (existing 304):',
  Object.fromEntries(Object.entries(byMode).map(([m, rows]) => [m, rows.length])))

const lens = unclassified.map(r => r.response.length).sort((a, b) => a - b)
console.log(`unclassified response length — min=${lens[0]} median=${lens[Math.floor(lens.length / 2)]} max=${lens[lens.length - 1]}`)

console.log('\n================ CALIBRATION SAMPLES (existing classifications) ================')
for (const m of ['hard', 'redirect', 'disclaimer', 'none']) {
  const rows = byMode[m] ?? []
  console.log(`\n----- mode=${m} (${rows.length} total) — 6 samples -----`)
  for (const row of rows.slice(0, 6)) {
    console.log(`  [${row.type}/${row.category}] ${JSON.stringify(row.response.slice(0, 320))}`)
  }
}

// --- write the 136 unclassified out for batched review ---
writeFileSync('/tmp/unclassified_136.json', JSON.stringify(unclassified, null, 2))
console.log(`\nwrote ${unclassified.length} unclassified rows -> /tmp/unclassified_136.json`)
