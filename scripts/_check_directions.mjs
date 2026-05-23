import { readFileSync, existsSync, statSync } from 'fs'

const RUN = 'data/runs/run_2026-05-17T19-57-17-480Z_8696f34a.json'
const SIDE = 'data/runs/run_2026-05-17T19-57-17-480Z_8696f34a.directions.json'
const STATE = 'data/runs/run_2026-05-17T19-57-17-480Z_8696f34a'

const run = JSON.parse(readFileSync(RUN, 'utf8'))
const { model, mode } = run.sequence[0]
const side = JSON.parse(readFileSync(SIDE, 'utf8'))
const computed = new Set(Object.keys(side[model][mode].per_category))

console.log(`sidecar size: ${(statSync(SIDE).size / 1048576).toFixed(2)} MB`)
console.log(`direction_per_layer in sidecar: ${readFileSync(SIDE, 'utf8').includes('direction_per_layer') ? 'PRESENT' : 'absent'}`)
console.log('')

const cats = {}
for (const prompt of run.prompts) {
  const result = prompt.model_results[model][mode]
  const cat = (cats[prompt.category] ??= {
    total: 0, npy: 0,
    allMode: { none: 0, hard: 0, redirect: 0, disclaimer: 0 },
    npyMode: { none: 0, hard: 0, redirect: 0, disclaimer: 0 },
  })
  cat.total += 1
  cat.allMode[result.refusal_mode] = (cat.allMode[result.refusal_mode] ?? 0) + 1
  if (existsSync(`${STATE}/${result.hidden_states_key}.npy`)) {
    cat.npy += 1
    cat.npyMode[result.refusal_mode] = (cat.npyMode[result.refusal_mode] ?? 0) + 1
  }
}

const allCats = Object.keys(cats).sort()
const refusalsOf = (m) => m.hard + m.redirect + m.disclaimer

console.log(`categories in run: ${allCats.length}`)
console.log(`categories with a computed direction: ${computed.size}`)
console.log('')
console.log('CATEGORIES WITHOUT A DIRECTION:')
for (const cat of allCats) {
  if (computed.has(cat)) continue
  const x = cats[cat]
  const reason = refusalsOf(x.allMode) === 0
    ? 'no refusals at all — model complied on every prompt'
    : refusalsOf(x.npyMode) === 0
      ? `${refusalsOf(x.allMode)} refusal(s) exist but ALL lack .npy hidden states`
      : x.npyMode.none === 0
        ? 'has refusals but no not-refused (none) baseline sample'
        : 'unknown'
  console.log(`  ${cat}: total=${x.total} npy=${x.npy}`)
  console.log(`    all modes: none=${x.allMode.none} hard=${x.allMode.hard} redirect=${x.allMode.redirect} disclaimer=${x.allMode.disclaimer}`)
  console.log(`    npy modes: none=${x.npyMode.none} hard=${x.npyMode.hard} redirect=${x.npyMode.redirect} disclaimer=${x.npyMode.disclaimer}`)
  console.log(`    → ${reason}`)
}
