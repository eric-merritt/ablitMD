import { readFileSync } from 'fs'

const SIDE = 'data/runs/run_2026-05-17T19-57-17-480Z_8696f34a.directions.json'
const side = JSON.parse(readFileSync(SIDE, 'utf8'))
const [model] = Object.keys(side)
const [mode] = Object.keys(side[model])
const perCategory = side[model][mode].per_category

const peakOf = (values) => {
  let idx = 0
  for (let i = 1; i < values.length; i += 1) if (values[i] > values[idx]) idx = i
  return { mag: values[idx], layer: idx }
}

const rows = []
for (const [category, result] of Object.entries(perCategory)) {
  const byMode = result.by_mode || {}
  const row = { category }
  for (const refMode of ['hard', 'redirect', 'disclaimer']) {
    const entry = byMode[refMode]
    if (entry) row[refMode] = { n: entry.sample_count, ...peakOf(entry.magnitude_per_layer) }
  }
  row.primary = (row.hard || row.redirect || row.disclaimer).mag
  rows.push(row)
}
rows.sort((a, b) => b.primary - a.primary)

const cell = (m) => m ? `n=${String(m.n).padStart(2)} |mag|=${m.mag.toFixed(2).padStart(5)} @L${String(m.layer).padStart(2)}` : '—'.padEnd(20)

console.log(`${rows.length} categories with a direction — ${model} / ${mode}, sorted by strongest direction\n`)
console.log(`${'#'.padStart(3)}  ${'CATEGORY'.padEnd(26)}  ${'HARD'.padEnd(20)}  ${'REDIRECT'.padEnd(20)}  DISCLAIMER`)
rows.forEach((row, idx) => {
  console.log(
    `${String(idx + 1).padStart(3)}  ${row.category.padEnd(26)}  ${cell(row.hard).padEnd(20)}  ${cell(row.redirect).padEnd(20)}  ${cell(row.disclaimer)}`
  )
})
