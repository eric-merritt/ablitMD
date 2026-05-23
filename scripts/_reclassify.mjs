import { readFileSync, writeFileSync } from 'fs'

const RUN_PATH = '/home/ermer/devproj/js/react/ablitMD/data/runs/run_2026-05-17T19-57-17-480Z_8696f34a.json'
const UNCLASSIFIED = '/tmp/unclassified_136.json'

// corrections — keyed by index into the unclassified dump → new refusal_mode
// realtime_information declines are capability caveats ("I can't see live data"),
// not moral refusals — the whole 46-51 cluster is the same disclaimer shape.
const OVERRIDES = {
  46: 'disclaimer',
  47: 'disclaimer',
  48: 'disclaimer',
  49: 'disclaimer',
  50: 'disclaimer',
  51: 'disclaimer',
}

const unclassified = JSON.parse(readFileSync(UNCLASSIFIED, 'utf8'))
const run = JSON.parse(readFileSync(RUN_PATH, 'utf8'))
const { model, mode } = run.sequence[0]
const promptById = new Map(run.prompts.map(prompt => [prompt.prompt_id, prompt]))

for (const [idx, newMode] of Object.entries(OVERRIDES)) {
  const row = unclassified[Number(idx)]
  if (!row) throw new Error(`no unclassified row at idx ${idx}`)
  const result = promptById.get(row.prompt_id)?.model_results?.[model]?.[mode]
  if (!result) throw new Error(`no result for idx ${idx} (${row.prompt_id})`)
  const was = result.refusal_mode
  result.refusal_mode = newMode
  result.refused = newMode !== 'none'
  console.log(`idx ${idx} (${row.prompt_id}, ${row.category}): ${was} -> ${newMode}`)
}

writeFileSync(RUN_PATH, JSON.stringify(run, null, 2))

const tally = { hard: 0, redirect: 0, disclaimer: 0, none: 0 }
for (const prompt of run.prompts) {
  const refusalMode = prompt.model_results?.[model]?.[mode]?.refusal_mode
  if (refusalMode in tally) tally[refusalMode] += 1
}
console.log('full run tally (440):', JSON.stringify(tally))
