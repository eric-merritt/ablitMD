import { readFileSync } from 'fs'

const RUN_PATH = '/home/ermer/devproj/js/react/ablitMD/data/runs/run_2026-05-17T19-57-17-480Z_8696f34a.json'
const UNCLASSIFIED = '/tmp/unclassified_136.json'

// same MODES array order as _classify_apply.mjs
const MODES = [
  'hard','none','none','none','none','none','redirect','redirect','redirect','redirect',
  'redirect','none','none','none','none','none','redirect','redirect','redirect','redirect',
  'redirect','none','none','none','none','none','redirect','redirect','redirect','redirect',
  'redirect','none','none','none','none','none','redirect','hard','redirect','redirect',
  'redirect','none','none','none','none','none','redirect','redirect','hard','none',
  'hard','none','none','none','none','none','hard','hard','hard','redirect',
  'hard','none','none','none','none','disclaimer','redirect','redirect','redirect','redirect',
  'redirect','none','none','none','none','none','hard','hard','hard','hard',
  'hard','none','none','none','none','none','hard','redirect','redirect','hard',
  'redirect','none','none','none','none','none','redirect','redirect','redirect','redirect',
  'redirect','none','none','none','none','none','redirect','redirect','redirect','redirect',
  'redirect','none','none','none','none','none','redirect','redirect','redirect','redirect',
  'redirect','none','none','disclaimer','none','none',
]

const unclassified = JSON.parse(readFileSync(UNCLASSIFIED, 'utf8'))
const run = JSON.parse(readFileSync(RUN_PATH, 'utf8'))
const textById = new Map(run.prompts.map(prompt => [prompt.prompt_id, prompt.text]))

const noneIdx = unclassified.map((_, idx) => idx).filter(idx => MODES[idx] === 'none')

// random sample of 10 (Fisher–Yates partial shuffle)
const pool = [...noneIdx]
const pick = []
for (let drawn = 0; drawn < 10 && pool.length; drawn += 1) {
  const swapAt = Math.floor(Math.random() * pool.length)
  pick.push(pool[swapAt])
  pool[swapAt] = pool[pool.length - 1]
  pool.pop()
}
pick.sort((first, second) => first - second)

console.log(`10 random non-refusals (of ${noneIdx.length} classified 'none'):\n`)
for (const idx of pick) {
  const row = unclassified[idx]
  const promptText = textById.get(row.prompt_id) ?? '(prompt text not found)'
  const response = row.response ?? ''
  const shown = response.length > 1000 ? `${response.slice(0, 1000)} …[+${response.length - 1000} chars]` : response
  console.log('═'.repeat(78))
  console.log(`idx ${idx} · ${row.prompt_id} · ${row.type} / ${row.category}`)
  console.log('─'.repeat(78))
  console.log(`PROMPT:   ${promptText}`)
  console.log('─'.repeat(78))
  console.log(`RESPONSE: ${shown}`)
  console.log('')
}
