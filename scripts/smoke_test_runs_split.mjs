// Smoke-test: confirms the direction_results split fixes the silent-die bug.
// Simulates the exact failure pattern: a run with bundled direction_results,
// followed by patchRunPrompt calls that previously rewrote 267MB per call.

import 'dotenv/config'
import { mkdtemp, rm, readFile, writeFile, stat } from 'fs/promises'
import { tmpdir } from 'os'
import { join } from 'path'

process.env.RUNS_DIR = await mkdtemp(join(tmpdir(), 'ablitmd-smoke-'))

const { createRun, readRun, writePromptResult, updateRunField, runPath, directionsPath } =
  await import('../backend/lib/runs.js')

const fail = (msg) => { console.error('FAIL:', msg); process.exit(1) }
const pass = (msg) => console.log('PASS:', msg)
const fileSize = async (path) => (await stat(path)).size

const sampleRun = () => ({
  models: ['Qwen/Qwen3.6-27B'],
  mode_selection: 'non_thinking',
  prompt_scope: { categories: ['x'] },
  sequence: [{ model: 'Qwen/Qwen3.6-27B', mode: 'non_thinking' }],
  prompts: Array.from({ length: 50 }, (_, idx) => ({
    prompt_id: `p${ idx }`,
    text: 'test',
    category: 'x',
    category_group: 'x',
    type: 'harmful',
    triggers: [],
    model_results: {},
  })),
})

const fakeDirections = () => {
  const oneCategory = {
    alignment: Array.from({ length: 50000 }, () => Math.random()),
    by_mode: { non_thinking: Array.from({ length: 10000 }, () => Math.random()) },
    computed_at: new Date().toISOString(),
  }
  return Object.fromEntries(
    Array.from({ length: 14 }, (_, idx) => [`cat_${ idx }`, oneCategory])
  )
}

const run = await createRun(sampleRun())
console.log(`[smoke] run_id: ${ run.run_id }`)
const baseSize = await fileSize(runPath(run.run_id))
console.log(`[smoke] empty run JSON: ${ (baseSize / 1024).toFixed(1) } KB`)

// inject a legacy 267MB-style bundle: write direction_results directly into the run file
const bundled = { ...run, direction_results: fakeDirections() }
await writeFile(runPath(run.run_id), JSON.stringify(bundled, null, 2))
const bundledSize = await fileSize(runPath(run.run_id))
console.log(`[smoke] bundled legacy run: ${ (bundledSize / 1e6).toFixed(1) } MB`)
if (bundledSize < 10 * 1e6) fail('bundled fixture should be >10MB to simulate the real failure')

// readRun on a bundled file should migrate directions into the sidecar
const loaded = await readRun(run.run_id)
if (!loaded.direction_results) fail('readRun lost direction_results on legacy bundle')
const sidecarSize = await fileSize(directionsPath(run.run_id))
pass(`readRun migrated bundle → sidecar (${ (sidecarSize / 1e6).toFixed(1) } MB)`)

// patchRunPrompt should NOT re-write directions into the main run JSON
const before = Date.now()
for (let idx = 0; idx < 50; idx++) {
  await writePromptResult(run.run_id, `p${ idx }`, 'Qwen/Qwen3.6-27B', 'non_thinking', {
    response: 'I cannot help with that.',
    hidden_states_key: `k_${ idx }`,
    refused: true,
    refusal_mode: 'hard',
    classified_at: new Date().toISOString(),
  })
}
const elapsed = Date.now() - before
const finalSize = await fileSize(runPath(run.run_id))
console.log(`[smoke] 50× patchRunPrompt: ${ elapsed }ms total — main JSON now ${ (finalSize / 1024).toFixed(1) } KB`)

if (finalSize > 5 * 1e6) fail(`main run JSON should stay <5MB after patches — got ${ (finalSize / 1e6).toFixed(1) } MB. direction_results is still bundled.`)
pass('main JSON stayed lean across 50 patches')

// updateRunField with direction_results should write only the sidecar, not bundle
const newDirs = fakeDirections()
await updateRunField(run.run_id, { direction_results: newDirs, incomplete: false, completed_at: new Date().toISOString() })
const finalMainSize = await fileSize(runPath(run.run_id))
const finalSidecarSize = await fileSize(directionsPath(run.run_id))
console.log(`[smoke] after updateRunField with directions: main=${ (finalMainSize / 1024).toFixed(1) } KB, sidecar=${ (finalSidecarSize / 1e6).toFixed(1) } MB`)
if (finalMainSize > 5 * 1e6) fail('updateRunField re-bundled directions into main run JSON')
pass('updateRunField wrote directions to sidecar, kept main JSON lean')

// listRuns must not surface the sidecar file as its own run
const { listRuns } = await import('../backend/lib/runs.js')
const runs = await listRuns()
const runFiles = runs.filter(r => r.run_id.endsWith('.directions'))
if (runFiles.length > 0) fail('listRuns surfaced .directions.json as a separate run')
pass('listRuns ignores .directions.json sidecars')

// readRun after split must merge directions back
const merged = await readRun(run.run_id)
if (!merged.direction_results || Object.keys(merged.direction_results).length !== 14) {
  fail('readRun did not merge directions sidecar correctly')
}
pass('readRun merges sidecar back into run object')

await rm(process.env.RUNS_DIR, { recursive: true, force: true })
console.log('\n[smoke] ALL PASS — submit path no longer rewrites the directions blob on every prompt patch')
process.exit(0)
