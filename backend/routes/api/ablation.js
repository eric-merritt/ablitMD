import { Router } from 'express'
import { spawn } from 'node:child_process'
import { readFile, access } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const router = Router()
const INFERENCE_BASE = process.env.INFERENCE_URL || 'http://localhost:8238'
const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const RUNS_DIR = path.join(PROJECT_ROOT, 'data', 'runs')

const fileExists = async (filePath) => {
  try { await access(filePath); return true } catch { return false }
}

const runPython = (args) => new Promise((resolve, reject) => {
  const proc = spawn('uv', ['run', 'python', ...args], { cwd: PROJECT_ROOT })
  let stderr = ''
  proc.stderr.on('data', chunk => { stderr += chunk })
  proc.on('error', reject)
  proc.on('close', code => code === 0 ? resolve() : reject(new Error(stderr || `exit ${code}`)))
})

// Strip the large direction vectors — the chart only needs ranges, factors, category ids.
const slimRecipe = (recipe) => ({
  run_id: recipe.run_id, model_id: recipe.model_id, gen_mode: recipe.gen_mode,
  onset: recipe.onset, split: recipe.split, last_layer: recipe.last_layer,
  factor_a: recipe.factor_a, factor_b: recipe.factor_b, built_at: recipe.built_at,
  modes: Object.fromEntries(Object.entries(recipe.modes).map(([mode, data]) => [mode, {
    phase_a: { layers: data.phase_a.layers,
               category_ids: Object.keys(data.phase_a.per_category) },
    phase_b: { layers: data.phase_b.layers },
  }])),
})

router.get('/:runId/divergence', async (req, res) => {
  const { runId } = req.params
  if (!await fileExists(path.join(RUNS_DIR, `${runId}.json`))) {
    res.status(404).json({ detail: 'Run not found' }); return
  }
  const divFile = path.join(RUNS_DIR, `${runId}.divergence.json`)
  if (!await fileExists(divFile)) {
    try { await runPython(['scripts/compute_divergence.py', runId]) }
    catch (err) { res.status(500).json({ detail: String(err.message) }); return }
  }
  res.json(JSON.parse(await readFile(divFile, 'utf8')))
})

const buildLocks = new Map()

router.post('/:runId/recipe', async (req, res) => {
  const { runId } = req.params
  const { onset, split, factorA, factorB } = req.body
  if (!await fileExists(path.join(RUNS_DIR, `${runId}.json`))) {
    res.status(404).json({ detail: 'Run not found' }); return
  }
  const prev = buildLocks.get(runId) || Promise.resolve()
  const work = prev.then(async () => {
    await runPython(['scripts/build_recipe.py', runId,
      '--onset', String(onset), '--split', String(split),
      '--factor-a', String(factorA), '--factor-b', String(factorB)])
  })
  buildLocks.set(runId, work.catch(() => {}))
  try {
    await work
  } catch (err) { res.status(500).json({ detail: String(err.message) }); return }
  const recipe = JSON.parse(await readFile(path.join(RUNS_DIR, `${runId}.recipe.json`), 'utf8'))
  res.json(slimRecipe(recipe))
})

const safeJson = async (response) => {
  const text = await response.text()
  try { return JSON.parse(text) } catch { return { detail: text } }
}

const proxyToInference = (inferencePath, buildBody) => async (req, res) => {
  const response = await fetch(`${INFERENCE_BASE}${inferencePath}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(buildBody(req)),
  })
  res.status(response.status).json(await safeJson(response))
}

router.post('/:runId/bake',
  proxyToInference('/ablate/bake', req => ({ run_id: req.params.runId, ...req.body })))

router.post('/:runId/directions/compare',
  proxyToInference('/ablate/directions/compare', req => ({ run_id: req.params.runId, ...req.body })))

const proxyNdjsonStream = (inferencePath) => async (req, res) => {
  const upstream = await fetch(`${INFERENCE_BASE}${inferencePath}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_id: req.params.runId, ...req.body }),
  })
  res.status(upstream.status)
  res.setHeader('Content-Type', 'application/x-ndjson')
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('X-Accel-Buffering', 'no')
  if (!upstream.body) { res.end(); return }
  req.on('close', () => { upstream.body.destroy?.() })
  for await (const chunk of upstream.body) {
    if (!res.write(chunk)) await new Promise(resolve => res.once('drain', resolve))
  }
  res.end()
}

router.post('/:runId/verify', async (req, res) => {
  const pending = buildLocks.get(req.params.runId)
  if (pending) { try { await pending } catch {} }
  return proxyNdjsonStream('/ablate/verify')(req, res)
})

router.post('/:runId/verify/classic', proxyNdjsonStream('/ablate/verify/classic'))

export default router
