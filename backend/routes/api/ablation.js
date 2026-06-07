import { Router } from 'express'
import { spawn } from 'node:child_process'
import { readFile, access, readdir, stat } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { Recipe } from '../../models/recipe.js'

const router = Router()
const INFERENCE_BASE = process.env.INFERENCE_URL || 'http://localhost:8238'
const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const RUNS_DIR = path.join(PROJECT_ROOT, 'data', 'runs')

const fileExists = async (filePath) => {
  try { await access(filePath); return true } catch { return false }
}

// Newest recipe file for a run (timestamped builds + any legacy <runId>.recipe.json).
const latestRecipeFile = async (runId) => {
  const prefix = `${runId}.recipe.`
  const names = (await readdir(RUNS_DIR)).filter(name => name.startsWith(prefix) && name.endsWith('.json'))
  if (names.length === 0) return null
  const timed = await Promise.all(names.map(async name => ({
    path: path.join(RUNS_DIR, name),
    mtime: (await stat(path.join(RUNS_DIR, name))).mtimeMs,
  })))
  timed.sort((a, b) => b.mtime - a.mtime)
  return timed[0].path
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
  factor_a_per_category: recipe.factor_a_per_category || {},
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
  const { onset, split, factorA, factorB, factorAByCategory } = req.body
  if (!await fileExists(path.join(RUNS_DIR, `${runId}.json`))) {
    res.status(404).json({ detail: 'Run not found' }); return
  }
  const perCategoryArgs = factorAByCategory && Object.keys(factorAByCategory).length
    ? ['--factor-a-per-category', JSON.stringify(factorAByCategory)]
    : []
  const prev = buildLocks.get(runId) || Promise.resolve()
  const work = prev.then(async () => {
    await runPython(['scripts/build_recipe.py', runId,
      '--onset', String(onset), '--split', String(split),
      '--factor-a', String(factorA), '--factor-b', String(factorB),
      ...perCategoryArgs])
  })
  buildLocks.set(runId, work.catch(() => {}))
  try {
    await work
  } catch (err) { res.status(500).json({ detail: String(err.message) }); return }
  const recipe = JSON.parse(await readFile(await latestRecipeFile(runId), 'utf8'))
  const slim = slimRecipe(recipe)
  // Reuse warning: has this exact master-factor combo been verified before?
  const prior = await Recipe.findOne(masterKey(runId, recipe)).lean().catch(() => null)
  if (prior) slim.prior_attempt = { categories: prior.categories, verified_at: prior.verified_at }
  res.json(slim)
})

const safeJson = async (response) => {
  const text = await response.text()
  try { return JSON.parse(text) } catch { return { detail: text } }
}

// Master-factor identity of a recipe — what the reuse warning keys on.
const readRecipeMaster = async (runId) => {
  const file = await latestRecipeFile(runId)
  if (!file) return null
  const recipe = JSON.parse(await readFile(file, 'utf8'))
  return {
    onset: recipe.onset, split: recipe.split,
    factor_a: recipe.factor_a, factor_b: recipe.factor_b,
    factor_a_per_category: recipe.factor_a_per_category || {},
  }
}

const masterKey = (runId, master) => ({
  run_id: runId, onset: master.onset, split: master.split,
  factor_a: master.factor_a, factor_b: master.factor_b,
})

// Persist the per-category verify outcome under the recipe's master-factor identity,
// so a later attempt at the same onset/split/factorA/factorB surfaces a warning.
const recordVerifyOutcome = async (runId, outcome) => {
  const master = await readRecipeMaster(runId)
  if (!master) return
  const categories = Object.entries(outcome).map(([category, counts]) => ({ category, ...counts }))
  await Recipe.updateOne(
    masterKey(runId, master),
    { ...masterKey(runId, master), factor_a_per_category: master.factor_a_per_category, categories, verified_at: new Date() },
    { upsert: true },
  )
}

const proxyToInference = (inferencePath, buildBody) => async (req, res) => {
  const response = await fetch(`${INFERENCE_BASE}${inferencePath}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(buildBody(req)),
  })
  res.status(response.status).json(await safeJson(response))
}

router.post('/:runId/bake', async (req, res) => {
  // Always bake the latest recipe: wait out any in-flight rebuild first.
  const pending = buildLocks.get(req.params.runId)
  if (pending) { try { await pending } catch {} }
  return proxyToInference('/ablate/bake', r => ({ run_id: r.params.runId, ...r.body }))(req, res)
})

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

// Like proxyNdjsonStream, but tees the stream: forwards every chunk to the client
// while parsing `prompt` events to tally per-category complied/refused, then records
// the outcome against the recipe's master-factor identity once the stream ends.
const proxyVerifyWithCapture = async (req, res) => {
  const runId = req.params.runId
  const upstream = await fetch(`${INFERENCE_BASE}/ablate/verify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_id: runId, ...req.body }),
  })
  res.status(upstream.status)
  res.setHeader('Content-Type', 'application/x-ndjson')
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('X-Accel-Buffering', 'no')
  if (!upstream.body) { res.end(); return }
  req.on('close', () => { upstream.body.destroy?.() })

  const outcome = {}
  const consume = (line) => {
    if (!line.trim()) return
    let event
    try { event = JSON.parse(line) } catch { return }
    if (event.type === 'prompt' && event.category) {
      const bucket = (outcome[event.category] ??= { complied: 0, refused: 0 })
      if (event.refused_after) bucket.refused++
      else bucket.complied++
    }
  }

  const decoder = new TextDecoder()
  let buffer = ''
  for await (const chunk of upstream.body) {
    if (!res.write(chunk)) await new Promise(resolve => res.once('drain', resolve))
    buffer += decoder.decode(chunk, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) consume(line)
  }
  consume(buffer)
  res.end()
  if (Object.keys(outcome).length) {
    await recordVerifyOutcome(runId, outcome).catch(err =>
      console.error(`[recipe] outcome record failed for ${runId}: ${err.message}`))
  }
}

router.post('/verify/label', async (req, res) => {
  const response = await fetch(`${INFERENCE_BASE}/ablate/verify/label`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req.body),
  })
  res.status(response.status).json(await safeJson(response))
})

router.post('/:runId/verify', async (req, res) => {
  const pending = buildLocks.get(req.params.runId)
  if (pending) { try { await pending } catch {} }
  return proxyVerifyWithCapture(req, res)
})

router.post('/:runId/verify/classic', proxyNdjsonStream('/ablate/verify/classic'))

export default router
