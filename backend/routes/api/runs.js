import { Router } from 'express'
import { Prompt } from '../../models/prompt.js'
import {
  createRun, readRun, writePromptResult, updateRunField, listRuns, stripDirectionVectors,
  listRemoteRunIds, syncRunsFromRemote
} from '../../lib/runs.js'

const SAFE_RUN_ID = /^[A-Za-z0-9._-]+$/

const router = Router()

// Registered before /:runId so "remote" is never read as a run id.
router.get('/remote', async (req, res) => {
  try {
    const runIds = await listRemoteRunIds()
    if (runIds === null) {
      res.status(503).json({ error: 'Remote data server not configured' })
      return
    }
    res.json(runIds)
  } catch (err) {
    res.status(502).json({ error: err.message })
  }
})

router.post('/sync', async (req, res) => {
  const { message, run_id, run_ids } = req.body
  if (message !== 'sync') {
    res.status(400).json({ error: 'Expected message: "sync"' })
    return
  }
  const requestedIds = run_ids ?? (run_id ? [run_id] : [])
  if (requestedIds.length === 0 || !requestedIds.every(id => SAFE_RUN_ID.test(id))) {
    res.status(400).json({ error: 'Provide run_id or run_ids of valid run ids' })
    return
  }
  try {
    const results = await syncRunsFromRemote(requestedIds)
    for (const result of results) stripDirectionVectors(result.run?.direction_results)
    res.json(run_ids ? results : results[0])
  } catch (err) {
    res.status(502).json({ error: err.message })
  }
})

router.get('/', async (req, res) => {
  const runs = await listRuns()
  res.json(runs)
})

router.post('/', async (req, res) => {
  const { models, mode_selection, prompt_scope } = req.body

  const categories = prompt_scope?.categories || []
  const query = categories.length > 0
    ? { category: { $in: categories } }
    : {}

  const dbPrompts = await Prompt.find(query).lean()

  const modes = mode_selection === 'both'
    ? ['non_thinking', 'thinking']
    : [mode_selection]

  const sequence = models.flatMap(model =>
    modes.map(mode => ({ model, mode }))
  )

  const prompts = dbPrompts.map(p => ({
    prompt_id: p._id.toString(),
    text: p.text,
    category: p.category,
    category_group: p.category_group,
    type: p.type,
    triggers: p.triggers,
    model_results: {},
  }))

  const run = await createRun({ models, mode_selection, prompt_scope, sequence, prompts })
  res.status(201).json(run)
})

router.get('/:runId', async (req, res) => {
  try {
    const run = await readRun(req.params.runId)
    // drop the heavy raw direction vectors — charts never read them
    stripDirectionVectors(run.direction_results)
    res.json(run)
  } catch {
    res.status(404).json({ error: 'Run not found' })
  }
})

router.patch('/:runId', async (req, res) => {
  const allowedFields = ['current_sequence_index', 'incomplete', 'completed_at', 'direction_results']
  const fields = Object.fromEntries(
    Object.entries(req.body).filter(([key]) => allowedFields.includes(key))
  )
  const run = await updateRunField(req.params.runId, fields)
  res.json(run)
})

router.patch('/:runId/prompts/:promptId', async (req, res) => {
  const { model_id, mode, result } = req.body
  const run = await writePromptResult(req.params.runId, req.params.promptId, model_id, mode, result)
  res.json(run)
})

export default router
