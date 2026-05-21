import { Router } from 'express'
import { Prompt } from '../../models/prompt.js'
import {
  createRun, readRun, writePromptResult, updateRunField, listRuns, stripDirectionVectors
} from '../../lib/runs.js'

const router = Router()

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
