import { readFile, writeFile, mkdir } from 'fs/promises'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import { randomUUID } from 'crypto'
import { Run } from '../models/run.js'
import { Direction } from '../models/direction.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const RUNS_DIR = process.env.RUNS_DIR || join(__dirname, '../../data/runs')

const mirrorRunToMongo = (run) => {
  const { direction_results, ...runDoc } = run
  Run.updateOne({ run_id: run.run_id }, runDoc, { upsert: true })
    .catch(err => console.error(`[runs] Mongo Run mirror failed for ${run.run_id}: ${err.message}`))
}

const mirrorDirectionsToMongo = (run) => {
  if (!run.direction_results) return
  for (const [category_id, payload] of Object.entries(run.direction_results)) {
    const doc = { run_id: run.run_id, category_id, ...payload }
    Direction.updateOne({ run_id: run.run_id, category_id }, doc, { upsert: true })
      .catch(err => console.error(`[runs] Mongo Direction mirror failed for ${run.run_id}/${category_id}: ${err.message}`))
  }
}

const mirrorToMongo = (run) => {
  mirrorRunToMongo(run)
  mirrorDirectionsToMongo(run)
}

export const runPath = (run_id) => join(RUNS_DIR, `${run_id}.json`)
export const statesDir = (run_id) => join(RUNS_DIR, run_id)
export const statesPath = (run_id, key) => join(statesDir(run_id), `${key}.npy`)

export const createRun = async ({ models, mode_selection, prompt_scope, sequence, prompts }) => {
  await mkdir(RUNS_DIR, { recursive: true })
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
  const run_id = `run_${timestamp}_${randomUUID().slice(0, 8)}`
  const run = {
    run_id,
    started_at: new Date().toISOString(),
    completed_at: null,
    incomplete: true,
    models,
    mode_selection,
    prompt_scope,
    sequence,
    current_sequence_index: 0,
    prompts,
    direction_results: null,
  }
  await writeFile(runPath(run_id), JSON.stringify(run, null, 2))
  await mkdir(statesDir(run_id), { recursive: true })
  mirrorToMongo(run)
  return run
}

export const readRun = async (run_id) => {
  const content = await readFile(runPath(run_id), 'utf-8')
  return JSON.parse(content)
}

export const writeRun = async (run) => {
  await writeFile(runPath(run.run_id), JSON.stringify(run, null, 2))
  mirrorToMongo(run)
}

export const writePromptResult = async (run_id, prompt_id, model_id, mode, result) => {
  const run = await readRun(run_id)
  const prompt = run.prompts.find(p => p.prompt_id === prompt_id)
  if (!prompt) throw new Error(`Prompt ${prompt_id} not in run ${run_id}`)

  prompt.model_results[model_id] ??= {}
  prompt.model_results[model_id][mode] = result

  await writeRun(run)
  return run
}

export const updateRunField = async (run_id, fields) => {
  const run = await readRun(run_id)
  Object.assign(run, fields)
  await writeRun(run)
  return run
}

export const listRuns = async () => {
  await mkdir(RUNS_DIR, { recursive: true })
  const { readdir } = await import('fs/promises')
  const files = await readdir(RUNS_DIR)

  const summaries = await Promise.all(
    files
      .filter(file => file.endsWith('.json'))
      .map(async (file) => {
        const content = await readFile(join(RUNS_DIR, file), 'utf-8')
        const { prompts, direction_results, ...summary } = JSON.parse(content)
        return { ...summary, prompt_count: prompts.length }
      })
  )

  return summaries.sort((first, second) =>
    new Date(second.started_at) - new Date(first.started_at)
  )
}
