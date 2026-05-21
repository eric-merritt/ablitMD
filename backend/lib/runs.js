import { readFile, writeFile, mkdir, readdir } from 'fs/promises'
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

const mirrorDirectionsToMongo = (run_id, direction_results) => {
  if (!direction_results) return
  for (const [category_id, payload] of Object.entries(direction_results)) {
    const doc = { run_id, category_id, ...payload }
    Direction.updateOne({ run_id, category_id }, doc, { upsert: true })
      .catch(err => console.error(`[runs] Mongo Direction mirror failed for ${run_id}/${category_id}: ${err.message}`))
  }
}

// raw per-layer direction vectors (direction_per_layer) are large — ~150 MB for this
// run — and read by no chart. strip them so the run payload sent to the browser stays
// small. walks the direction_results tree and mutates the object in place.
export const stripDirectionVectors = (node) => {
  if (Array.isArray(node)) {
    for (const item of node) stripDirectionVectors(item)
  } else if (node && typeof node === 'object') {
    for (const [key, value] of Object.entries(node)) {
      if (key === 'direction_per_layer') delete node[key]
      else stripDirectionVectors(value)
    }
  }
  return node
}

export const runPath = (run_id) => join(RUNS_DIR, `${run_id}.json`)
export const directionsPath = (run_id) => join(RUNS_DIR, `${run_id}.directions.json`)
export const statesDir = (run_id) => join(RUNS_DIR, run_id)
export const statesPath = (run_id, key) => join(statesDir(run_id), `${key}.npy`)

const readDirectionsSidecar = async (run_id) => {
  try {
    return JSON.parse(await readFile(directionsPath(run_id), 'utf-8'))
  } catch (err) {
    if (err.code === 'ENOENT') return null
    throw err
  }
}

const writeDirectionsSidecar = async (run_id, direction_results) => {
  await writeFile(directionsPath(run_id), JSON.stringify(direction_results, null, 2))
}

// reads the slim run document only — no direction_results sidecar load.
// use this on hot paths (per-prompt patch) where directions are dead weight.
const readRunSlim = async (run_id) => {
  const content = await readFile(runPath(run_id), 'utf-8')
  return JSON.parse(content)
}

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
  mirrorRunToMongo(run)
  return run
}

export const readRun = async (run_id) => {
  const run = await readRunSlim(run_id)

  // legacy runs bundled direction_results into the main file. Preserve them in a sidecar
  // the first time we touch the file, then let writeRun strip them on next write.
  if (run.direction_results) {
    const existing = await readDirectionsSidecar(run_id)
    if (existing == null) await writeDirectionsSidecar(run_id, run.direction_results)
  } else {
    run.direction_results = await readDirectionsSidecar(run_id)
  }
  return run
}

// writes only the slim run document — direction_results is never bundled into the run JSON.
// callers that change direction_results must call writeDirectionsSidecar themselves.
export const writeRun = async (run) => {
  const { direction_results, ...slim } = run
  await writeFile(runPath(run.run_id), JSON.stringify(slim, null, 2))
  mirrorRunToMongo(slim)
}

export const writePromptResult = async (run_id, prompt_id, model_id, mode, result) => {
  const run = await readRunSlim(run_id)
  const prompt = run.prompts.find(p => p.prompt_id === prompt_id)
  if (!prompt) throw new Error(`Prompt ${prompt_id} not in run ${run_id}`)

  prompt.model_results[model_id] ??= {}
  prompt.model_results[model_id][mode] = result

  await writeRun(run)
  return run
}

export const updateRunField = async (run_id, fields) => {
  const run = await readRunSlim(run_id)
  const { direction_results, ...otherFields } = fields
  Object.assign(run, otherFields)

  if (direction_results !== undefined) {
    if (direction_results === null) {
      run.direction_results = null
    } else {
      await writeDirectionsSidecar(run_id, direction_results)
      mirrorDirectionsToMongo(run_id, direction_results)
      run.direction_results = direction_results
    }
  } else {
    run.direction_results = await readDirectionsSidecar(run_id)
  }

  await writeRun(run)
  return run
}

export const listRuns = async () => {
  await mkdir(RUNS_DIR, { recursive: true })
  const files = await readdir(RUNS_DIR)

  const summaries = await Promise.all(
    files
      .filter(file => file.endsWith('.json') && !file.endsWith('.directions.json'))
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
