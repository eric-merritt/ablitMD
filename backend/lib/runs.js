import { readFile, writeFile, mkdir, readdir, access } from 'fs/promises'
import { spawn } from 'node:child_process'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import { randomUUID } from 'crypto'
import { Run } from '../models/run.js'
import { Direction } from '../models/direction.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const RUNS_DIR = process.env.RUNS_DIR || join(__dirname, '../../data/runs')
const REMOTE_DATA_BASE = process.env.REMOTE_DATA_BASE
const REMOTE_DATA_KEY = process.env.REMOTE_DATA_KEY

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

const fileExists = async (path) => {
  try { await access(path); return true } catch { return false }
}

// Fetch a whole run (json + sidecars + .npy hidden states) as one tarball from the
// remote data server (eric-merritt.com/ablitMD/data) and extract it in place. Lets a
// zero-touch vast.ai instance pull everything a run needs on open — no rsync.
const fetchRunFromRemote = async (run_id) => {
  if (!REMOTE_DATA_BASE || !REMOTE_DATA_KEY) return false
  const url = `${REMOTE_DATA_BASE}/run/${encodeURIComponent(run_id)}.tar?key=${encodeURIComponent(REMOTE_DATA_KEY)}`
  let response
  try { response = await fetch(url) }
  catch (err) { console.error(`[runs] remote fetch failed for ${run_id}: ${err.message}`); return false }
  if (!response.ok || !response.body) {
    console.error(`[runs] remote fetch ${run_id}: HTTP ${response.status}`)
    return false
  }
  await mkdir(RUNS_DIR, { recursive: true })
  const tar = spawn('tar', ['-xf', '-', '-C', RUNS_DIR])
  const extracted = new Promise((resolve, reject) => {
    tar.on('error', reject)
    tar.on('close', code => code === 0 ? resolve() : reject(new Error(`tar exit ${code}`)))
  })
  try {
    for await (const chunk of response.body) {
      if (!tar.stdin.write(chunk)) await new Promise(drain => tar.stdin.once('drain', drain))
    }
    tar.stdin.end()
    await extracted
  } catch (err) { console.error(`[runs] extract failed for ${run_id}: ${err.message}`); return false }
  return fileExists(runPath(run_id))
}

// Run ids on the remote data server, for the Merge Run Data picker. Null when the
// remote isn't configured (e.g. running on the home box itself).
export const listRemoteRunIds = async () => {
  if (!REMOTE_DATA_BASE || !REMOTE_DATA_KEY) return null
  const url = `${REMOTE_DATA_BASE}/runs?key=${encodeURIComponent(REMOTE_DATA_KEY)}`
  const response = await fetch(url)
  if (!response.ok) throw new Error(`remote run list: HTTP ${response.status}`)
  return response.json()
}

// Sync each requested run from the remote data server onto local disk, then read it
// back. Returns { run_id, ok, run } per entry so one bad id doesn't sink the batch.
export const syncRunsFromRemote = async (runIds) => {
  const results = []
  for (const runId of runIds) {
    const fetched = await fetchRunFromRemote(runId)
    results.push(fetched
      ? { run_id: runId, ok: true, run: await readRun(runId) }
      : { run_id: runId, ok: false, run: null })
  }
  return results
}

// Pull a single run (and its directions) from Mongo to disk — used when opening/resuming
// a run that isn't local, so you don't have to rsync run data to a fresh instance.
const pullRunFromMongo = async (run_id) => {
  const doc = await Run.findOne({ run_id }).lean()
  if (!doc) return false
  const { _id, __v, direction_results, ...slim } = doc
  await mkdir(RUNS_DIR, { recursive: true })
  await writeFile(runPath(run_id), JSON.stringify(slim, null, 2))
  const dirs = await Direction.find({ run_id }).lean()
  if (dirs.length) {
    const results = {}
    for (const dir of dirs) {
      const { _id, __v, run_id: rid, category_id, ...payload } = dir
      results[category_id] = payload
    }
    await writeDirectionsSidecar(run_id, results)
  }
  await mkdir(statesDir(run_id), { recursive: true })
  return true
}

export const readRun = async (run_id) => {
  // Zero-touch fetch when a run isn't local: full tarball (incl .npy) from the remote
  // data server first, Mongo (metadata + stripped directions) as fallback.
  if (!await fileExists(runPath(run_id))) {
    const fetched = await fetchRunFromRemote(run_id)
    if (!fetched) await pullRunFromMongo(run_id)
  }
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

  // Only genuine run files: `<run_id>.json` has exactly one dot (run ids carry no dots).
  // Every sidecar — `.recipe.<ts>.json`, `.directions.json`, `.verify.json`,
  // `.divergence.json` — has more; the multi-MB recipe files would otherwise OOM the heap.
  const runFiles = files.filter(file => /^[^.]+\.json$/.test(file))

  // Sequential, not Promise.all: a legacy run can be hundreds of MB (bundled directions),
  // and parsing several at once blows the heap. One file in flight caps peak memory.
  const summaries = []
  for (const file of runFiles) {
    const content = await readFile(join(RUNS_DIR, file), 'utf-8')
    const { prompts, direction_results, ...summary } = JSON.parse(content)
    summaries.push({
      run_id: summary.run_id ?? file.replace(/\.json$/, ''),
      started_at: summary.started_at ?? null,
      completed_at: summary.completed_at ?? null,
      incomplete: summary.incomplete ?? true,
      models: Array.isArray(summary.models) ? summary.models : [],
      mode_selection: summary.mode_selection ?? null,
      prompt_count: prompts?.length ?? 0,
    })
  }

  return summaries
    .filter(run => run.started_at)
    .sort((first, second) => new Date(second.started_at) - new Date(first.started_at))
}
