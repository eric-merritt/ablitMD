import { readFile, writeFile, mkdir, readdir, access } from 'fs/promises'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import { Run } from '../models/run.js'
import { Direction } from '../models/direction.js'
import { runPath, directionsPath, statesDir } from './runs.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const RUNS_DIR = process.env.RUNS_DIR || join(__dirname, '../../data/runs')

const fileExists = async (path) => {
  try { await access(path); return true } catch { return false }
}

const directionResultsFromDocs = (docs) => {
  const results = {}
  for (const doc of docs) {
    const { _id, __v, run_id, category_id, ...payload } = doc
    results[category_id] = payload
  }
  return results
}

// Pull every run + its directions from Mongo to local files, so a fresh instance
// works without rsync. Never clobbers a run that already exists locally. The .npy
// hidden states are intentionally NOT pulled — they stay an rsync/provenance concern,
// and recipe building reads the directions sidecar this writes instead.
export const pullAllFromMongo = async () => {
  await mkdir(RUNS_DIR, { recursive: true })
  const runs = await Run.find().lean()
  let pulled = 0
  for (const run of runs) {
    const runId = run.run_id
    if (await fileExists(runPath(runId))) continue
    const { _id, __v, direction_results, ...slim } = run
    await writeFile(runPath(runId), JSON.stringify(slim, null, 2))
    const docs = await Direction.find({ run_id: runId }).lean()
    if (docs.length) {
      await writeFile(directionsPath(runId), JSON.stringify(directionResultsFromDocs(docs), null, 2))
    }
    await mkdir(statesDir(runId), { recursive: true })
    pulled++
  }
  return pulled
}

const readDirectionsSidecar = async (runId) => {
  try { return JSON.parse(await readFile(directionsPath(runId), 'utf-8')) }
  catch (err) { if (err.code === 'ENOENT') return null; throw err }
}

const localRunIds = async () => {
  const files = await readdir(RUNS_DIR)
  const ids = []
  for (const file of files) {
    if (!file.endsWith('.json') || file.includes('.recipe.')) continue
    if (file.endsWith('.directions.json') || file.endsWith('.verify.json') || file.endsWith('.divergence.json')) continue
    try {
      const run = JSON.parse(await readFile(join(RUNS_DIR, file), 'utf-8'))
      // Skip spoof/test runs — they should never reach the shared Mongo.
      if (run.run_id && run.started_at && run.prompt_scope !== 'spoof') ids.push(run.run_id)
    } catch { /* not a run file */ }
  }
  return ids
}

// Reconcile: upsert every local run AND its directions, regardless of whether the run
// already exists in Mongo. Guards the fire-and-forget write mirrors — any change that
// silently failed to push (a prompt classification, a directions compute) is caught on
// next boot. Upserts are idempotent; the stored directions are stripped/small.
export const pushLocalToMongo = async () => {
  let runsPushed = 0
  let dirsPushed = 0
  for (const runId of await localRunIds()) {
    const run = JSON.parse(await readFile(runPath(runId), 'utf-8'))
    const { direction_results, ...slim } = run
    await Run.updateOne({ run_id: runId }, slim, { upsert: true })
    runsPushed++
    const sidecar = await readDirectionsSidecar(runId)
    if (sidecar) {
      for (const [categoryId, payload] of Object.entries(sidecar)) {
        await Direction.updateOne(
          { run_id: runId, category_id: categoryId },
          { run_id: runId, category_id: categoryId, ...payload },
          { upsert: true },
        )
        dirsPushed++
      }
    }
  }
  return { runsPushed, dirsPushed }
}

// Run once at startup: reconcile-push first (so nothing local is lost), then pull
// anything this instance is missing. Best-effort — logs and swallows so Mongo being
// unreachable never blocks the server.
export const syncWithMongo = async () => {
  try {
    const { runsPushed, dirsPushed } = await pushLocalToMongo()
    const pulled = await pullAllFromMongo()
    console.log(`[sync] startup reconcile: pushed ${runsPushed} run(s) + ${dirsPushed} direction doc(s), pulled ${pulled} run(s)`)
  } catch (err) {
    console.error(`[sync] startup sync failed (continuing): ${err.message}`)
  }
}
