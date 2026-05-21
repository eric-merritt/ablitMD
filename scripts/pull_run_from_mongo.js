import 'dotenv/config'
import mongoose from 'mongoose'
import { writeFile, mkdir } from 'fs/promises'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import { Run } from '../backend/models/run.js'
import { Direction } from '../backend/models/direction.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const RUN_ID = process.argv[2]
const OUT_DIR = process.argv[3] || join(__dirname, '../data/runs')

if (!RUN_ID) { console.error('usage: pull_run_from_mongo.js <run_id> [out_dir]'); process.exit(1) }

const main = async () => {
  if (!process.env.MONGO_URI) throw new Error('MONGO_URI not set')
  await mongoose.connect(process.env.MONGO_URI, { serverSelectionTimeoutMS: 10000 })

  const run = await Run.findOne({ run_id: RUN_ID }).lean()
  if (!run) throw new Error(`Run ${RUN_ID} not found in Mongo`)

  const directions = await Direction.find({ run_id: RUN_ID }).lean()
  console.log(`[pull] run found — ${run.prompts?.length || 0} prompts, ${directions.length} direction docs`)

  const direction_results = {}
  for (const direction of directions) {
    const { _id, __v, run_id, category_id, ...payload } = direction
    direction_results[category_id] = payload
  }

  const { _id, __v, ...runDoc } = run
  runDoc.prompts = (runDoc.prompts || []).map(prompt => ({
    ...prompt,
    model_results: prompt.model_results ?? {},
  }))
  runDoc.direction_results = directions.length > 0 ? direction_results : null

  await mkdir(OUT_DIR, { recursive: true })
  const outPath = join(OUT_DIR, `${RUN_ID}.json`)
  await writeFile(outPath, JSON.stringify(runDoc, null, 2))

  const classified = (run.prompts || []).filter(prompt =>
    Object.values(prompt.model_results || {}).some(modeMap =>
      Object.values(modeMap).some(result => result?.refusal_mode)
    )
  ).length
  console.log(`[pull] wrote ${outPath} — ${classified} classified prompts`)

  await mongoose.disconnect()
}

main().catch(err => { console.error(err); process.exit(1) })
