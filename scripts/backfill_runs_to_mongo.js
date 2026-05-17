import 'dotenv/config'
import mongoose from 'mongoose'
import { stat } from 'fs/promises'
import { createReadStream } from 'fs'
import { Run } from '../backend/models/run.js'
import { Direction } from '../backend/models/direction.js'

const RUN_FILE = process.argv[2]
if (!RUN_FILE) { console.error('usage: backfill_runs_to_mongo.js <run.json>'); process.exit(1) }

const readJsonStreaming = (path) => new Promise((resolve, reject) => {
  const chunks = []
  createReadStream(path)
    .on('data', chunk => chunks.push(chunk))
    .on('end', () => {
      try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf-8'))) }
      catch (err) { reject(err) }
    })
    .on('error', reject)
})

const main = async () => {
  if (!process.env.MONGO_URI) throw new Error('MONGO_URI not set')
  await mongoose.connect(process.env.MONGO_URI, { serverSelectionTimeoutMS: 10000 })

  const { size } = await stat(RUN_FILE)
  console.log(`[backfill] reading ${RUN_FILE} (${(size / 1024 / 1024).toFixed(1)} MB)`)
  const run = await readJsonStreaming(RUN_FILE)

  const { direction_results, ...runDoc } = run
  console.log(`[backfill] upserting Run ${run.run_id} (${run.prompts?.length || 0} prompts)`)
  await Run.updateOne({ run_id: run.run_id }, runDoc, { upsert: true })

  if (direction_results) {
    const categories = Object.keys(direction_results)
    console.log(`[backfill] upserting ${categories.length} direction docs`)
    let done = 0
    for (const category_id of categories) {
      const doc = { run_id: run.run_id, category_id, ...direction_results[category_id] }
      try {
        await Direction.updateOne({ run_id: run.run_id, category_id }, doc, { upsert: true })
        done++
        console.log(`[backfill]   ${done}/${categories.length} ${category_id}`)
      } catch (err) {
        console.error(`[backfill]   FAILED ${category_id}: ${err.message}`)
      }
    }
  }

  console.log(`[backfill] done`)
  await mongoose.disconnect()
}

main().catch(err => { console.error(err); process.exit(1) })
