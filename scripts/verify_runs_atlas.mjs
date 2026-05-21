import 'dotenv/config'
import mongoose from 'mongoose'
import { Run } from '/home/ermer/devproj/js/react/ablitMD/backend/models/run.js'
import { Direction } from '/home/ermer/devproj/js/react/ablitMD/backend/models/direction.js'

const targets = [
  'run_2026-05-17T19-57-17-480Z_8696f34a',
  'run_2026-05-18T01-14-01-984Z_5396f873',
]

const main = async () => {
  if (!process.env.MONGO_URI) throw new Error('MONGO_URI not set')
  const masked = process.env.MONGO_URI.replace(/\/\/[^@]+@/, '//<creds>@')
  console.log('[atlas] connecting:', masked)
  await mongoose.connect(process.env.MONGO_URI, { serverSelectionTimeoutMS: 10000 })

  for (const run_id of targets) {
    const run = await Run.findOne({ run_id }).lean()
    const directionCount = await Direction.countDocuments({ run_id })
    if (!run) {
      console.log(`  MISSING — ${run_id}`)
      continue
    }
    const classified = (run.prompts || []).filter(prompt =>
      Object.values(prompt.model_results || {}).some(modeMap =>
        Object.values(modeMap).some(result => result?.refusal_mode)
      )
    ).length
    console.log(`  FOUND   — ${run_id}`)
    console.log(`           prompts=${run.prompts?.length ?? 0} classified=${classified} directions=${directionCount} incomplete=${run.incomplete}`)
  }

  await mongoose.disconnect()
}

main().catch(err => { console.error(err); process.exit(1) })
