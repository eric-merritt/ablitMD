import mongoose from 'mongoose'
import { readdir } from 'fs/promises'
import { join, dirname } from 'path'
import { fileURLToPath, pathToFileURL } from 'url'
import { CATEGORIES } from '../constants/Categories.js'
import { LLM } from '../models/llm.js'
import { Prompt } from '../models/prompt.js'
import { qwenSeed } from './seeds/llms/qwen.js'
import { qwen359bSeed } from './seeds/llms/qwen35_9b.js'
import { gemmaSeed } from './seeds/llms/gemma.js'
import { llamaSeed } from './seeds/llms/llama.js'
// deepseek parked — MoE, ~1.3TB bf16, infeasible pre-abliteration

const __dirname = dirname(fileURLToPath(import.meta.url))
const REFCATS_DIR = join(__dirname, 'seeds/refCats')
const MONGO_URI = process.env.MONGO_URI
if (!MONGO_URI) throw new Error('MONGO_URI is not set')

const catGroupMap = Object.fromEntries(CATEGORIES.map(cat => [cat.id, cat.group]))

const loadAllPromptSeeds = async () => {
  const entries = await readdir(REFCATS_DIR)
  const allPrompts = []

  for (const entry of entries) {
    if (entry === 'catSeed.js') continue
    const groupDir = join(REFCATS_DIR, entry)
    const files = await readdir(groupDir)

    for (const file of files) {
      if (!file.endsWith('.js')) continue
      const mod = await import(pathToFileURL(join(groupDir, file)).href)
      allPrompts.push(...mod.default)
    }
  }

  return allPrompts
}

const STALE_MODEL_IDS = [
  'meta-llama/Llama-4-Maverick',  // replaced by Llama-3.3-70B-Instruct (MoE, too large)
]

const seedLLMs = async () => {
  for (const staleId of STALE_MODEL_IDS) {
    const removed = await LLM.deleteOne({ modelId: staleId })
    if (removed.deletedCount) console.log(`  - removed stale: ${staleId}`)
  }

  const seeds = [qwenSeed, qwen359bSeed, gemmaSeed, llamaSeed]
  let inserted = 0

  for (const seed of seeds) {
    const existing = await LLM.findOne({ modelId: seed.modelId })
    if (!existing) {
      await seed.save()
      inserted++
      console.log(`  + ${seed.name}`)
    } else {
      console.log(`  = ${seed.name} (already exists)`)
    }
  }

  return inserted
}

const seedPrompts = async () => {
  const rawPrompts = await loadAllPromptSeeds()
  let inserted = 0

  for (const rawPrompt of rawPrompts) {
    if (!rawPrompt.text?.trim()) continue

    const category_group = catGroupMap[rawPrompt.category]
    if (!category_group) {
      console.warn(`  ! Unknown category: ${rawPrompt.category} — skipping`)
      continue
    }

    const existing = await Prompt.findOne({ text: rawPrompt.text, category: rawPrompt.category })
    if (!existing) {
      await Prompt.create({ ...rawPrompt, category_group })
      inserted++
    }
  }

  return inserted
}

const main = async () => {
  await mongoose.connect(MONGO_URI)
  console.log('Connected to MongoDB')

  console.log('\nSeeding LLMs...')
  const llmCount = await seedLLMs()

  console.log('\nSeeding prompts...')
  const promptCount = await seedPrompts()

  console.log(`\nDone. ${llmCount} LLMs inserted, ${promptCount} prompts inserted.`)
  await mongoose.disconnect()
}

main().catch(err => { console.error(err); process.exit(1) })
