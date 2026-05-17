# Prompt Testing UI — Backend & Python Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the REST API and Python inference service that power the prompt testing UI —
loading LLMs, running prompts with hidden state capture, storing per-run results, and
computing per-category refusal direction geometry post-run.

**Architecture:** Express 5 on :8199 serves all API routes, proxying inference calls to
a FastAPI service on :8200. Express owns run JSON manifests in `./data/runs/`. Python owns
hidden state `.npy` files in `./data/runs/<run_id>/`. Direction computation runs in Python
after classification is complete; results are written back into the run JSON by Express.

**Tech Stack:** Express 5 (ESM, `"type":"module"`), Mongoose 9, Vitest + supertest +
mongodb-memory-server; Python FastAPI + uvicorn, HuggingFace transformers (bfloat16,
cuda:0), numpy, pytest + pytest-asyncio + httpx

---

## File Map

**Created**
- `backend/lib/runs.js` — run JSON CRUD + hidden state dir management
- `backend/routes/api/runs.js` — /api/runs/* routes
- `backend/routes/api/inference.js` — proxy routes to :8200
- `backend/db/seed.js` — seed runner for LLMs + prompts with category_group
- `backend/tests/runs.test.js` — run I/O unit tests
- `backend/tests/routes.test.js` — route integration tests
- `backend/vitest.config.js` — vitest config for ESM
- `backend/inference/__init__.py`
- `backend/inference/model_loader.py`
- `backend/inference/generator.py`
- `backend/inference/direction.py`
- `backend/inference/service.py`
- `backend/inference/tests/__init__.py`
- `backend/inference/tests/test_direction.py`
- `backend/inference/tests/test_service.py`
- `data/runs/.gitkeep`

**Modified**
- `package.json` — add `concurrently` devDep, add `dev` script
- `.gitignore` — keep `data/runs/.gitkeep`, ignore `data/runs/*/` and `data/runs/*.json`
- `backend/bin/www` — fix PORT bitwise-OR bug, fix createServer call
- `backend/app.js` — full Express app with MongoDB connection, all routes mounted
- `backend/models/prompt.js` — add `category_group` field, update `refusalMode` enum
- `backend/routes/api/model.js` — implement GET /api/models (was empty)
- `backend/package.json` — add vitest, supertest, mongodb-memory-server devDeps
- `pyproject.toml` — add fastapi, uvicorn, numpy, httpx, pytest, pytest-asyncio

---

## Task 1: Project Setup

**Files:**
- Modify: `package.json`
- Modify: `backend/bin/www`
- Modify: `backend/package.json`
- Modify: `.gitignore`
- Create: `backend/vitest.config.js`
- Create: `data/runs/.gitkeep`

- [ ] **Step 1: Fix `backend/bin/www`**

The current file has two bugs: `|` (bitwise OR) instead of `||` for the PORT default,
and incorrect `http.createServer` signature.

```js
#!/usr/bin/env node
import http from 'http'
import app from '../app.js'

const PORT = process.env.PORT || 8199

const server = http.createServer(app)
server.listen(PORT, () => console.log(`backend listening on :${PORT}`))
```

- [ ] **Step 2: Add test infrastructure to `backend/package.json`**

```json
{
  "name": "ablitmd-backend",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "node --watch bin/www",
    "start": "node bin/www",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "express": "^5.2.1",
    "express-async-handler": "^1.2.0",
    "mongoose": "^9.6.2"
  },
  "devDependencies": {
    "@types/express": "^5.0.3",
    "@types/node": "^25.7.0",
    "mongodb-memory-server": "^10.0.0",
    "supertest": "^7.0.0",
    "vitest": "^2.0.0"
  }
}
```

- [ ] **Step 3: Create `backend/vitest.config.js`**

```js
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: { environment: 'node' },
})
```

- [ ] **Step 4: Add `concurrently` to root `package.json` and add `dev` script**

```json
{
  "name": "ablitmd",
  "version": "0.1.0",
  "private": true,
  "workspaces": ["frontend", "backend"],
  "scripts": {
    "dev": "concurrently --names backend,inference,frontend --prefix-colors cyan,yellow,magenta \"npm run dev --workspace=backend\" \"uv run python backend/inference/service.py\" \"npm run dev --workspace=frontend\"",
    "dev:frontend": "npm run dev --workspace=frontend",
    "dev:backend": "npm run dev --workspace=backend"
  },
  "devDependencies": {
    "concurrently": "^9.0.0"
  }
}
```

- [ ] **Step 5: Install root devDependencies**

Run: `npm install` from project root.
Expected: `node_modules/concurrently` appears.

- [ ] **Step 6: Install backend devDependencies**

Run: `npm install --workspace=backend` from project root.
Expected: vitest, supertest, mongodb-memory-server appear in `backend/node_modules`.

- [ ] **Step 7: Update `.gitignore`**

Add to existing `.gitignore` (create it at project root if it doesn't exist):

```
# Run data — keep .gitkeep, ignore actual run files
data/runs/*.json
data/runs/*/

# Python
__pycache__/
*.pyc
.venv/
```

- [ ] **Step 8: Create `data/runs/.gitkeep`**

Run: `mkdir -p data/runs && touch data/runs/.gitkeep`

- [ ] **Step 9: Commit**

```bash
git add package.json backend/package.json backend/bin/www backend/vitest.config.js data/runs/.gitkeep .gitignore
git commit -m "chore: project setup — concurrently dev script, test infra, fix bin/www

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 2: Prompt Schema Update

**Files:**
- Modify: `backend/models/prompt.js`

- [ ] **Step 1: Write the failing test** — `backend/tests/prompt-schema.test.js`

```js
import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import mongoose from 'mongoose'
import { MongoMemoryServer } from 'mongodb-memory-server'
import { Prompt } from '../models/prompt.js'

let mongod

beforeAll(async () => {
  mongod = await MongoMemoryServer.create()
  await mongoose.connect(mongod.getUri())
})

afterAll(async () => {
  await mongoose.disconnect()
  await mongod.stop()
})

describe('Prompt schema', () => {
  it('requires category_group field', async () => {
    const prompt = new Prompt({
      category: 'dangerous_activity',
      type: 'harmful',
      text: 'test prompt',
      triggers: [],
    })
    await expect(prompt.validate()).rejects.toThrow('category_group')
  })

  it('saves with category_group', async () => {
    const prompt = new Prompt({
      category: 'dangerous_activity',
      category_group: 'violence_physical_harm',
      type: 'harmful',
      text: 'test prompt',
      triggers: [],
    })
    await prompt.save()
    const found = await Prompt.findById(prompt._id)
    expect(found.category_group).toBe('violence_physical_harm')
  })

  it('rejects invalid refusalMode', async () => {
    const result = new Prompt({
      category: 'dangerous_activity',
      category_group: 'violence_physical_harm',
      type: 'harmful',
      text: 'test',
      triggers: [],
      results: [{ model: new mongoose.Types.ObjectId(), refused: true, refusalMode: 'soft' }],
    })
    await expect(result.validate()).rejects.toThrow()
  })

  it('accepts valid refusalMode values', async () => {
    for (const mode of ['hard', 'redirect', 'disclaimer', 'none']) {
      const prompt = new Prompt({
        category: 'dangerous_activity',
        category_group: 'violence_physical_harm',
        type: 'harmful',
        text: 'test',
        triggers: [],
        results: [{ model: new mongoose.Types.ObjectId(), refused: true, refusalMode: mode }],
      })
      await expect(prompt.validate()).resolves.toBeUndefined()
    }
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test --workspace=backend`
Expected: FAIL — `category_group` field not found on schema.

- [ ] **Step 3: Update `backend/models/prompt.js`**

```js
import { Schema, model } from 'mongoose'

const resultSchema = new Schema({
  model: { type: Schema.Types.ObjectId, ref: 'LLM', required: true },
  refused: { type: Boolean, required: true },
  refusalMode: { type: String, enum: ['hard', 'redirect', 'disclaimer', 'none'] },
  activeLayers: [{ type: Number }],
  directionSimilarity: { type: Number },
  refusalWeight: { type: Number },
  testedAt: { type: Date, default: Date.now },
}, { _id: false })

const promptSchema = new Schema({
  category: { type: String, required: true, index: true },
  category_group: { type: String, required: true, index: true },
  type: { type: String, enum: ['harmful', 'harmless'], required: true },
  text: { type: String, required: true },
  triggers: [{ type: String }],
  results: [resultSchema],
  addedAt: { type: Date, default: Date.now },
})

export const Prompt = model('Prompt', promptSchema)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test --workspace=backend`
Expected: PASS — all 4 prompt schema tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/models/prompt.js backend/tests/prompt-schema.test.js
git commit -m "feat: add category_group to Prompt schema, update refusalMode enum

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 3: Seed Runner

**Files:**
- Create: `backend/db/seed.js`

The seed runner dynamically loads all refCat prompt seed files, derives `category_group`
from `CATEGORIES`, and upserts both LLMs and Prompts into MongoDB.

- [ ] **Step 1: Create `backend/db/seed.js`**

```js
import mongoose from 'mongoose'
import { readdir } from 'fs/promises'
import { join, dirname } from 'path'
import { fileURLToPath, pathToFileURL } from 'url'
import { CATEGORIES } from '../constants/Categories.js'
import { LLM } from '../models/llm.js'
import { Prompt } from '../models/prompt.js'
import { qwenSeed } from './seeds/llms/qwen.js'
import { gemmaSeed } from './seeds/llms/gemma.js'
import { llamaSeed } from './seeds/llms/llama.js'
import { deepseekSeed } from './seeds/llms/deepseek.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const REFCATS_DIR = join(__dirname, 'seeds/refCats')
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/ablitmd'

const catGroupMap = Object.fromEntries(CATEGORIES.map(cat => [cat.id, cat.group]))

const loadAllPromptSeeds = async () => {
  const groups = await readdir(REFCATS_DIR)
  const allPrompts = []

  for (const entry of groups) {
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

const seedLLMs = async () => {
  const seeds = [qwenSeed, gemmaSeed, llamaSeed, deepseekSeed]
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
```

- [ ] **Step 2: Add seed script to `backend/package.json`**

Add to `scripts`:
```json
"seed": "node db/seed.js"
```

- [ ] **Step 3: Verify seed runner runs without errors**

Ensure MongoDB is running locally. Run:
```bash
npm run seed --workspace=backend
```
Expected output:
```
Connected to MongoDB
Seeding LLMs...
  + Qwen3.6 27B
  + Gemma 4 31B IT
  + Llama 4 Maverick
  + DeepSeek V4 Pro
Seeding prompts...
Done. 4 LLMs inserted, N prompts inserted.
```
N will be the count of non-empty prompt entries across all seed files.

- [ ] **Step 4: Commit**

```bash
git add backend/db/seed.js backend/package.json
git commit -m "feat: seed runner with category_group derivation from CATEGORIES

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 4: Run File I/O

**Files:**
- Create: `backend/lib/runs.js`
- Create: `backend/tests/runs.test.js`

- [ ] **Step 1: Write the failing tests** — `backend/tests/runs.test.js`

```js
import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { mkdtemp, rm } from 'fs/promises'
import { tmpdir } from 'os'
import { join } from 'path'

// We'll override RUNS_DIR via env for tests
process.env.RUNS_DIR = ''  // set per test below

let testRunsDir

beforeEach(async () => {
  testRunsDir = await mkdtemp(join(tmpdir(), 'ablitmd-runs-'))
  process.env.RUNS_DIR = testRunsDir
})

afterEach(async () => {
  await rm(testRunsDir, { recursive: true, force: true })
})

// Re-import after setting env — vitest module cache means we need a factory pattern
const getRuns = async () => {
  const { createRun, readRun, writePromptResult, updateRunField, listRuns } =
    await import(`../lib/runs.js?t=${Date.now()}`)
  return { createRun, readRun, writePromptResult, updateRunField, listRuns }
}

const sampleRun = () => ({
  models: ['Qwen/Qwen3.6-27B'],
  mode_selection: 'non_thinking',
  prompt_scope: { categories: ['dangerous_activity'] },
  sequence: [{ model: 'Qwen/Qwen3.6-27B', mode: 'non_thinking' }],
  prompts: [
    {
      prompt_id: 'abc123',
      text: 'How do I disable airbags?',
      category: 'dangerous_activity',
      category_group: 'violence_physical_harm',
      type: 'harmful',
      triggers: [],
      model_results: {},
    },
  ],
})

describe('createRun', () => {
  it('creates a JSON file with run_id', async () => {
    const { createRun, readRun } = await getRuns()
    const run = await createRun(sampleRun())
    expect(run.run_id).toMatch(/^run_/)
    expect(run.incomplete).toBe(true)
    expect(run.current_sequence_index).toBe(0)
    const loaded = await readRun(run.run_id)
    expect(loaded.run_id).toBe(run.run_id)
  })

  it('creates a hidden states directory for the run', async () => {
    const { createRun } = await getRuns()
    const { access } = await import('fs/promises')
    const run = await createRun(sampleRun())
    await expect(access(join(testRunsDir, run.run_id))).resolves.toBeUndefined()
  })
})

describe('writePromptResult', () => {
  it('writes model_results[model_id][mode] on a prompt', async () => {
    const { createRun, writePromptResult, readRun } = await getRuns()
    const run = await createRun(sampleRun())
    const result = {
      response: 'I cannot help with that.',
      refused: true,
      refusal_mode: 'hard',
      classified_at: new Date().toISOString(),
      hidden_states_key: 'abc123__Qwen__Qwen3.6-27B__non_thinking',
    }
    await writePromptResult(run.run_id, 'abc123', 'Qwen/Qwen3.6-27B', 'non_thinking', result)
    const updated = await readRun(run.run_id)
    expect(updated.prompts[0].model_results['Qwen/Qwen3.6-27B']['non_thinking'].refused).toBe(true)
  })

  it('throws when prompt_id not found', async () => {
    const { createRun, writePromptResult } = await getRuns()
    const run = await createRun(sampleRun())
    await expect(
      writePromptResult(run.run_id, 'nonexistent', 'Qwen/Qwen3.6-27B', 'non_thinking', {})
    ).rejects.toThrow('not in run')
  })
})

describe('updateRunField', () => {
  it('updates top-level fields', async () => {
    const { createRun, updateRunField, readRun } = await getRuns()
    const run = await createRun(sampleRun())
    await updateRunField(run.run_id, { current_sequence_index: 1, incomplete: false })
    const updated = await readRun(run.run_id)
    expect(updated.current_sequence_index).toBe(1)
    expect(updated.incomplete).toBe(false)
  })
})

describe('listRuns', () => {
  it('returns runs sorted newest first', async () => {
    const { createRun, listRuns } = await getRuns()
    await createRun(sampleRun())
    await createRun(sampleRun())
    const runs = await listRuns()
    expect(runs).toHaveLength(2)
    expect(new Date(runs[0].started_at) >= new Date(runs[1].started_at)).toBe(true)
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test --workspace=backend`
Expected: FAIL — `../lib/runs.js` module not found.

- [ ] **Step 3: Create `backend/lib/runs.js`**

```js
import { readFile, writeFile, mkdir } from 'fs/promises'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import { randomUUID } from 'crypto'

const __dirname = dirname(fileURLToPath(import.meta.url))
const RUNS_DIR = process.env.RUNS_DIR || join(__dirname, '../../data/runs')

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
  return run
}

export const readRun = async (run_id) => {
  const content = await readFile(runPath(run_id), 'utf-8')
  return JSON.parse(content)
}

export const writeRun = async (run) => {
  await writeFile(runPath(run.run_id), JSON.stringify(run, null, 2))
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test --workspace=backend`
Expected: PASS — all run I/O tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/lib/runs.js backend/tests/runs.test.js
git commit -m "feat: run file I/O utilities (create, read, update, list)

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 5: Express Routes — Models & Prompts

**Files:**
- Modify: `backend/routes/api/model.js`
- Create: `backend/routes/api/prompts.js`

- [ ] **Step 1: Write failing route tests** — `backend/tests/routes.test.js`

```js
import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest'
import mongoose from 'mongoose'
import { MongoMemoryServer } from 'mongodb-memory-server'
import request from 'supertest'
import app from '../app.js'
import { LLM } from '../models/llm.js'
import { Prompt } from '../models/prompt.js'

let mongod

beforeAll(async () => {
  mongod = await MongoMemoryServer.create()
  await mongoose.connect(mongod.getUri())
})

afterAll(async () => {
  await mongoose.disconnect()
  await mongod.stop()
})

beforeEach(async () => {
  await LLM.deleteMany({})
  await Prompt.deleteMany({})
})

describe('GET /api/models', () => {
  it('returns empty array when no models seeded', async () => {
    const res = await request(app).get('/api/models')
    expect(res.status).toBe(200)
    expect(res.body).toEqual([])
  })

  it('returns seeded models', async () => {
    await LLM.create({
      name: 'Test Model', modelId: 'test/model', vendor: 'Test', apiProvider: 'self_hosted',
      apiModelId: 'test/model',
      architecture: { numLayers: 10, hiddenSize: 512, architectureType: 'transformer' },
    })
    const res = await request(app).get('/api/models')
    expect(res.status).toBe(200)
    expect(res.body).toHaveLength(1)
    expect(res.body[0].modelId).toBe('test/model')
  })
})

describe('GET /api/prompts/all', () => {
  it('returns all prompts', async () => {
    await Prompt.create([
      { category: 'dangerous_activity', category_group: 'violence_physical_harm', type: 'harmful', text: 'p1', triggers: [] },
      { category: 'hate_speech', category_group: 'hate_discrimination', type: 'harmless', text: 'p2', triggers: [] },
    ])
    const res = await request(app).get('/api/prompts/all')
    expect(res.status).toBe(200)
    expect(res.body).toHaveLength(2)
  })
})

describe('POST /api/prompts/selected', () => {
  beforeEach(async () => {
    await Prompt.create([
      { category: 'dangerous_activity', category_group: 'violence_physical_harm', type: 'harmful', text: 'p1', triggers: [] },
      { category: 'hate_speech', category_group: 'hate_discrimination', type: 'harmful', text: 'p2', triggers: [] },
      { category: 'medical_advice', category_group: 'professional_advice', type: 'harmless', text: 'p3', triggers: [] },
    ])
  })

  it('returns prompts matching selected categories', async () => {
    const res = await request(app)
      .post('/api/prompts/selected')
      .send({ categories: ['dangerous_activity', 'hate_speech'] })
    expect(res.status).toBe(200)
    expect(res.body).toHaveLength(2)
    expect(res.body.map(p => p.category)).toEqual(
      expect.arrayContaining(['dangerous_activity', 'hate_speech'])
    )
  })

  it('returns prompts matching selected groups', async () => {
    const res = await request(app)
      .post('/api/prompts/selected')
      .send({ groups: ['violence_physical_harm'] })
    expect(res.status).toBe(200)
    expect(res.body).toHaveLength(1)
    expect(res.body[0].category).toBe('dangerous_activity')
  })

  it('returns 400 when neither categories nor groups provided', async () => {
    const res = await request(app).post('/api/prompts/selected').send({})
    expect(res.status).toBe(400)
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test --workspace=backend`
Expected: FAIL — routes not yet implemented.

- [ ] **Step 3: Implement `backend/routes/api/model.js`**

```js
import { Router } from 'express'
import { LLM } from '../../models/llm.js'

const router = Router()

router.get('/', async (req, res) => {
  const models = await LLM.find({}).lean()
  res.json(models)
})

export default router
```

- [ ] **Step 4: Create `backend/routes/api/prompts.js`**

```js
import { Router } from 'express'
import { Prompt } from '../../models/prompt.js'

const router = Router()

router.get('/all', async (req, res) => {
  const prompts = await Prompt.find({}).lean()
  res.json(prompts)
})

router.post('/selected', async (req, res) => {
  const { categories, groups } = req.body

  const hasCats = Array.isArray(categories) && categories.length > 0
  const hasGroups = Array.isArray(groups) && groups.length > 0

  if (!hasCats && !hasGroups) {
    return res.status(400).json({ error: 'Provide categories or groups array' })
  }

  const query = {}
  if (hasCats && hasGroups) {
    query.$or = [{ category: { $in: categories } }, { category_group: { $in: groups } }]
  } else if (hasCats) {
    query.category = { $in: categories }
  } else {
    query.category_group = { $in: groups }
  }

  const prompts = await Prompt.find(query).lean()
  res.json(prompts)
})

export default router
```

- [ ] **Step 5: Run tests to verify models + prompts routes pass**

Run: `npm test --workspace=backend`
Expected: PASS on all model and prompt route tests (runs tests may warn about missing app routes — that's fine, they'll pass after Task 7).

- [ ] **Step 6: Commit**

```bash
git add backend/routes/api/model.js backend/routes/api/prompts.js backend/tests/routes.test.js
git commit -m "feat: /api/models and /api/prompts/all + /selected routes

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 6: Express Routes — Runs

**Files:**
- Create: `backend/routes/api/runs.js`

- [ ] **Step 1: Add runs route tests to `backend/tests/routes.test.js`**

Append to the existing describe blocks:

```js
describe('POST /api/runs', () => {
  it('creates a run and returns run_id', async () => {
    const prompts = await Prompt.insertMany([
      { category: 'dangerous_activity', category_group: 'violence_physical_harm', type: 'harmful', text: 'p1', triggers: [] },
    ])
    const res = await request(app)
      .post('/api/runs')
      .send({
        models: ['Qwen/Qwen3.6-27B'],
        mode_selection: 'non_thinking',
        prompt_scope: { categories: ['dangerous_activity'] },
      })
    expect(res.status).toBe(201)
    expect(res.body.run_id).toMatch(/^run_/)
    expect(res.body.incomplete).toBe(true)
    expect(res.body.prompts).toHaveLength(1)
  })
})

describe('GET /api/runs', () => {
  it('returns run list', async () => {
    const res = await request(app).get('/api/runs')
    expect(res.status).toBe(200)
    expect(Array.isArray(res.body)).toBe(true)
  })
})

describe('GET /api/runs/:runId', () => {
  it('returns 404 for unknown run', async () => {
    const res = await request(app).get('/api/runs/nonexistent')
    expect(res.status).toBe(404)
  })
})
```

- [ ] **Step 2: Create `backend/routes/api/runs.js`**

```js
import { Router } from 'express'
import { Prompt } from '../../models/prompt.js'
import {
  createRun, readRun, writePromptResult, updateRunField, listRuns
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
```

- [ ] **Step 3: Run tests to verify runs routes pass**

Run: `npm test --workspace=backend`
Expected: PASS — run route tests pass (app must be wired in Task 7 first; this step verifies the router logic in isolation is sound).

- [ ] **Step 4: Commit**

```bash
git add backend/routes/api/runs.js backend/tests/routes.test.js
git commit -m "feat: /api/runs CRUD routes

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 7: Express Routes — Inference Proxy

**Files:**
- Create: `backend/routes/api/inference.js`

The inference proxy forwards requests to the Python FastAPI service on :8200.
It uses native `fetch` (available in Node 18+).

- [ ] **Step 1: Create `backend/routes/api/inference.js`**

```js
import { Router } from 'express'

const router = Router()
const INFERENCE_BASE = process.env.INFERENCE_URL || 'http://localhost:8200'

const proxyPost = (path) => async (req, res) => {
  const response = await fetch(`${INFERENCE_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req.body),
  })
  const data = await response.json()
  res.status(response.status).json(data)
}

router.get('/status', async (req, res) => {
  const response = await fetch(`${INFERENCE_BASE}/status`)
  const data = await response.json()
  res.json(data)
})

router.post('/load', proxyPost('/load'))
router.post('/generate', proxyPost('/generate'))
router.post('/compute', proxyPost('/compute'))

export default router
```

- [ ] **Step 2: Commit**

```bash
git add backend/routes/api/inference.js
git commit -m "feat: inference proxy routes to Python :8200

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 8: App.js — Full Wiring

**Files:**
- Modify: `backend/app.js`

- [ ] **Step 1: Rewrite `backend/app.js`**

```js
import express from 'express'
import mongoose from 'mongoose'
import modelsRouter from './routes/api/model.js'
import promptsRouter from './routes/api/prompts.js'
import runsRouter from './routes/api/runs.js'
import inferenceRouter from './routes/api/inference.js'

const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/ablitmd'

mongoose.connect(MONGO_URI)
  .then(() => console.log('MongoDB connected'))
  .catch(err => console.error('MongoDB connection error:', err))

const app = express()

app.use(express.json())

app.use('/api/models', modelsRouter)
app.use('/api/prompts', promptsRouter)
app.use('/api/runs', runsRouter)
app.use('/api/inference', inferenceRouter)

app.use((err, req, res, next) => {
  console.error(err)
  res.status(500).json({ error: err.message || 'Internal server error' })
})

export default app
```

- [ ] **Step 2: Run full test suite**

Run: `npm test --workspace=backend`
Expected: PASS — all tests pass with app fully wired.

- [ ] **Step 3: Smoke test the running server**

Run: `npm run dev --workspace=backend` in one terminal, then in another:
```bash
curl http://localhost:8199/api/models
```
Expected: `[]` (empty array, MongoDB connected, no models seeded yet).

- [ ] **Step 4: Commit**

```bash
git add backend/app.js
git commit -m "feat: complete Express app wiring — all routes mounted

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 9: Python Service — Dependencies & Scaffold

**Files:**
- Modify: `pyproject.toml`
- Create: `backend/inference/__init__.py`
- Create: `backend/inference/tests/__init__.py`

- [ ] **Step 1: Update `pyproject.toml`**

```toml
[project]
name = "ablitMD"
version = "0.1.0"
description = "Multi-dimensional categorical abliteration research sandbox"
requires-python = ">=3.10"
dependencies = [
    "torch",
    "transformers>=4.40",
    "tqdm",
    "accelerate",
    "huggingface-hub",
    "datasets>=4.8.5",
    "fastapi>=0.115.0",
    "uvicorn>=0.32.0",
    "numpy>=1.26.0",
    "httpx>=0.27.0",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
]

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu118" }

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: Sync dependencies**

Run: `uv sync`
Expected: fastapi, uvicorn, numpy, httpx, pytest-asyncio appear in `.venv`.

- [ ] **Step 3: Create package init files**

```bash
touch backend/inference/__init__.py backend/inference/tests/__init__.py
mkdir -p backend/inference/tests
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock backend/inference/__init__.py backend/inference/tests/__init__.py
git commit -m "chore: add Python inference service dependencies

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 10: Python Direction Computation

Direction computation is pure math — no GPU needed. Test it first.

**Files:**
- Create: `backend/inference/direction.py`
- Create: `backend/inference/tests/test_direction.py`

- [ ] **Step 1: Write failing tests** — `backend/inference/tests/test_direction.py`

```python
import numpy as np
import pytest
from backend.inference.direction import compute_direction, compute_similarity, compute_run_directions


def make_states(n_prompts: int, n_layers: int, hidden: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_prompts, n_layers, hidden)).astype(np.float32)


class TestComputeDirection:
    def test_output_shape(self):
        refusal = make_states(5, 10, 64, seed=0)
        non_refusal = make_states(5, 10, 64, seed=1)
        direction = compute_direction(refusal, non_refusal)
        assert direction.shape == (10, 64)

    def test_direction_is_unit_normalized(self):
        refusal = make_states(5, 10, 64, seed=0)
        non_refusal = make_states(5, 10, 64, seed=1)
        direction = compute_direction(refusal, non_refusal)
        norms = np.linalg.norm(direction, axis=1)
        np.testing.assert_allclose(norms, np.ones(10), atol=1e-5)

    def test_zero_diff_layer_returns_zeros(self):
        # If harmful and harmless means are identical, direction should be zero
        states = make_states(3, 4, 8, seed=0)
        direction = compute_direction(states, states)
        np.testing.assert_array_equal(direction, np.zeros((4, 8)))

    def test_single_prompt_each_side(self):
        refusal = make_states(1, 6, 32, seed=2)
        non_refusal = make_states(1, 6, 32, seed=3)
        direction = compute_direction(refusal, non_refusal)
        assert direction.shape == (6, 32)


class TestComputeSimilarity:
    def test_output_shape(self):
        hidden_state = make_states(1, 10, 64, seed=0)[0]
        direction = make_states(1, 10, 64, seed=1)[0]
        similarity = compute_similarity(hidden_state, direction)
        assert similarity.shape == (10,)

    def test_identical_vectors_similarity_is_one(self):
        vec = make_states(1, 5, 32, seed=0)[0]
        similarity = compute_similarity(vec, vec)
        np.testing.assert_allclose(similarity, np.ones(5), atol=1e-5)

    def test_opposite_vectors_similarity_is_minus_one(self):
        vec = make_states(1, 5, 32, seed=0)[0]
        similarity = compute_similarity(vec, -vec)
        np.testing.assert_allclose(similarity, -np.ones(5), atol=1e-5)

    def test_range_is_minus_one_to_one(self):
        hidden_state = make_states(1, 20, 128, seed=4)[0]
        direction = make_states(1, 20, 128, seed=5)[0]
        similarity = compute_similarity(hidden_state, direction)
        assert (similarity >= -1.0 - 1e-5).all()
        assert (similarity <= 1.0 + 1e-5).all()


class TestComputeRunDirections:
    def test_groups_by_classification(self):
        # Prompts: 2 refused, 2 not refused, 1 refused
        hidden_states = {
            'p1__model__non_thinking': make_states(1, 4, 16, seed=0)[0],
            'p2__model__non_thinking': make_states(1, 4, 16, seed=1)[0],
            'p3__model__non_thinking': make_states(1, 4, 16, seed=2)[0],
            'p4__model__non_thinking': make_states(1, 4, 16, seed=3)[0],
        }
        classifications = [
            { 'hidden_states_key': 'p1__model__non_thinking', 'refused': True },
            { 'hidden_states_key': 'p2__model__non_thinking', 'refused': False },
            { 'hidden_states_key': 'p3__model__non_thinking', 'refused': True },
            { 'hidden_states_key': 'p4__model__non_thinking', 'refused': False },
        ]
        result = compute_run_directions(hidden_states, classifications, category='test_cat')
        assert 'direction_per_layer' in result
        assert 'similarity_per_prompt' in result
        assert len(result['similarity_per_prompt']) == 4
        # Each similarity is a list of floats, one per layer
        for key, sims in result['similarity_per_prompt'].items():
            assert len(sims) == 4

    def test_returns_none_when_insufficient_data(self):
        # Only refused prompts, no non-refused — can't compute direction
        hidden_states = {
            'p1__model__non_thinking': make_states(1, 4, 16, seed=0)[0],
        }
        classifications = [
            { 'hidden_states_key': 'p1__model__non_thinking', 'refused': True },
        ]
        result = compute_run_directions(hidden_states, classifications, category='test_cat')
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest backend/inference/tests/test_direction.py -v`
Expected: FAIL — `backend.inference.direction` module not found.

- [ ] **Step 3: Create `backend/inference/direction.py`**

```python
import numpy as np
from datetime import timezone, datetime


def compute_direction(
  refusal_states: np.ndarray,
  non_refusal_states: np.ndarray
) -> np.ndarray:
  """
  Compute per-layer refusal direction vectors.
  refusal_states: (n_prompts, n_layers, hidden_size)
  non_refusal_states: (n_prompts, n_layers, hidden_size)
  Returns: (n_layers, hidden_size) unit-normalized direction per layer.
  Zero vector for layers where the means are identical.
  """
  refusal_mean = refusal_states.mean(axis=0)
  non_refusal_mean = non_refusal_states.mean(axis=0)
  diff = refusal_mean - non_refusal_mean
  norms = np.linalg.norm(diff, axis=1, keepdims=True)
  safe_norms = np.where(norms < 1e-8, 1.0, norms)
  direction = np.where(norms < 1e-8, np.zeros_like(diff), diff / safe_norms)
  return direction.astype(np.float32)


def compute_similarity(
  hidden_state: np.ndarray,
  direction: np.ndarray
) -> np.ndarray:
  """
  Compute cosine similarity between a prompt's hidden state and the direction.
  hidden_state: (n_layers, hidden_size)
  direction: (n_layers, hidden_size)
  Returns: (n_layers,) cosine similarity per layer, range [-1, 1].
  """
  hs_norms = np.linalg.norm(hidden_state, axis=1, keepdims=True)
  dir_norms = np.linalg.norm(direction, axis=1, keepdims=True)
  safe_hs = np.where(hs_norms < 1e-8, 1.0, hs_norms)
  safe_dir = np.where(dir_norms < 1e-8, 1.0, dir_norms)
  hs_norm = hidden_state / safe_hs
  dir_norm = direction / safe_dir
  return np.clip((hs_norm * dir_norm).sum(axis=1), -1.0, 1.0).astype(np.float32)


def compute_run_directions(
  hidden_states: dict[str, np.ndarray],
  classifications: list[dict],
  category: str
) -> dict | None:
  """
  Compute direction and per-prompt similarity for one (model, mode, category) group.
  hidden_states: key → (n_layers, hidden_size) array
  classifications: list of { hidden_states_key, refused: bool }
  Returns direction_results dict or None if insufficient data.
  """
  refused_keys = [c['hidden_states_key'] for c in classifications if c['refused']]
  non_refused_keys = [c['hidden_states_key'] for c in classifications if not c['refused']]

  if len(refused_keys) == 0 or len(non_refused_keys) == 0:
    return None

  refused_stack = np.stack([hidden_states[key] for key in refused_keys if key in hidden_states])
  non_refused_stack = np.stack([hidden_states[key] for key in non_refused_keys if key in hidden_states])

  if refused_stack.shape[0] == 0 or non_refused_stack.shape[0] == 0:
    return None

  direction = compute_direction(refused_stack, non_refused_stack)

  similarity_per_prompt = {}
  for classification in classifications:
    key = classification['hidden_states_key']
    if key in hidden_states:
      sims = compute_similarity(hidden_states[key], direction)
      similarity_per_prompt[key] = sims.tolist()

  return {
    'computed_at': datetime.now(timezone.utc).isoformat(),
    'direction_per_layer': direction.tolist(),
    'similarity_per_prompt': similarity_per_prompt,
  }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest backend/inference/tests/test_direction.py -v`
Expected: PASS — all direction computation tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/inference/direction.py backend/inference/tests/test_direction.py
git commit -m "feat: refusal direction computation — compute_direction, compute_similarity

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 11: Python Model Loader

**Files:**
- Create: `backend/inference/model_loader.py`

This module holds the single loaded model in module-level state. Only one model
at a time. Loading is blocking and expensive — expected.

- [ ] **Step 1: Create `backend/inference/model_loader.py`**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_loaded_model_id: str | None = None
_model = None
_tokenizer = None
DEVICE = "cuda:0"


def get_loaded_model_id() -> str | None:
  return _loaded_model_id


def load_model(model_id: str, api_model_id: str) -> None:
  global _loaded_model_id, _model, _tokenizer

  if _loaded_model_id == model_id:
    return  # already loaded

  unload_model()

  print(f"Loading model {api_model_id} on {DEVICE}...")
  _model = AutoModelForCausalLM.from_pretrained(
    api_model_id,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
    trust_remote_code=True,
  )
  _tokenizer = AutoTokenizer.from_pretrained(api_model_id, trust_remote_code=True)
  _loaded_model_id = model_id
  print(f"Model loaded: {model_id}")


def unload_model() -> None:
  global _loaded_model_id, _model, _tokenizer
  if _model is not None:
    del _model
    del _tokenizer
    torch.cuda.empty_cache()
    _model = None
    _tokenizer = None
    _loaded_model_id = None


def get_model():
  if _model is None:
    raise RuntimeError("No model loaded. Call /load first.")
  return _model


def get_tokenizer():
  if _tokenizer is None:
    raise RuntimeError("No tokenizer loaded. Call /load first.")
  return _tokenizer
```

- [ ] **Step 2: Commit**

```bash
git add backend/inference/model_loader.py
git commit -m "feat: Python model loader — single-slot GPU model management

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 12: Python Generator — Inference + Hidden State Capture

**Files:**
- Create: `backend/inference/generator.py`

Two forward passes per prompt: one to capture hidden states at the last input token,
one to generate the response text. Thinking mode uses `enable_thinking` in the chat template.

- [ ] **Step 1: Create `backend/inference/generator.py`**

```python
import gc
import re
from pathlib import Path

import numpy as np
import torch

from backend.inference.model_loader import get_model, get_tokenizer, DEVICE


THINKING_STRIP_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


@torch.inference_mode()
def run_prompt(
  prompt_text: str,
  mode: str,
  run_id: str,
  hidden_states_key: str,
  runs_dir: Path,
) -> str:
  """
  Run a single prompt through the loaded model.
  Captures hidden states at last input token position.
  Saves hidden states as .npy file.
  Returns the response text.
  """
  model = get_model()
  tokenizer = get_tokenizer()
  enable_thinking = mode == 'thinking'

  messages = [{"role": "user", "content": prompt_text}]

  try:
    input_ids = tokenizer.apply_chat_template(
      conversation=messages,
      add_generation_prompt=True,
      return_tensors="pt",
      enable_thinking=enable_thinking,
    ).to(DEVICE)
  except TypeError:
    # Model tokenizer doesn't support enable_thinking — fall back
    input_ids = tokenizer.apply_chat_template(
      conversation=messages,
      add_generation_prompt=True,
      return_tensors="pt",
    ).to(DEVICE)

  # Forward pass to capture hidden states (no generation)
  output = model(input_ids, output_hidden_states=True)
  n_layers = model.config.num_hidden_layers

  hidden_states = np.array([
    output.hidden_states[layer_idx][0, -1, :].cpu().float().numpy()
    for layer_idx in range(n_layers + 1)
  ], dtype=np.float32)

  del output
  torch.cuda.empty_cache()

  # Save hidden states
  state_dir = runs_dir / run_id
  state_dir.mkdir(parents=True, exist_ok=True)
  np.save(str(state_dir / f"{hidden_states_key}.npy"), hidden_states)

  # Generate response
  try:
    gen_output = model.generate(
      input_ids,
      max_new_tokens=512,
      do_sample=False,
    )
  except Exception:
    gen_output = model.generate(input_ids, max_new_tokens=512, do_sample=False)

  response = tokenizer.decode(gen_output[0][input_ids.shape[1]:], skip_special_tokens=True)

  # Strip thinking tokens from response if present
  response = THINKING_STRIP_RE.sub('', response).strip()

  del gen_output
  torch.cuda.empty_cache()
  gc.collect()

  return response
```

- [ ] **Step 2: Commit**

```bash
git add backend/inference/generator.py
git commit -m "feat: Python generator — hidden state capture + response generation

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 13: Python FastAPI Service

**Files:**
- Create: `backend/inference/service.py`
- Create: `backend/inference/tests/test_service.py`

- [ ] **Step 1: Write failing service tests** — `backend/inference/tests/test_service.py`

```python
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock
from backend.inference.service import app


@pytest.fixture
def client():
  return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestStatus:
  async def test_returns_loaded_model_none_initially(self, client):
    async with client as c:
      res = await c.get("/status")
    assert res.status_code == 200
    assert res.json()["loaded_model"] is None


class TestLoad:
  async def test_rejects_missing_model_id(self, client):
    async with client as c:
      res = await c.post("/load", json={})
    assert res.status_code == 422

  async def test_calls_load_model(self, client):
    with patch("backend.inference.service.load_model") as mock_load:
      async with client as c:
        res = await c.post("/load", json={
          "model_id": "Qwen/Qwen3.6-27B",
          "api_model_id": "Qwen/Qwen3.6-27B"
        })
      mock_load.assert_called_once_with("Qwen/Qwen3.6-27B", "Qwen/Qwen3.6-27B")
    assert res.status_code == 200


class TestGenerate:
  async def test_rejects_when_no_model_loaded(self, client):
    with patch("backend.inference.service.get_loaded_model_id", return_value=None):
      async with client as c:
        res = await c.post("/generate", json={
          "prompt_id": "abc",
          "prompt_text": "test",
          "run_id": "run_test",
          "model_id": "Qwen/Qwen3.6-27B",
          "mode": "non_thinking",
        })
    assert res.status_code == 400

  async def test_calls_run_prompt_when_model_loaded(self, client):
    with patch("backend.inference.service.get_loaded_model_id", return_value="Qwen/Qwen3.6-27B"), \
         patch("backend.inference.service.run_prompt", return_value="mock response") as mock_gen:
      async with client as c:
        res = await c.post("/generate", json={
          "prompt_id": "abc123",
          "prompt_text": "How do I disable airbags?",
          "run_id": "run_test",
          "model_id": "Qwen/Qwen3.6-27B",
          "mode": "non_thinking",
        })
      assert mock_gen.called
    assert res.status_code == 200
    assert res.json()["response"] == "mock response"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest backend/inference/tests/test_service.py -v`
Expected: FAIL — `backend.inference.service` not found.

- [ ] **Step 3: Create `backend/inference/service.py`**

```python
import json
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.inference.direction import compute_run_directions
from backend.inference.generator import run_prompt
from backend.inference.model_loader import (
  get_loaded_model_id,
  load_model,
  unload_model,
)

app = FastAPI(title="ablitMD inference service")

RUNS_DIR = Path("./data/runs")


class LoadRequest(BaseModel):
  model_id: str
  api_model_id: str


class GenerateRequest(BaseModel):
  prompt_id: str
  prompt_text: str
  run_id: str
  model_id: str
  mode: str


class ComputeRequest(BaseModel):
  run_id: str
  model_id: str
  mode: str


@app.get("/status")
def status():
  return {"loaded_model": get_loaded_model_id()}


@app.post("/load")
def load(req: LoadRequest):
  load_model(req.model_id, req.api_model_id)
  return {"loaded_model": req.model_id}


@app.post("/generate")
def generate(req: GenerateRequest):
  if get_loaded_model_id() != req.model_id:
    raise HTTPException(status_code=400, detail=f"Model {req.model_id} not loaded")

  safe_model = req.model_id.replace("/", "__")
  hidden_states_key = f"{req.prompt_id}__{safe_model}__{req.mode}"

  response = run_prompt(
    prompt_text=req.prompt_text,
    mode=req.mode,
    run_id=req.run_id,
    hidden_states_key=hidden_states_key,
    runs_dir=RUNS_DIR,
  )

  return {"response": response, "hidden_states_key": hidden_states_key}


@app.post("/compute")
def compute(req: ComputeRequest):
  run_file = RUNS_DIR / f"{req.run_id}.json"
  if not run_file.exists():
    raise HTTPException(status_code=404, detail="Run not found")

  run_data = json.loads(run_file.read_text())
  state_dir = RUNS_DIR / req.run_id

  # Collect classifications and hidden states for this (model, mode) pair
  per_category: dict[str, list] = {}

  for prompt in run_data["prompts"]:
    result = prompt.get("model_results", {}).get(req.model_id, {}).get(req.mode)
    if not result:
      continue

    key = result["hidden_states_key"]
    npy_path = state_dir / f"{key}.npy"
    if not npy_path.exists():
      continue

    category = prompt["category"]
    per_category.setdefault(category, []).append({
      "hidden_states_key": key,
      "refused": result["refused"],
      "hidden_state": np.load(str(npy_path)),
    })

  # Compute direction per category
  direction_results: dict[str, dict] = {}

  for category, entries in per_category.items():
    hidden_states_map = {e["hidden_states_key"]: e["hidden_state"] for e in entries}
    classifications = [
      {"hidden_states_key": e["hidden_states_key"], "refused": e["refused"]}
      for e in entries
    ]
    result = compute_run_directions(hidden_states_map, classifications, category=category)
    if result:
      direction_results[category] = result

  return direction_results


if __name__ == "__main__":
  uvicorn.run("backend.inference.service:app", host="0.0.0.0", port=8200, reload=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest backend/inference/tests/test_service.py -v`
Expected: PASS — all service tests pass.

- [ ] **Step 5: Smoke test the service**

Run from project root: `uv run python backend/inference/service.py`
Then in another terminal: `curl http://localhost:8200/status`
Expected: `{"loaded_model": null}`

- [ ] **Step 6: Commit**

```bash
git add backend/inference/service.py backend/inference/tests/test_service.py
git commit -m "feat: FastAPI inference service — load, generate, compute endpoints

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Task 14: End-to-End Smoke Test

Verify the full backend stack works together before handing off to frontend plan.

- [ ] **Step 1: Start all backend services**

Terminal 1: `npm run dev --workspace=backend`
Terminal 2: `uv run python backend/inference/service.py`

- [ ] **Step 2: Run seed**

```bash
npm run seed --workspace=backend
```
Expected: LLMs and prompts inserted.

- [ ] **Step 3: Verify API endpoints**

```bash
# List models
curl http://localhost:8199/api/models | python3 -m json.tool

# Get all prompts (first 3)
curl http://localhost:8199/api/prompts/all | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d)} prompts')"

# Check inference status
curl http://localhost:8199/api/inference/status
# Expected: {"loaded_model": null}

# Create a run
curl -X POST http://localhost:8199/api/runs \
  -H "Content-Type: application/json" \
  -d '{"models":["Qwen/Qwen3.6-27B"],"mode_selection":"non_thinking","prompt_scope":{"categories":["dangerous_activity"]}}' \
  | python3 -m json.tool
# Expected: run JSON with run_id, prompts array populated
```

- [ ] **Step 4: Verify run file was created on disk**

```bash
ls data/runs/
# Expected: run_<timestamp>_<uuid>.json  run_<timestamp>_<uuid>/
```

- [ ] **Step 5: Commit plan-A completion marker**

```bash
git add .
git commit -m "feat: backend + inference service complete — Plan A done

Co-Authored-By: Unnamed AI Assistant <unnamed_ai@assistant>"
```

---

## Self-Review Notes

**Spec coverage check:**
- ✅ Single `npm run dev` startup via concurrently
- ✅ Prompt schema: category_group added, refusal enum updated to [hard, redirect, disclaimer, none]
- ✅ Run file format: JSON manifest + .npy per-prompt hidden states in run subdirectory
- ✅ model_results[model_id][mode] structure
- ✅ direction_results[model_id][mode] structure
- ✅ sequence array with (model, mode) pairs, current_sequence_index tracking
- ✅ GET /api/prompts/all and POST /api/prompts/selected
- ✅ Full run CRUD
- ✅ Inference proxy routes
- ✅ Python: load, generate, compute endpoints
- ✅ Direction computation uses classification outcomes (refused bool), not prompt.type
- ✅ Compute is per (model_id, mode) independently
- ✅ Seed runner with category_group derivation

**Implementation note:** Hidden states are saved as individual `.npy` files in
`data/runs/<run_id>/` rather than a monolithic `.npz`. This avoids the full-archive
rewrite overhead on every prompt. The `hidden_states_key` in the JSON is the filename
stem. The spec's `.npz` reference is functionally equivalent — same numpy binary format.
