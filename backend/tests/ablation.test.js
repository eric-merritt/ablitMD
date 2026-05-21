import { describe, it, expect, afterEach } from 'vitest'
import { fileURLToPath } from 'node:url'
import { writeFile, rm, mkdir } from 'node:fs/promises'
import path from 'node:path'
import request from 'supertest'
import app from '../app.js'

const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
const RUNS_DIR = path.join(PROJECT_ROOT, 'data', 'runs')
const TEST_RUN = 'run_ablation_test'

afterEach(async () => {
  for (const suffix of ['.json', '.divergence.json', '.recipe.json']) {
    await rm(path.join(RUNS_DIR, `${TEST_RUN}${suffix}`), { force: true })
  }
})

describe('GET /api/ablation/:runId/divergence', () => {
  it('returns 404 for an unknown run', async () => {
    const res = await request(app).get('/api/ablation/does_not_exist/divergence')
    expect(res.status).toBe(404)
  })

  it('returns a pre-computed divergence file without spawning', async () => {
    await mkdir(RUNS_DIR, { recursive: true })
    await writeFile(path.join(RUNS_DIR, `${TEST_RUN}.json`), '{}')
    const payload = { computed_at: 'x', 'Test/Model': { non_thinking: { hard: {} } } }
    await writeFile(path.join(RUNS_DIR, `${TEST_RUN}.divergence.json`), JSON.stringify(payload))
    const res = await request(app).get(`/api/ablation/${TEST_RUN}/divergence`)
    expect(res.status).toBe(200)
    expect(res.body).toEqual(payload)
  })
})
