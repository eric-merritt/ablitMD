import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest'
import mongoose from 'mongoose'
import { MongoMemoryServer } from 'mongodb-memory-server'
import request from 'supertest'
import app from '../app.js'
import { LLM } from '../models/llm.js'
import { Prompt } from '../models/prompt.js'

let mongod

beforeAll(async () => {
  // Wait briefly for app.js connection attempt to timeout, then disconnect and reconnect to MongoMemoryServer
  await new Promise(resolve => setTimeout(resolve, 100))

  // Force disconnect from any pending connection
  try {
    await mongoose.connection.close()
  } catch (e) {
    // Connection may not exist yet
  }

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
      name: 'Test Model',
      modelId: 'test/model',
      vendor: 'Test',
      apiProvider: 'self_hosted',
      apiModelId: 'test/model',
      architecture: {
        numLayers: 10,
        hiddenSize: 512,
        architectureType: 'transformer',
      },
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
      {
        category: 'dangerous_activity',
        category_group: 'violence_physical_harm',
        type: 'harmful',
        text: 'p1',
        triggers: [],
      },
      {
        category: 'hate_speech',
        category_group: 'hate_discrimination',
        type: 'harmless',
        text: 'p2',
        triggers: [],
      },
    ])
    const res = await request(app).get('/api/prompts/all')
    expect(res.status).toBe(200)
    expect(res.body).toHaveLength(2)
  })
})

describe('POST /api/prompts/selected', () => {
  beforeEach(async () => {
    await Prompt.create([
      {
        category: 'dangerous_activity',
        category_group: 'violence_physical_harm',
        type: 'harmful',
        text: 'p1',
        triggers: [],
      },
      {
        category: 'hate_speech',
        category_group: 'hate_discrimination',
        type: 'harmful',
        text: 'p2',
        triggers: [],
      },
      {
        category: 'medical_advice',
        category_group: 'professional_advice',
        type: 'harmless',
        text: 'p3',
        triggers: [],
      },
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

describe('POST /api/runs', () => {
  it('creates a run and returns run_id', async () => {
    await Prompt.create([
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
