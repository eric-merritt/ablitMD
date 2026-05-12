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
