import { Router } from 'express'
import { LLM } from '../../models/llm.js'

const router = Router()

router.get('/', async (req, res) => {
  const models = await LLM.find({}).lean()
  res.json(models)
})

export default router
