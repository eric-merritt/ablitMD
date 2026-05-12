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
