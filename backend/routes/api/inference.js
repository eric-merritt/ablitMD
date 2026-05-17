import { Router } from 'express'

const router = Router()
const INFERENCE_BASE = process.env.INFERENCE_URL || 'http://localhost:8238'

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
