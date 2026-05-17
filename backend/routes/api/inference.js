import { Router } from 'express'

const router = Router()
const INFERENCE_BASE = process.env.INFERENCE_URL || 'http://localhost:8238'

const safeJson = async (response) => {
  const text = await response.text()
  try { return JSON.parse(text) } catch { return { detail: text } }
}

const proxyPost = (path) => async (req, res) => {
  const response = await fetch(`${INFERENCE_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req.body),
  })
  res.status(response.status).json(await safeJson(response))
}

router.get('/status', async (req, res) => {
  const response = await fetch(`${INFERENCE_BASE}/status`)
  res.json(await safeJson(response))
})

router.post('/load', proxyPost('/load'))
router.post('/generate', proxyPost('/generate'))
router.post('/compute', proxyPost('/compute'))

export default router
