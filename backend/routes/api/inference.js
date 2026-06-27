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
router.post('/compute', proxyPost('/compute'))
router.post('/direction_pca', proxyPost('/direction_pca'))

router.post('/generate', async (req, res) => {
  const upstream = await fetch(`${INFERENCE_BASE}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req.body),
  })

  res.status(upstream.status)
  res.setHeader('Content-Type', 'application/x-ndjson')
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('X-Accel-Buffering', 'no')

  if (!upstream.body) { res.end(); return }

  req.on('close', () => { upstream.body.destroy?.() })

  for await (const chunk of upstream.body) {
    if (!res.write(chunk)) await new Promise(resolve => res.once('drain', resolve))
  }
  res.end()
})

export default router
