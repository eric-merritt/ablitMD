// Gated run-data server. Run this on the always-on box behind eric-merritt.com so
// vast.ai instances fetch run data on demand instead of rsyncing it.
//
//   DATA_KEY=<secret> DATA_PORT=8240 RUNS_DIR=/path/to/data/runs node scripts/data_server.mjs
//
// Front it at eric-merritt.com/ablitMD/data (reverse-proxy /ablitMD/data -> this port).
// Endpoints:
//   GET /health                     -> "ok"
//   GET /run/:runId.tar?key=SECRET  -> tarball of <runId>.json, <runId>.*, and <runId>/ (npy)

import http from 'node:http'
import { spawn } from 'node:child_process'
import { readdir } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const RUNS_DIR = process.env.RUNS_DIR || join(__dirname, '..', 'data', 'runs')
const DATA_KEY = process.env.DATA_KEY
const PORT = Number(process.env.DATA_PORT || 8240)

if (!DATA_KEY) throw new Error('DATA_KEY must be set')

// run ids are `run_<iso>_<hex>` or simple slugs — never contains a slash or dot-dot.
const SAFE_RUN_ID = /^[A-Za-z0-9._-]+$/

const tarEntriesFor = async (runId) => {
  const names = await readdir(RUNS_DIR)
  // the run json, every sidecar (<runId>.directions.json, .recipe.*, .verify.json …),
  // and the hidden-state directory <runId>/.
  return names.filter(name => name === runId || name === `${runId}.json` || name.startsWith(`${runId}.`))
}

const streamRunTar = (runId, entries, res) => {
  res.writeHead(200, { 'Content-Type': 'application/x-tar' })
  const tar = spawn('tar', ['-cf', '-', '-C', RUNS_DIR, ...entries])
  tar.stdout.pipe(res)
  tar.stderr.on('data', chunk => console.error(`[data-server] tar: ${chunk}`))
  tar.on('error', () => { if (!res.headersSent) res.writeHead(500); res.end() })
  res.on('close', () => tar.kill())
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`)
  if (url.pathname === '/health') { res.writeHead(200); res.end('ok'); return }

  const match = url.pathname.match(/^\/run\/([^/]+)\.tar$/)
  if (!match) { res.writeHead(404); res.end('not found'); return }
  if (url.searchParams.get('key') !== DATA_KEY) { res.writeHead(403); res.end('forbidden'); return }

  const runId = decodeURIComponent(match[1])
  if (!SAFE_RUN_ID.test(runId)) { res.writeHead(400); res.end('bad run id'); return }

  const entries = await tarEntriesFor(runId)
  if (entries.length === 0) { res.writeHead(404); res.end('run not found'); return }
  console.log(`[data-server] serving ${runId} (${entries.length} entries)`)
  streamRunTar(runId, entries, res)
})

server.listen(PORT, () => console.log(`[data-server] serving ${RUNS_DIR} on :${PORT}`))
