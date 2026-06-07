// Gated run-data server. Run this on the always-on box behind eric-merritt.com so
// vast.ai instances fetch run data on demand instead of rsyncing it.
//
//   DATA_KEY=<secret> DATA_PORT=8240 RUNS_DIR=/path/to/data/runs node scripts/data_server.mjs
//
// Front it at eric-merritt.com/ablitMD/data (reverse-proxy /ablitMD/data -> this port).
// Endpoints:
//   GET /health                     -> "ok"
//   GET /run/:runId.tar?key=SECRET  -> tarball of <runId>.json, <runId>.*, and <runId>/ (npy)

import http from "node:http";
import { spawn } from "node:child_process";
import { readdir, readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { homedir } from "node:os";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const RUNS_DIR = process.env.RUNS_DIR || join(__dirname, "..", "data", "runs");
const DATA_KEY = process.env.DATA_KEY;
const PORT = Number(process.env.DATA_PORT || 8240);
const SSH_DIR = join(homedir(), ".ssh");
const SSH_CONFIG = join(SSH_DIR, "config");
const DEFAULT_ALIAS = process.env.VAST_SSH_ALIAS || "vastai";

if (!DATA_KEY) throw new Error("DATA_KEY must be set");

// run ids are `run_<iso>_<hex>` or simple slugs — never contains a slash or dot-dot.
const SAFE_RUN_ID = /^[A-Za-z0-9._-]+$/;
const SAFE_NAME = /^[A-Za-z0-9._-]+$/; // alias, user
const SAFE_HOST = /^[A-Za-z0-9.:-]+$/; // IPv4/IPv6/hostname, no spaces/newlines
const escapeRe = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const sshBlock = ({ alias, ip, port, user }) =>
  `# >>> ${alias} (auto-managed by ablitmd data-server) >>>\n` +
  `Host ${alias}\n` +
  `    HostName ${ip}\n` +
  `    Port ${port}\n` +
  `    User ${user}\n` +
  `    StrictHostKeyChecking no\n` + // vast instances are ephemeral — host key churns
  `    UserKnownHostsFile /dev/null\n` +
  `# <<< ${alias} <<<`;

// Replace (or insert) the alias's managed block in ~/.ssh/config, leaving everything else intact.
const updateSshConfig = async ({ alias, ip, port, user }) => {
  await mkdir(SSH_DIR, { recursive: true, mode: 0o700 });
  let existing = "";
  try {
    existing = await readFile(SSH_CONFIG, "utf8");
  } catch (err) {
    if (err.code !== "ENOENT") throw err;
  }
  const block = new RegExp(
    `\\n*# >>> ${escapeRe(alias)} \\(auto-managed[^\\n]*\\n[\\s\\S]*?# <<< ${escapeRe(alias)} <<<\\n*`,
    "g",
  );
  const cleaned = existing.replace(block, "\n").trimEnd();
  const next =
    (cleaned ? `${cleaned}\n\n` : "") +
    sshBlock({ alias, ip, port, user }) +
    "\n";
  await writeFile(SSH_CONFIG, next, { mode: 0o600 });
};

const ATLAS_PUBLIC = process.env.ATLAS_API_PUBLIC_KEY;
const ATLAS_PRIVATE = process.env.ATLAS_API_PRIVATE_KEY;
const ATLAS_PROJECT = process.env.ATLAS_PROJECT_ID;

// Whitelist an IP in the Atlas project's access list via the Admin API, so the instance
// can reach Mongo without you hand-adding it. Digest auth — shelled to curl (Node fetch
// can't do HTTP Digest). 201 = added, 409 = already present; both are success.
const atlasWhitelist = (ip) =>
  new Promise((resolve) => {
    if (!ATLAS_PUBLIC || !ATLAS_PRIVATE || !ATLAS_PROJECT) {
      resolve({ ok: false, status: "atlas-not-configured" });
      return;
    }
    const body = JSON.stringify([
      {
        ipAddress: ip,
        comment: `ablitmd-vast ${new Date().toISOString().slice(0, 10)}`,
      },
    ]);
    const curl = spawn("curl", [
      "-s",
      "-o",
      "/dev/null",
      "-w",
      "%{http_code}",
      "--digest",
      "-u",
      `${ATLAS_PUBLIC}:${ATLAS_PRIVATE}`,
      "-X",
      "POST",
      `https://cloud.mongodb.com/api/atlas/v1.0/groups/${ATLAS_PROJECT}/accessList`,
      "-H",
      "Content-Type: application/json",
      "-d",
      body,
    ]);
    let code = "";
    curl.stdout.on("data", (chunk) => {
      code += chunk;
    });
    curl.on("error", () => resolve({ ok: false, status: "curl-error" }));
    curl.on("close", () =>
      resolve({ ok: code.startsWith("2") || code === "409", status: code }),
    );
  });

const tarEntriesFor = async (runId) => {
  const names = await readdir(RUNS_DIR);
  // the run json, every sidecar (<runId>.directions.json, .recipe.*, .verify.json …),
  // and the hidden-state directory <runId>/.
  return names.filter(
    (name) =>
      name === runId ||
      name === `${runId}.json` ||
      name.startsWith(`${runId}.`),
  );
};

const streamRunTar = (runId, entries, res) => {
  res.writeHead(200, { "Content-Type": "application/x-tar" });
  const tar = spawn("tar", ["-cf", "-", "-C", RUNS_DIR, ...entries]);
  tar.stdout.pipe(res);
  tar.stderr.on("data", (chunk) =>
    console.error(`[data-server] tar: ${chunk}`),
  );
  tar.on("error", () => {
    if (!res.headersSent) res.writeHead(500);
    res.end();
  });
  res.on("close", () => tar.kill());
};

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  if (url.pathname === "/health") {
    res.writeHead(200);
    res.end("ok");
    return;
  }

  // Instance self-registration: update ~/.ssh/config + whitelist the IP in Atlas.
  if (req.method === "POST" && url.pathname === "/instance") {
    if (url.searchParams.get("key") !== DATA_KEY) {
      res.writeHead(403);
      res.end("forbidden");
      return;
    }
    const ip = url.searchParams.get("ip") || "";
    const port = url.searchParams.get("port") || "";
    const user = url.searchParams.get("user") || "root";
    const alias = url.searchParams.get("alias") || DEFAULT_ALIAS;
    if (!SAFE_HOST.test(ip) || !/^\d{1,5}$/.test(port) || !SAFE_NAME.test(user) || !SAFE_NAME.test(alias)) {
      res.writeHead(400);
      res.end("bad ip/port/user/alias");
      return;
    }
    await updateSshConfig({ alias, ip, port, user });
    const atlas = await atlasWhitelist(ip);
    console.log(`[data-server] instance ${alias} -> ${ip}:${port} (ssh-config updated, atlas ${atlas.status})`);
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ ok: true, alias, ip, port, atlas }));
    return;
  }

  const match = url.pathname.match(/^\/run\/([^/]+)\.tar$/);
  if (!match) {
    res.writeHead(404);
    res.end("not found");
    return;
  }
  if (url.searchParams.get("key") !== DATA_KEY) {
    res.writeHead(403);
    res.end("forbidden");
    return;
  }

  const runId = decodeURIComponent(match[1]);
  if (!SAFE_RUN_ID.test(runId)) {
    res.writeHead(400);
    res.end("bad run id");
    return;
  }

  const entries = await tarEntriesFor(runId);
  if (entries.length === 0) {
    res.writeHead(404);
    res.end("run not found");
    return;
  }
  console.log(`[data-server] serving ${runId} (${entries.length} entries)`);
  streamRunTar(runId, entries, res);
});

// Bind loopback only — nginx fronts it; the port must never be publicly reachable.
const HOST = process.env.DATA_HOST || "127.0.0.1";
server.listen(PORT, HOST, () =>
  console.log(`[data-server] serving ${RUNS_DIR} on ${HOST}:${PORT}`),
);
