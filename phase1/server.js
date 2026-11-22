const express = require('express');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 4173;

// Dev mode: phase1/ sits inside the git repo, so src/ is right next to it.
// Packaged mode: electron-builder's extraResources copies src/models/
// requirements.txt to <resourcesPath>/repo -- __dirname points into the
// asar archive there, which never contains src/ directly, so the dev-mode
// check below correctly falls through to the packaged path.
function resolveRepoRoot() {
  const devRoot = path.join(__dirname, '..');
  if (fs.existsSync(path.join(devRoot, 'src'))) return devRoot;
  if (typeof process.resourcesPath === 'string') {
    const packagedRoot = path.join(process.resourcesPath, 'repo');
    if (fs.existsSync(path.join(packagedRoot, 'src'))) return packagedRoot;
  }
  return devRoot;
}

const REPO_ROOT = resolveRepoRoot();
// No bundled .venv, on purpose -- a virtualenv isn't relocatable (bit us
// once already in this project) and a Linux-built one is useless on
// Windows anyway. The user creates their own venv in REPO_ROOT after
// install; the /api/launch error message below tells them exactly how.
const PYTHON = process.platform === 'win32'
  ? path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe')
  : path.join(REPO_ROOT, '.venv', 'bin', 'python3');
const TOOLS = JSON.parse(fs.readFileSync(path.join(__dirname, 'tools.json'), 'utf8'));
const MAX_LOG_LINES = 500;

const running = new Map(); // tool id -> child process
const logsById = new Map(); // tool id -> string[], survives past process exit (reset only on next launch)
const logSubscribers = new Map(); // tool id -> Set<res>

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

function stopTool(id) {
  const child = running.get(id);
  if (!child) return false;
  try {
    process.kill(-child.pid, 'SIGTERM'); // detached -> own process group, kill the whole group
  } catch (e) {
    try {
      child.kill('SIGTERM');
    } catch (_) {
      /* already gone */
    }
  }
  running.delete(id);
  return true;
}

function broadcastLog(id, line) {
  const subs = logSubscribers.get(id);
  if (!subs) return;
  for (const res of subs) res.write(`data: ${JSON.stringify(line)}\n\n`);
}

app.get('/api/tools', (req, res) => {
  res.json(TOOLS.map((t) => ({ ...t, running: running.has(t.id) })));
});

app.post('/api/launch/:id', (req, res) => {
  const tool = TOOLS.find((t) => t.id === req.params.id);
  if (!tool) return res.status(404).json({ error: 'unknown tool' });
  if (running.has(tool.id)) return res.json({ ok: true, alreadyRunning: true, mode: tool.mode, port: tool.port });

  if (!fs.existsSync(PYTHON)) {
    const setupCmd = process.platform === 'win32'
      ? `cd "${REPO_ROOT}" && python -m venv .venv && .venv\\Scripts\\pip install -r requirements.txt`
      : `cd "${REPO_ROOT}" && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`;
    return res.status(500).json({
      error: `venv python not found at ${PYTHON}. One-time setup, run: ${setupCmd}`,
    });
  }
  const scriptPath = path.join(REPO_ROOT, 'src', tool.script);
  if (!fs.existsSync(scriptPath)) {
    return res.status(404).json({ error: `script not found: ${scriptPath}` });
  }

  // The webcam is one exclusive device -- only one tool can hold it at a time.
  for (const otherId of Array.from(running.keys())) stopTool(otherId);

  const scriptArgs = [scriptPath];
  if (tool.mode === 'stream') scriptArgs.push('--serve', String(tool.port));

  const child = spawn(PYTHON, scriptArgs, {
    cwd: REPO_ROOT,
    env: process.env,
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  running.set(tool.id, child);
  logsById.set(tool.id, []); // fresh log buffer per launch, independent of process lifetime

  const onData = (d) => {
    const text = d.toString();
    process.stdout.write(`[${tool.id}] ${text}`);
    const logs = logsById.get(tool.id);
    for (const line of text.split('\n')) {
      if (!line) continue;
      logs.push(line);
      if (logs.length > MAX_LOG_LINES) logs.shift();
      broadcastLog(tool.id, line);
    }
  };
  child.stdout.on('data', onData);
  child.stderr.on('data', onData);
  child.on('exit', (code) => {
    running.delete(tool.id);
    broadcastLog(tool.id, `[exited with code ${code}]`);
    console.log(`[${tool.id}] exited with code ${code}`);
  });
  child.unref();

  res.json({ ok: true, pid: child.pid, mode: tool.mode, port: tool.port });
});

app.post('/api/param/:id', (req, res) => {
  // Same relay pattern as /api/key/:id -- keeps the frontend only ever talking
  // to :4173. Used for live-adjustable controls, e.g. the sensitivity slider.
  const tool = TOOLS.find((t) => t.id === req.params.id);
  if (!tool || tool.mode !== 'stream') return res.status(404).json({ error: 'not a stream tool' });
  const { name, value } = req.body || {};
  if (!name || value === undefined) return res.status(400).json({ error: 'name and value required' });
  fetch(`http://127.0.0.1:${tool.port}/param?name=${encodeURIComponent(name)}&value=${encodeURIComponent(value)}`)
    .then(() => res.json({ ok: true }))
    .catch((err) => res.status(502).json({ error: err.message }));
});

app.get('/api/gesture/:id', (req, res) => {
  // Reads models/gestures.json directly -- it's just a JSON file the python side
  // writes to, no need to round-trip through the running process for a status check.
  const gesturesPath = path.join(REPO_ROOT, 'models', 'gestures.json');
  let recorded = {};
  try {
    recorded = JSON.parse(fs.readFileSync(gesturesPath, 'utf8'));
  } catch (e) {
    /* not recorded yet, or file doesn't exist -- empty is fine */
  }
  res.json({ actions: Object.keys(recorded) });
});

app.post('/api/gesture/:id/record', (req, res) => {
  const tool = TOOLS.find((t) => t.id === req.params.id);
  if (!tool || tool.mode !== 'stream') return res.status(404).json({ error: 'not a stream tool' });
  const { action } = req.body || {};
  if (!action) return res.status(400).json({ error: 'action required' });
  fetch(`http://127.0.0.1:${tool.port}/record?action=${encodeURIComponent(action)}`)
    .then(() => res.json({ ok: true }))
    .catch((err) => res.status(502).json({ error: err.message }));
});

app.post('/api/gesture/:id/clear', (req, res) => {
  const tool = TOOLS.find((t) => t.id === req.params.id);
  if (!tool || tool.mode !== 'stream') return res.status(404).json({ error: 'not a stream tool' });
  const { action } = req.body || {};
  if (!action) return res.status(400).json({ error: 'action required' });
  fetch(`http://127.0.0.1:${tool.port}/record?action=${encodeURIComponent(action)}&clear=1`)
    .then(() => res.json({ ok: true }))
    .catch((err) => res.status(502).json({ error: err.message }));
});

app.post('/api/stop/:id', (req, res) => {
  const wasRunning = stopTool(req.params.id);
  res.json({ ok: true, wasRunning });
});

app.post('/api/key/:id', (req, res) => {
  // Convenience proxy so the frontend only ever talks to :4173 -- forwards to the
  // tool's own StreamServer /key endpoint. Stream-mode tools are also reachable
  // directly on their own port, this just avoids a second fetch target in the JS.
  const tool = TOOLS.find((t) => t.id === req.params.id);
  if (!tool || tool.mode !== 'stream') return res.status(404).json({ error: 'not a stream tool' });
  const code = req.body && req.body.code;
  fetch(`http://127.0.0.1:${tool.port}/key?code=${encodeURIComponent(code)}`)
    .then(() => res.json({ ok: true }))
    .catch((err) => res.status(502).json({ error: err.message }));
});

app.get('/api/logs/:id', (req, res) => {
  const id = req.params.id;
  res.set({
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  });
  res.flushHeaders();

  const logs = logsById.get(id) || [];
  for (const line of logs) res.write(`data: ${JSON.stringify(line)}\n\n`);

  if (!logSubscribers.has(id)) logSubscribers.set(id, new Set());
  logSubscribers.get(id).add(res);

  req.on('close', () => {
    const subs = logSubscribers.get(id);
    if (subs) subs.delete(res);
  });
});

app.listen(PORT, () => {
  console.log(`Phase 1 launcher running at http://localhost:${PORT}`);
  console.log(`Using python: ${PYTHON}`);
});
