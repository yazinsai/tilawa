import { createServer } from 'node:http';
import { createHash, randomBytes, randomInt } from 'node:crypto';
import { appendFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync, readFileSync } from 'node:fs';
import { extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { generatePrompt, KIND_WEIGHTS, loadQuran } from './prompts.mjs';

const here = fileURLToPath(new URL('.', import.meta.url));
const PORT = Number(process.env.PORT ?? 8787);
const DATA_DIR = resolve(process.env.DATA_DIR ?? join(here, 'data'));
const QURAN_JSON = resolve(process.env.QURAN_JSON ?? join(here, '../lab/data/quran.json'));
const ADMIN_TOKEN = process.env.ADMIN_TOKEN ?? '';
const IP_SALT = process.env.IP_SALT ?? randomBytes(16).toString('hex');
const MAX_BYTES = 8 * 1024 * 1024;
const MIN_BYTES = 4 * 1024;
const MAX_PER_HOUR = 60;
export const CONSENT_VERSION = '2026-09-30';

const quran = loadQuran(JSON.parse(readFileSync(QURAN_JSON, 'utf8')));
const SUBMISSIONS = join(DATA_DIR, 'submissions.jsonl');
await mkdir(join(DATA_DIR, 'audio'), { recursive: true });

const used = new Set();
if (existsSync(SUBMISSIONS)) {
  for (const line of (await readFile(SUBMISSIONS, 'utf8')).split('\n')) {
    if (line) used.add(`${JSON.parse(line).speaker}:${JSON.parse(line).seed}`);
  }
}

const hits = new Map();
function rateLimited(ipHash) {
  const now = Date.now();
  const recent = (hits.get(ipHash) ?? []).filter(t => now - t < 3600_000);
  recent.push(now);
  hits.set(ipHash, recent);
  return recent.length > MAX_PER_HOUR;
}

const AUDIO_EXT = { 'audio/webm': 'webm', 'audio/ogg': 'ogg', 'audio/mp4': 'm4a', 'audio/mpeg': 'mp3', 'audio/wav': 'wav', 'audio/x-wav': 'wav' };
const STATIC_TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.ttf': 'font/ttf' };
const ID = /^[A-Za-z0-9_-]{8,40}$/;
const ENUMS = {
  device: new Set(['phone', 'laptop', 'desktop', 'tablet', 'headset', 'other']),
  gender: new Set(['male', 'female', '']),
  level: new Set(['beginner', 'intermediate', 'hafiz', '']),
};

const json = (res, status, body) => {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(body));
};

async function readBody(req) {
  const chunks = [];
  let size = 0;
  for await (const c of req) {
    size += c.length;
    if (size > MAX_BYTES) throw Object.assign(new Error('too large'), { status: 413 });
    chunks.push(c);
  }
  return Buffer.concat(chunks);
}

async function submit(req, res, url) {
  const ip = req.headers['x-forwarded-for']?.split(',')[0].trim() ?? req.socket.remoteAddress ?? '';
  const ipHash = createHash('sha256').update(IP_SALT + ip).digest('hex').slice(0, 16);
  if (rateLimited(ipHash)) return json(res, 429, { error: 'Too many submissions, try again later.' });

  const q = Object.fromEntries(url.searchParams);
  const seed = Number(q.seed);
  const duration = Number(q.duration);
  const mime = (req.headers['content-type'] ?? '').split(';')[0].trim();
  const problems = [
    !Number.isInteger(seed) || seed < 0 || seed > 2 ** 32 ? 'seed' : '',
    !ID.test(q.speaker ?? '') ? 'speaker' : '',
    q.consent !== CONSENT_VERSION ? 'consent' : '',
    q.followed !== '1' ? 'followed' : '',
    !AUDIO_EXT[mime] ? 'content-type' : '',
    !(duration >= 1 && duration <= 90) ? 'duration' : '',
    ...Object.entries(ENUMS).map(([k, set]) => (set.has(q[k] ?? '') ? '' : k)),
  ].filter(Boolean);
  if (problems.length) return json(res, 400, { error: `Invalid: ${problems.join(', ')}` });
  if (used.has(`${q.speaker}:${seed}`)) return json(res, 409, { error: 'Already submitted.' });

  const audio = await readBody(req);
  if (audio.length < MIN_BYTES) return json(res, 400, { error: 'Recording is empty or too short.' });

  const prompt = generatePrompt(quran, seed);
  const id = `${Date.now().toString(36)}_${randomBytes(4).toString('hex')}`;
  const file = `audio/${id}.${AUDIO_EXT[mime]}`;
  await writeFile(join(DATA_DIR, file), audio);
  used.add(`${q.speaker}:${seed}`);

  const record = {
    id,
    file,
    ts: new Date().toISOString(),
    seed,
    prompt_version: prompt.v,
    speaker: q.speaker,
    device: q.device,
    gender: q.gender || null,
    level: q.level || null,
    ref: prompt.label.ref,
    kind: prompt.label.kind,
    mistakes: prompt.label.mistakes,
    extra_mistake: q.extra === '1',
    duration_s: duration,
    mime,
    bytes: audio.length,
    user_agent: String(req.headers['user-agent'] ?? '').slice(0, 200),
    ip_hash: ipHash,
    consent: CONSENT_VERSION,
  };
  await appendFile(SUBMISSIONS, JSON.stringify(record) + '\n');
  json(res, 200, { ok: true, id });
}

async function serveStatic(res, pathname) {
  const rel = pathname === '/' ? 'index.html' : pathname.slice(1);
  const file = resolve(here, 'public', rel);
  if (!file.startsWith(resolve(here, 'public')) || !existsSync(file)) {
    res.writeHead(404);
    return res.end('Not found');
  }
  res.writeHead(200, { 'content-type': STATIC_TYPES[extname(file)] ?? 'application/octet-stream' });
  res.end(await readFile(file));
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, 'http://x');
  try {
    if (req.method === 'GET' && url.pathname === '/api/prompt') {
      const kind = url.searchParams.get('kind');
      const seed = randomInt(0, 2 ** 32 - 1);
      const p = generatePrompt(quran, seed, kind && kind in KIND_WEIGHTS ? kind : undefined);
      return json(res, 200, { seed: p.seed, kind: p.label.kind, display: p.display, instruction: p.instruction });
    }
    if (req.method === 'POST' && url.pathname === '/api/submit') return await submit(req, res, url);
    const admin = ADMIN_TOKEN && req.headers.authorization === `Bearer ${ADMIN_TOKEN}`;
    if (req.method === 'GET' && url.pathname.startsWith('/api/audio/')) {
      const name = url.pathname.slice('/api/audio/'.length);
      if (!admin) return json(res, 401, { error: 'unauthorized' });
      if (!/^[a-z0-9_]+\.(webm|ogg|m4a|mp3|wav)$/.test(name) || !existsSync(join(DATA_DIR, 'audio', name))) return json(res, 404, { error: 'not found' });
      res.writeHead(200, { 'content-type': 'application/octet-stream' });
      return res.end(await readFile(join(DATA_DIR, 'audio', name)));
    }
    if (req.method === 'GET' && url.pathname === '/api/export') {
      if (!admin) return json(res, 401, { error: 'unauthorized' });
      res.writeHead(200, { 'content-type': 'application/x-ndjson' });
      return res.end(existsSync(SUBMISSIONS) ? await readFile(SUBMISSIONS) : '');
    }
    if (req.method === 'GET' && url.pathname === '/api/config') return json(res, 200, { consent: CONSENT_VERSION });
    if (req.method === 'GET') return await serveStatic(res, url.pathname);
    json(res, 405, { error: 'method not allowed' });
  } catch (err) {
    json(res, err.status ?? 500, { error: err.status ? err.message : 'server error' });
    if (!err.status) console.error(err);
  }
});

server.listen(PORT, () => console.log(`collect: http://localhost:${PORT}  data=${DATA_DIR}`));
