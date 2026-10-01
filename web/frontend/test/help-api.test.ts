import { afterAll, test } from 'vitest';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { generatePrompt, KIND_WEIGHTS, loadQuran } from '../server/help/prompts.mjs';
import { generatePrompt as generateV4 } from '../server/help/prompts-v4.mjs';
import { signPrompt } from '../server/help/prompt-token.mjs';

const data = await mkdtemp(join(tmpdir(), 'tilawa-help-api-'));
const previous = { data: process.env.HELP_DATA_DIR, quran: process.env.QURAN_JSON, secret: process.env.HELP_PROMPT_SECRET };
process.env.HELP_DATA_DIR = data;
process.env.HELP_PROMPT_SECRET = 'help-api-test-only-secret';
process.env.QURAN_JSON = fileURLToPath(new URL('../public/quran.json', import.meta.url));
const { helpApi } = await import('../server/help/index');
const quran = loadQuran(JSON.parse(await readFile(process.env.QURAN_JSON, 'utf8')));
const records = async () => (await readFile(join(data, 'submissions.jsonl'), 'utf8')).trim().split('\n').map(line => JSON.parse(line));

function submit(prompt, speaker, overrides = {}) {
  const params = new URLSearchParams({
    seed: String(prompt.seed), speaker, consent: '2026-09-30', followed: '1',
    duration: '4', device: 'phone', gender: '', level: '',
    ...(prompt.v ? { v: String(prompt.v), token: prompt.token } : {}), ...overrides,
  });
  return helpApi.request(`/submit?${params}`, { method: 'POST', headers: { 'content-type': 'audio/webm' }, body: new Uint8Array(5000) });
}

afterAll(async () => {
  await rm(data, { recursive: true, force: true });
  for (const [key, value] of [['HELP_DATA_DIR', previous.data], ['QURAN_JSON', previous.quran], ['HELP_PROMPT_SECRET', previous.secret]]) {
    if (value === undefined) delete process.env[key]; else process.env[key] = value;
  }
});

const exists = async () => readFile(join(data, 'submissions.jsonl'), 'utf8').then(() => true, () => false);
const count = async () => ((await exists()) ? (await records()).length : 0);

test('only correct-recitation v4 prompts are issued, whatever kind is requested', async () => {
  for (const kind of [undefined, ...Object.keys(KIND_WEIGHTS)]) {
    const prompt = await (await helpApi.request(kind ? `/prompt?kind=${kind}` : '/prompt')).json();
    assert.equal(prompt.v, 4);
    assert.equal(prompt.kind, 'none');
    assert.ok(prompt.display.every(a => a.words.every(w => !w.mark)), 'no highlighted alteration');
    assert.ok(prompt.display.length >= 1 && prompt.display.length <= 3);
  }
});

test('a v4 submission stores the issued label with no mistakes', async () => {
  const prompt = await (await helpApi.request('/prompt')).json();
  assert.equal((await submit(prompt, 'test_v4_ok')).status, 200);
  const record = (await records()).at(-1);
  const expected = generateV4(quran, prompt.seed);
  assert.equal(record.prompt_version, 4);
  assert.equal(record.kind, 'none');
  assert.equal(record.elicitation, 'normal_recitation');
  assert.equal(record.ref, expected.label.ref);
  assert.deepEqual(record.ayahs, expected.label.ayahs);
  assert.deepEqual(record.mistakes, []);
  assert.equal((await submit(prompt, 'test_v4_ok')).status, 409);
});

test('tampered or missing v4 tokens are rejected before audio is stored', async () => {
  const prompt = await (await helpApi.request('/prompt')).json();
  const before = await count();
  for (const overrides of [{ token: '' }, { token: `${prompt.token}x` }, { v: '999' }, { seed: String(prompt.seed + 1) }]) {
    assert.equal((await submit(prompt, 'tamper_test', overrides)).status, 400);
  }
  assert.equal(await count(), before);
});

test('retired mistake prompts (legacy v1, signed v2/v3) are never stored', async () => {
  const before = await count();
  assert.equal((await submit({ seed: 987654 }, 'legacy_test')).status, 400);
  for (const [v, kind] of [[2, 'vowel'], [3, 'skip_word'], [3, 'none']]) {
    const p = generatePrompt(quran, 34567, kind, v);
    const token = signPrompt(p, process.env.HELP_PROMPT_SECRET);
    assert.equal((await submit({ ...p, token }, `retired_v${v}_${kind}`)).status, 400);
  }
  const forged = { ...generateV4(quran, 4242), label: { ...generateV4(quran, 4242).label, kind: 'vowel' } };
  assert.equal((await submit({ ...forged, token: signPrompt(forged, process.env.HELP_PROMPT_SECRET) }, 'forged_kind')).status, 400);
  assert.equal(await count(), before);
});

test('v4 passages are real consecutive ayahs within one surah', () => {
  const shapes = { single: 0, run: 0 };
  for (let seed = 1; seed < 2000; seed++) {
    const p = generateV4(quran, seed);
    shapes[p.label.passage]++;
    const refs = p.label.ayahs.map(r => r.split(':').map(Number));
    assert.ok(refs.every(([s]) => s === refs[0][0]), 'same surah');
    refs.forEach(([, a], i) => i && assert.equal(a, refs[i - 1][1] + 1, 'consecutive'));
    assert.deepEqual(p.display.map(d => d.ref), p.label.ayahs);
    p.display.forEach(d => assert.deepEqual(d.words.map(w => w.text), quran.byRef.get(d.ref).words, 'text unaltered'));
  }
  assert.ok(Math.abs(shapes.single / 1999 - 0.6) < 0.05, JSON.stringify(shapes));
});
