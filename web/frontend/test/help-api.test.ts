import { afterAll, test } from 'vitest';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { generatePrompt, KIND_WEIGHTS, loadQuran } from '../server/help/prompts.mjs';

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

test('every requested kind stores exactly the issued v3 label', async () => {
  for (const kind of Object.keys(KIND_WEIGHTS)) {
    const res = await helpApi.request(`/prompt?kind=${kind}`);
    assert.equal(res.status, 200);
    const prompt = await res.json();
    assert.equal(prompt.kind, kind);
    assert.equal(prompt.v, 3);
    assert.ok(prompt.token);
    assert.equal((await submit(prompt, `test_${kind}`)).status, 200);
    const record = (await records()).at(-1);
    const expected = generatePrompt(quran, prompt.seed, kind);
    assert.equal(record.kind, kind);
    assert.equal(record.prompt_version, 3);
    assert.equal(record.scenario, expected.label.scenario ?? null);
    assert.equal(record.mechanism, expected.label.mechanism ?? null);
    assert.equal(record.passage_family, expected.label.passage_family);
    assert.equal(record.fluency_control, kind === 'repeat');
    assert.deepEqual(record.mistakes, expected.label.mistakes);
    assert.equal((await submit(prompt, `test_${kind}`)).status, 409);
  }
});

test('tampered or missing v3 tokens are rejected before audio is stored', async () => {
  const prompt = await (await helpApi.request('/prompt?kind=vowel')).json();
  const before = (await records()).length;
  for (const overrides of [{ token: '' }, { token: `${prompt.token}x` }, { v: '999' }, { seed: String(prompt.seed + 1) }]) {
    assert.equal((await submit(prompt, 'tamper_test', overrides)).status, 400);
  }
  assert.equal((await records()).length, before);
});

test('an old client still stores the frozen v1 prompt label', async () => {
  const seed = 987654;
  assert.equal((await submit({ seed }, 'legacy_test')).status, 200);
  const record = (await records()).at(-1);
  const expected = generatePrompt(quran, seed, undefined, 1);
  assert.equal(record.prompt_version, 1);
  assert.equal(record.kind, expected.label.kind);
  assert.deepEqual(record.mistakes, expected.label.mistakes);
  assert.equal(record.scenario, null);
});


test('a signed in-progress v2 recording remains valid after bank expansion', async () => {
  const { signPrompt } = await import('../server/help/prompt-token.mjs');
  const { verifyPrompt } = await import('../server/help/prompt-token.mjs');
  const p = generatePrompt(quran, 34567, 'vowel', 2);
  const token = signPrompt(p, process.env.HELP_PROMPT_SECRET);
  const restored = verifyPrompt(token, p.seed, process.env.HELP_PROMPT_SECRET);
  assert.equal(restored.v, 2);
  assert.deepEqual(restored.label, p.label);
  assert.equal((await submit({ ...p, token }, 'pending_v2_test')).status, 200);
  const record = (await records()).at(-1);
  assert.equal(record.prompt_version, 2);
  assert.deepEqual(record.mistakes, p.label.mistakes);
  assert.equal((await submit({ ...p, token }, 'version_mismatch_test', { v: '3' })).status, 400);
});
