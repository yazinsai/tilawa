import { test } from 'vitest';
import assert from 'node:assert/strict';
import { signPrompt, verifyPrompt } from '../server/help/prompt-token.mjs';

const prompt = { v: 2, seed: 123, label: { kind: 'substitution', ref: '7:111', scenario: 'parallel-send', mistakes: [{ word: 4, said: 'وابعث' }] } };
const secret = 'test-only-key';

test('the issued label survives generator changes and forced kind requests', () => {
  const token = signPrompt(prompt, secret);
  prompt.label.scenario = 'changed-after-issuance';
  const verified = verifyPrompt(token, 123, secret);
  assert.equal(verified.label.kind, 'substitution');
  assert.equal(verified.label.scenario, 'parallel-send');
  assert.deepEqual(verified.label.mistakes, [{ word: 4, said: 'وابعث' }]);
  prompt.label.scenario = 'parallel-send';
});

test('cannot alter label, seed or signature to relabel a recording', () => {
  const token = signPrompt(prompt, secret);
  assert.equal(verifyPrompt(token, 124, secret), null);
  assert.equal(verifyPrompt(token, 123, 'another-key'), null);
  const [payload, signature] = token.split('.');
  const altered = JSON.parse(Buffer.from(payload, 'base64url').toString());
  altered.label.kind = 'none';
  assert.equal(verifyPrompt(`${Buffer.from(JSON.stringify(altered)).toString('base64url')}.${signature}`, 123, secret), null);
  for (const invalid of ['', '.', 'a.b.c', `${payload}.a`, 'x'.repeat(8193), undefined]) assert.equal(verifyPrompt(invalid, 123, secret), null);
});
