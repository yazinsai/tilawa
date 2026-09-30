import { createHmac, timingSafeEqual } from 'node:crypto';

// Bind the full issued label, not just its seed: changing the scenario bank,
// requested kind, or corpus must not relabel a recording already in progress.
export function signPrompt(prompt, secret) {
  const payload = Buffer.from(JSON.stringify({ v: prompt.v, seed: prompt.seed, label: prompt.label })).toString('base64url');
  const signature = createHmac('sha256', secret).update(payload).digest('base64url');
  return `${payload}.${signature}`;
}

export function verifyPrompt(token, seed, secret) {
  if (typeof token !== 'string' || token.length > 8192) return null;
  const parts = token.split('.');
  if (parts.length !== 2 || parts.some(p => !/^[A-Za-z0-9_-]+$/.test(p))) return null;
  const [payload, signature] = parts;
  const expected = createHmac('sha256', secret).update(payload).digest();
  const received = Buffer.from(signature, 'base64url');
  if (received.length !== expected.length || !timingSafeEqual(received, expected)) return null;
  try {
    const prompt = JSON.parse(Buffer.from(payload, 'base64url').toString('utf8'));
    return [2, 3].includes(prompt.v) && prompt.seed === seed && prompt.label ? prompt : null;
  } catch {
    return null;
  }
}
