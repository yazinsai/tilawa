import { describe, expect, it } from 'vitest';
import { FramePosteriors, GOP_FLOOR, encodePhonemeSpan, encodePhonemes, forcedLogLik, freeLogLik, pairScores, wordGop } from '../../src/recitation/posteriors';
import { BLANK_ID, TOKENS, VOCAB_SIZE } from '../../src/recitation/tokens';

const id = (t: string) => TOKENS.indexOf(t);
/** One frame per symbol: the named token at p≈0.9, everything else sharing the rest. */
function frames(seq: Array<string | null>): Float32Array {
  const lp = new Float32Array(seq.length * VOCAB_SIZE).fill(Math.log(0.1 / (VOCAB_SIZE - 1)));
  seq.forEach((s, t) => { lp[t * VOCAB_SIZE + (s === null ? BLANK_ID : id(s))] = Math.log(0.9); });
  return lp;
}
const ring = (seq: Array<string | null>, at = 0) => { const p = new FramePosteriors(VOCAB_SIZE, 64); p.push(frames(seq), seq.length, at); return p; };
const B = 'بَ', T = 'تُ', A = 'اا';

describe('frame posteriors + CTC GOP', () => {
  it('tokenizes phonemes greedily, longest token first', () => {
    expect(encodePhonemes(B + A)).toEqual([id(B), id(A)]);
    expect(encodePhonemes('\u0001')).toBeNull();
  });
  it('scores the expected word ~0 and a different word well below', () => {
    const p = ring([null, B, B, null, A, null]);
    expect(forcedLogLik(p, [id(B), id(A)], 0, 6)).toBeCloseTo(freeLogLik(p, 0, 6), 5);
    const good = wordGop(p, [id(B), id(A)], 0, 6)!;
    const bad = wordGop(p, [id(T), id(A)], 0, 6)!;
    expect(good.gop).toBeCloseTo(0, 5);
    expect(bad.gop).toBeLessThan(-2);
    // Speech present: the silence hypothesis is worse than the right word.
    expect(good.gopNone).toBeLessThan(good.gop - 2);
  });
  it('separates repetition and omission hypotheses', () => {
    const twice = wordGop(ring([B, null, A, null, B, null, A]), [id(B), id(A)], 0, 7)!;
    expect(twice.gopTwice).toBeCloseTo(0, 5);
    expect(twice.gop).toBeLessThan(-2);
    const gap = wordGop(ring([null, null]), [id(B), id(A)], 0, 2)!;
    expect(gap.gopNone).toBeCloseTo(0, 5);
    expect(gap.gop).toBeLessThan(-5);
    expect(wordGop(ring([null]), [id(B), id(A)], 0, 1)!.gop).toBe(GOP_FLOOR); // one frame cannot hold two tokens
    expect(wordGop(ring([B]), [id(B)], 1, 1)!.gop).toBe(GOP_FLOOR); // empty window
  });
  it('pair scores: a repeated word gains, a boundary misallocation fits as a pair', () => {
    const w = [id(B)], next = [id(T), id(A)];
    const rep = pairScores(ring([B, null, B, null, T, null, A]), 1, [...w, ...next], [...w, ...w, ...next], 0, 7)!;
    expect(rep.repGain).toBeGreaterThan(2);
    expect(rep.pairGop).toBeLessThan(-2);
    const clean = pairScores(ring([B, null, T, null, A]), 1, [...w, ...next], [...w, ...w, ...next], 0, 5)!;
    expect(clean.pairGop).toBeCloseTo(0, 5);
    expect(clean.repGain).toBeLessThan(-2);
  });
  it('tokenizes a word in context, keeping a token that straddles its boundary', () => {
    // "اا" spans the end of word 1 and the start of word 2.
    const text = B + 'ا' + 'ا' + T; // word 1 = chars [0, 3), word 2 = [3, 6)
    expect(encodePhonemeSpan(text, 0, 3)).toEqual([id(B), id(A)]);
    expect(encodePhonemeSpan(text, 3, 6)).toEqual([id(A), id(T)]);
  });
  it('identical consecutive tokens need a blank between them', () => {
    expect(forcedLogLik(ring([B, B]), [id(B), id(B)], 0, 2)).toBeLessThan(forcedLogLik(ring([B, null, B]), [id(B), id(B)], 0, 3));
    expect(forcedLogLik(ring([B]), [id(B), id(B)], 0, 1)).toBe(-Infinity);
  });
  it('keeps a bounded window of decoder frames and resets on a frame jump', () => {
    const p = new FramePosteriors(VOCAB_SIZE, 4);
    p.push(frames([B, B, B, B, B, B]), 6, 0);
    expect([p.firstFrame, p.endFrame]).toEqual([2, 6]);
    expect(p.has(1, 3)).toBe(false);
    expect(wordGop(p, [id(B)], 0, 3)).toBeNull();
    p.push(frames([A]), 1, 0);
    expect([p.firstFrame, p.endFrame]).toEqual([0, 1]);
    expect(p.argmax(0)).toBe(id(A));
  });
});
