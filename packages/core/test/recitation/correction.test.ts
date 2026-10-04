import { describe, expect, it } from 'vitest';
import { CorrectionController, DEFAULT_CORRECTION_THRESHOLDS, possibleWordIssues, type CorrectionThresholds } from '../../src/recitation/correction';
import type { WordVerdict } from '../../src/recitation/types';
const word = (n: number, patch: Partial<WordVerdict> = {}): WordVerdict => ({ surah: 112, ayah: 3, word: n, wordIndex: 100 + n, state: 'ok', distance: 0, margin: .9, heardRatio: 1, vowelErrors: 0, vowelMargin: 0, ...patch });
const correct = [word(0), word(1), word(2), word(3)];
const omission = [word(0), word(1, { state: 'skipped', distance: 1, heardRatio: 0, margin: 0 }), word(2), word(3)];
const substitution = [word(0), word(1, { state: 'wrong', distance: .8 }), word(2), word(3)];
const vowel = [word(0), word(1, { distance: .02, vowelErrors: 1, vowelMargin: .8 }), word(2), word(3)];
const cursor = { surah: 112, ayah: 3, word: 3 };
// Rules v2 GOP shape (GOP is off by default).
const V2_GOP: CorrectionThresholds = { ...DEFAULT_CORRECTION_THRESHOLDS, gopFlag: -3, gopAnchor: -2, gopOnSkipped: true, gopOmission: true, settleGop: true };
function flag() {
  const c = new CorrectionController(); c.setMode('correction');
  expect(c.observe(omission, cursor, 30)).toBe(false);
  expect(c.observe(omission, cursor, 42)).toBe(true); return c;
}
describe('conservative word correction', () => {
  it('defaults to Tracking and leaves correct recitation unflagged', () => {
    const c = new CorrectionController(); c.observe(omission, cursor, 30); c.observe(omission, cursor, 50);
    expect(c.state.phase).toBe('idle'); expect(possibleWordIssues(correct)).toEqual([]);
  });
  it('detects interior omissions and gross substitutions with clear anchors', () => {
    expect(possibleWordIssues(omission)[0]).toMatchObject({ word: 1, kind: 'possible_omission' });
    expect(possibleWordIssues(substitution)[0]).toMatchObject({ word: 1, kind: 'possible_substitution' });
  });
  it('treats a partly heard skipped word as an omission (omissionMaxHeard)', () => {
    const partial = [word(0), word(1, { state: 'skipped', distance: 1, heardRatio: .25, margin: 0 }), word(2)];
    expect(possibleWordIssues(partial)[0]).toMatchObject({ word: 1, kind: 'possible_omission' });
    expect(possibleWordIssues(partial, { vowelMargin: .05, vowelWordMargin: .5, omissionMaxHeard: 0 })).toEqual([]);
    expect(possibleWordIssues(partial, { vowelMargin: .05, vowelWordMargin: .5, omissionMaxHeard: .2 })).toEqual([]);
    // Thresholds objects from before the field existed keep working.
    expect(possibleWordIssues(partial, { vowelMargin: .05, vowelWordMargin: .5 })).toHaveLength(1);
    // Still needs clear anchors on both sides.
    expect(possibleWordIssues([word(0, { margin: .2 }), partial[1]!, word(2)])).toEqual([]);
  });
  it('GOP rule: a disputed word that fits badly, between words that fit', () => {
    const gp = (v: WordVerdict[], th: Partial<CorrectionThresholds> = {}) => possibleWordIssues(v, { ...V2_GOP, ...th });
    // Neighbours are not `clear` (low margin) but fit acoustically (GOP ~0).
    const fit = (n: number) => word(n, { margin: .3, gop: -.2, gopNone: -9 });
    const bad = { state: 'wrong' as const, distance: .45, margin: .4, heardRatio: .8, gop: -6, gopNone: -9 };
    expect(gp([fit(0), word(1, bad), fit(2)])[0]).toMatchObject({ word: 1, kind: 'possible_substitution' });
    // Silence explains the window better than the expected word: omission.
    expect(gp([fit(0), word(1, { ...bad, gopNone: -1 }), fit(2)])[0]).toMatchObject({ kind: 'possible_omission' });
    expect(gp([fit(0), word(1, { ...bad, state: 'skipped', heardRatio: .2 }), fit(2)])[0]).toMatchObject({ kind: 'possible_omission' });
    // GOP alone never flags a word the aligner accepts or is unsure about.
    for (const state of ['ok', 'unsure', 'pending'] as const) {
      expect(gp([fit(0), word(1, { ...bad, state, gop: -15 }), fit(2)])).toEqual([]);
    }
    // Not low enough, not the local minimum, or a neighbour that does not fit.
    expect(gp([fit(0), word(1, { ...bad, gop: -2.5 }), fit(2)])).toEqual([]);
    expect(gp([fit(0), word(1, bad), word(2, { margin: .3, gop: -8 })])).toEqual([]);
    expect(gp([fit(0), word(1, bad), word(2, { margin: .3, gop: -2.5 })])).toEqual([]);
    expect(gp([fit(0), word(1, bad), word(2, { margin: .3 })])).toEqual([]);
    // Other ayah, disabled, or no GOP: the margin/distance rules only.
    expect(gp([fit(0), word(1, bad), fit(2), word(3, { ayah: 4, word: 0 })].slice(0, 2))).toEqual([]);
    expect(gp([fit(0), word(1, bad), fit(2)], { vowelMargin: .05, vowelWordMargin: .5, gopFlag: -Infinity })).toEqual([]);
    expect(gp([word(0), word(1, { ...bad, gop: undefined }), word(2)])).toEqual([]);
  });
  it('repetition: a word that fits once but gains from a second copy', () => {
    const rep = word(1, { gop: -.3, gopNone: -9, repGain: 7 });
    const ctx = (r = rep, b = word(0), a = word(2)) => [b, r, a];
    expect(possibleWordIssues(ctx())[0]).toMatchObject({ word: 1, kind: 'possible_repetition' });
    // Fires without a clear right neighbour as long as the left one is clear.
    expect(possibleWordIssues(ctx(rep, word(0), word(2, { margin: .3 })))[0]).toMatchObject({ kind: 'possible_repetition' });
    // Phrase restart: a neighbour that also repeats vetoes it.
    expect(possibleWordIssues(ctx(rep, word(0, { repGain: 6 })))).toEqual([]);
    expect(possibleWordIssues(ctx(rep, word(0), word(2, { repGain: 6 })))).toEqual([]);
    // Small gain, a word that does not fit, an unclear left neighbour, disabled, or no GOP.
    expect(possibleWordIssues(ctx({ ...rep, repGain: 4 }))).toEqual([]);
    expect(possibleWordIssues(ctx({ ...rep, gop: -2.5 }))).toEqual([]);
    expect(possibleWordIssues(ctx(rep, word(0, { margin: .3 })))).toEqual([]);
    expect(possibleWordIssues(ctx(), { vowelMargin: .05, vowelWordMargin: .5, repetitionGain: Infinity })).toEqual([]);
    expect(possibleWordIssues(ctx({ ...rep, gop: undefined }))).toEqual([]);
  });
  it('GOP gating knobs: states, kinds, local minimum, omission margin', () => {
    const gp = (v: WordVerdict[], th: Partial<CorrectionThresholds> = {}) => possibleWordIssues(v, { ...V2_GOP, ...th });
    const fit = (n: number) => word(n, { margin: .3, gop: -.2, gopNone: -9 });
    const bad = { state: 'wrong' as const, distance: .45, margin: .4, heardRatio: .8, gop: -6, gopNone: -9 };
    const base = {};
    const sub = [fit(0), word(1, bad), fit(2)];
    const skip = [fit(0), word(1, { ...bad, state: 'skipped', heardRatio: .2 }), fit(2)];
    expect(gp(sub, { ...base, gopOnWrong: false })).toEqual([]);
    expect(gp(skip, { ...base, gopOnSkipped: false })).toEqual([]);
    expect(gp(sub, { ...base, gopSubstitution: false })).toEqual([]);
    expect(gp(skip, { ...base, gopOmission: false })).toEqual([]);
    expect(gp(skip, { ...base, gopSubstitution: false })[0]).toMatchObject({ kind: 'possible_omission' });
    const worseRight = [fit(0), word(1, bad), word(2, { margin: .3, gop: -1.5, gopNone: -9 }), fit(3)];
    expect(gp(worseRight, base)[0]).toMatchObject({ word: 1 });
    const silent = [fit(0), word(1, { ...bad, gopNone: -2 }), fit(2)];
    expect(gp(silent, base)[0]).toMatchObject({ kind: 'possible_omission' });
    expect(gp(silent, { ...base, gopNoneMargin: 5 })[0]).toMatchObject({ kind: 'possible_substitution' });
    expect(gp(silent, { ...base, gopNoneMin: -1 })[0]).toMatchObject({ kind: 'possible_substitution' });
  });
  it('GOP-only flags can need longer persistence, and can be kept out of settle()', () => {
    const fit = (n: number) => word(n, { margin: .3, gop: -.2, gopNone: -9 });
    const sub = [fit(0), word(1, { state: 'wrong', distance: .45, margin: .4, heardRatio: .8, gop: -6, gopNone: -9 }), fit(2), fit(3)];
    const c = new CorrectionController(); c.setMode('correction');
    c.thresholds = { ...c.thresholds, gopFlag: -3, gopPersistFrames: 24 };
    c.observe(sub, cursor, 10); expect(c.observe(sub, cursor, 22)).toBe(false);
    expect(c.observe(sub, cursor, 34)).toBe(true);
    const s = new CorrectionController(); s.setMode('correction');
    s.thresholds = { ...s.thresholds, gopFlag: -3, settleGop: false };
    expect(s.settle(sub, cursor)).toBe(false);
    s.thresholds = { ...s.thresholds, settleGop: true };
    expect(s.settle(sub, cursor)).toBe(true);
  });
  it('defaults: GOP off, repetition is a note, vowel flags need a confident word', () => {
    const fit = (n: number) => word(n, { margin: .3, gop: -.2, gopNone: -9 });
    const bad = word(1, { state: 'wrong', distance: .45, margin: .4, heardRatio: .8, gop: -9, gopNone: -9 });
    expect(possibleWordIssues([fit(0), bad, fit(2)])).toEqual([]);
    expect(possibleWordIssues([fit(0), bad, fit(2)], { ...DEFAULT_CORRECTION_THRESHOLDS, gopFlag: -5 })[0])
      .toMatchObject({ kind: 'possible_substitution' });
    expect(DEFAULT_CORRECTION_THRESHOLDS.repetitionMode).toBe('note');
    expect(possibleWordIssues([word(0), { ...vowel[1]!, margin: .7 }, word(2)])).toEqual([]);
  });
  it('repetition as a soft note never interrupts', () => {
    const rep = [word(0), word(1, { gop: -.3, repGain: 7 }), word(2), word(3)];
    const c = new CorrectionController(); c.setMode('correction');
    c.thresholds = { ...c.thresholds, repetitionGain: 5, repetitionMode: 'note' };
    c.observe(rep, cursor, 30); expect(c.observe(rep, cursor, 42)).toBe(false);
    expect(c.state.phase).toBe('idle');
    expect(c.takeNotes()).toEqual([expect.objectContaining({ kind: 'possible_repetition', word: 1 })]);
    expect(c.takeNotes()).toEqual([]);
    // Noted once: the same word is not noted again.
    c.observe(rep, cursor, 60); c.observe(rep, cursor, 80); expect(c.takeNotes()).toEqual([]);
    const off = new CorrectionController(); off.setMode('correction');
    off.thresholds = { ...off.thresholds, repetitionGain: 5, repetitionMode: 'off' };
    off.observe(rep, cursor, 30); off.observe(rep, cursor, 42);
    expect(off.state.phase).toBe('idle'); expect(off.takeNotes()).toEqual([]);
  });
  it('does not accept a retry that repeats the word again', () => {
    const rep = [word(0), word(1, { gop: -.3, repGain: 7 }), word(2), word(3)];
    const c = new CorrectionController(); c.setMode('correction');
    c.thresholds = { ...c.thresholds, repetitionMode: 'flag' };
    c.observe(rep, cursor, 30); expect(c.observe(rep, cursor, 42)).toBe(true);
    expect(c.state.issue).toMatchObject({ kind: 'possible_repetition', word: 1 });
    c.act('retry');
    c.observe(rep, cursor, 100); c.observe(rep, cursor, 120); expect(c.state.phase).toBe('retrying');
    c.observe(correct, cursor, 130); c.observe(correct, cursor, 142); expect(c.state.phase).toBe('corrected');
  });
  it('detects a confident harakah error on an otherwise matching word', () => {
    expect(possibleWordIssues(vowel)[0]).toMatchObject({ word: 1, kind: 'possible_vowel' });
    // Same skeleton, but the decoder was unsure which vowel it heard, or the word itself was weak.
    for (const patch of [{ vowelMargin: .01 }, { vowelErrors: 0 }, { margin: .3 }, { distance: .3 }, { state: 'wrong' as const }, { heardRatio: .5 }]) {
      expect(possibleWordIssues([word(0), { ...vowel[1]!, ...patch }, word(2)])).toEqual([]);
    }
    // Thresholds are tunable per controller.
    expect(possibleWordIssues(vowel, { vowelMargin: .9, vowelWordMargin: .5 })).toEqual([]);
    expect(possibleWordIssues([word(0), { ...vowel[1]!, vowelMargin: .2 }, word(2)], { vowelMargin: .1, vowelWordMargin: .5 })).toHaveLength(1);
    // A contested vowel (p(heard) 0.54 vs p(expected) 0.44, the rabbuka case) still counts by default.
    expect(possibleWordIssues([word(0), { ...vowel[1]!, vowelMargin: .1 }, word(2)])).toHaveLength(1);
  });
  it('does not accept a retry that repeats the vowel error', () => {
    const c = new CorrectionController(); c.setMode('correction');
    c.observe(vowel, cursor, 30); expect(c.observe(vowel, cursor, 42)).toBe(true);
    expect(c.state.issue).toMatchObject({ kind: 'possible_vowel', word: 1 });
    c.act('retry');
    c.observe(vowel, cursor, 100); c.observe(vowel, cursor, 120); expect(c.state.phase).toBe('retrying');
    c.observe(correct, cursor, 130); c.observe(correct, cursor, 142); expect(c.state.phase).toBe('corrected');
  });
  it('abstains on unclear audio, pending alignment, small phonetic differences and boundary omissions', () => {
    for (const patch of [{ state: 'unsure' as const }, { state: 'pending' as const }, { margin: .2 }, { margin: NaN }, { distance: .45 }, { heardRatio: .2 }]) {
      expect(possibleWordIssues([word(0), { ...substitution[1], ...patch }, word(2)])).toEqual([]);
    }
    expect(possibleWordIssues([word(0, { margin: .2 }), omission[1], word(2)])).toEqual([]);
    expect(possibleWordIssues(omission.slice(1))).toEqual([]);
    expect(possibleWordIssues(omission.slice(0, 2))).toEqual([]);
    expect(possibleWordIssues([word(0), omission[1], word(2, { ayah: 4 })])).toEqual([]);
  });
  it('requires advancing frames and cancels a revised hypothesis', () => {
    const c = new CorrectionController(); c.setMode('correction');
    c.observe(omission, cursor, 30); c.observe(omission, cursor, 30); expect(c.state.phase).toBe('idle');
    c.observe(correct, cursor, 50); c.observe(omission, cursor, 60); expect(c.state.phase).toBe('idle');
    c.observe(omission, cursor, 72); expect(c.state.phase).toBe('error');
  });
  it('keeps position and distinguishes dismissal from correction', () => {
    const c = flag(); c.act('dismiss');
    expect(c.state).toMatchObject({ phase: 'idle', outcome: 'dismissed', resume: cursor });
    c.observe(omission, cursor, 60); c.observe(omission, cursor, 90); expect(c.state.phase).toBe('idle');
    c.reset(); c.observe(omission, cursor, 10); c.observe(omission, cursor, 25); expect(c.state.phase).toBe('error');
  });
  it('requires fresh clear retry prefix, rejects stale evidence and supports practicing again', () => {
    const c = flag(); const old = c.state.attempt; c.act('retry');
    c.observe(correct, cursor, 100, old); c.observe(correct, cursor, 120, old); expect(c.state.phase).toBe('retrying');
    c.observe([word(1)], cursor, 20); c.observe([word(1)], cursor, 40);
    c.observe([word(0), word(1, { margin: .2 })], cursor, 60); expect(c.state.phase).toBe('retrying');
    c.observe(correct, cursor, 70); c.observe(correct, cursor, 82);
    expect(c.state).toMatchObject({ phase: 'corrected', outcome: 'corrected', resume: cursor });
    c.act('retry'); expect(c.state.phase).toBe('retrying'); c.act('stop_retry'); expect(c.state.phase).toBe('error');
    c.act('retry'); c.observe(correct, cursor, 10); c.observe(correct, cursor, 22); c.act('continue');
    expect(c.state).toMatchObject({ phase: 'idle', outcome: 'corrected', resume: cursor });
  });
  it('raise() takes ayah-level issues and a retry must clear all their words', () => {
    const issue = { surah: 112, ayah: 3, word: 0, wordIndex: 100, kind: 'unclear_ayah' as const, words: 4 };
    const c = new CorrectionController();
    expect(c.raise(issue, cursor)).toBe(false); // tracking mode
    c.setMode('correction');
    expect(c.raise(issue, cursor)).toBe(true);
    expect(c.state).toMatchObject({ phase: 'error', issue, resume: cursor });
    expect(c.raise(issue, cursor)).toBe(false); // not idle
    c.act('retry');
    c.observe([word(0), word(1)], cursor, 10); c.observe([word(0), word(1)], cursor, 30);
    expect(c.state.phase).toBe('retrying');
    c.observe(correct, cursor, 40); c.observe(correct, cursor, 52);
    expect(c.state.phase).toBe('corrected');
    c.act('continue');
    expect(c.raise(issue, cursor)).toBe(false); // suppressed for the session
    c.reset();
    expect(c.raise({ ...issue, kind: 'possible_skipped_ayah' }, cursor)).toBe(true);
    c.act('dismiss');
    expect(c.state).toMatchObject({ phase: 'idle', outcome: 'dismissed' });
    expect(c.raise(issue, cursor)).toBe(false);
  });
  it('settle() checks the words observe() never saw in full context', () => {
    // The skipped word is second to last: observe() never saw the word after next settled.
    const tail = [word(0), word(1), word(2, { state: 'skipped', distance: 1, heardRatio: 0, margin: 0 }), word(3)];
    const live = tail.map((v, i) => i === 3 ? { ...v, state: 'pending' as const } : v);
    const c = new CorrectionController(); c.setMode('correction');
    c.observe(live, cursor, 10); c.observe(live, cursor, 22); expect(c.state.phase).toBe('idle');
    expect(c.settle(tail, cursor)).toBe(true);
    expect(c.state).toMatchObject({ phase: 'error', issue: { word: 2, kind: 'possible_omission' }, resume: cursor });
    // A word observe() already judged with settled context had its chance.
    const d = new CorrectionController(); d.setMode('correction');
    d.observe([word(0, { margin: .2 }), ...omission.slice(1)], cursor, 10);
    expect(d.settle(omission, cursor)).toBe(false);
    // Vowel flags stay observe-only; settle can be turned off; tracking mode never settles.
    const e = new CorrectionController(); e.setMode('correction');
    expect(e.settle([word(0), word(1), { ...vowel[1]!, word: 2, wordIndex: 102 }, word(3)], cursor)).toBe(false);
    const f = new CorrectionController(); f.setMode('correction');
    f.thresholds = { ...f.thresholds, settle: false };
    expect(f.settle(tail, cursor)).toBe(false);
    expect(new CorrectionController().settle(tail, cursor)).toBe(false);
  });
  it('slip head is off unless slipFlag is set, and the aligner still names the kind', () => {
    const hot = { slip: 0.9 };
    const ctx = (mid: Partial<WordVerdict>) => [word(0), word(1, { ...hot, ...mid }), word(2)];
    // Default threshold is Infinity: a high score on a word the aligner accepts does not flag.
    expect(possibleWordIssues(ctx({ state: 'ok', distance: 0.02, vowelErrors: 0 }))).toEqual([]);
    const on = { slipFlag: 0.5 };
    expect(possibleWordIssues(ctx({ state: 'skipped', distance: 1, heardRatio: 0, margin: 0 }), on)[0])
      .toMatchObject({ word: 1, kind: 'possible_omission' });
    // distance ≤ 0.15 is a vowel even when the aligner counted no vowel error.
    expect(possibleWordIssues(ctx({ state: 'ok', distance: 0.02, vowelErrors: 0 }), on)[0])
      .toMatchObject({ word: 1, kind: 'possible_vowel' });
    // Wrong, but under the aligner's distance/margin gates: the head still flags it as a substitution.
    expect(possibleWordIssues(ctx({ state: 'wrong', distance: 0.4, margin: 0.4, heardRatio: 0.8 }), on)[0])
      .toMatchObject({ word: 1, kind: 'possible_substitution' });
    // Below the cutoff: no head flag. A neighbour that is not itself clear still
    // counts; the head only needs both neighbours in the same ayah.
    expect(possibleWordIssues(ctx({ state: 'ok', distance: 0.02, vowelErrors: 0, slip: 0.4 }), on)).toEqual([]);
    expect(possibleWordIssues([word(0, { margin: 0.2 }), word(1, { state: 'skipped', distance: 1, heardRatio: 0, margin: 0, slip: 0.9 }), word(2)], on)[0])
      .toMatchObject({ word: 1, kind: 'possible_omission' });
    // The aligner already flagged it: that kind wins, the head does not replace it.
    const both = possibleWordIssues(ctx({ state: 'skipped', distance: 1, heardRatio: 0, margin: 0 }), on);
    expect(both).toHaveLength(1);
    expect(both[0]).toMatchObject({ kind: 'possible_omission' });
    // A slip vowel can be raised at settle. An aligner vowel still cannot.
    const c = new CorrectionController(); c.setMode('correction');
    c.thresholds = { ...c.thresholds, ...on };
    const slipVowel = [word(0), word(1, { distance: 0.02, vowelErrors: 0, slip: 0.9 }), word(2), word(3)];
    expect(c.settle(slipVowel, cursor)).toBe(true);
    expect(c.state.issue).toMatchObject({ word: 1, kind: 'possible_vowel' });
  });
  it('closing or reviewing later never claims success', () => {
    for (const action of ['close', 'review_later'] as const) {
      const c = flag(); c.act('retry'); c.act(action);
      expect(c.state).toMatchObject({ phase: 'idle', outcome: 'deferred', resume: cursor });
    }
  });
});
