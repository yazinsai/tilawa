import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { loadReferenceIndex } from '../src/index.js';
import type { PhraseAsset, ReferenceAsset } from '../src/index.js';
import { scoreClip } from '../src/bench/acceptance.js';
import type { AcceptanceClip } from '../src/bench/acceptance.js';
import { replayComponents, replaySettings, REPLAY_DEFAULTS } from '../src/bench/component-replay.js';
import { syntheticFixture, withPosteriors } from '../src/fixtures/index.js';

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL('../assets/ref-v2.json', import.meta.url), 'utf8'));
const phrases: PhraseAsset = JSON.parse(readFileSync(new URL('../assets/phrases-v1.json', import.meta.url), 'utf8'));
const index = loadReferenceIndex(asset);
const from = { surah: 109, ayah: 2, word: 1 }, to = { surah: 109, ayah: 4, word: 4 };
const decoded = withPosteriors(index, syntheticFixture(index, { id: 'replay', from, to, edits: [] }).evidence[0]);
const evidence = { inventory: decoded.inventory, frameMs: decoded.frameMs, nFrames: decoded.posteriors!.nFrames, phonemes: decoded.phonemes, posteriors: { k: decoded.posteriors!.k, ids: [...decoded.posteriors!.ids], probs: [...decoded.posteriors!.probs] } };
const clip: AcceptanceClip = { id: 'replay', ayahs: ['109:2','109:3','109:4'], events: [], nonErrors: [], evidence };

describe('component replay', () => {
  it.each(['free', 'declared'] as const)('reproduces the %s benchmark reports with tracing enabled', arm => {
    const expected = scoreClip(index, clip, { arm, phrases });
    const result = replayComponents(index, phrases, evidence, arm === 'declared' ? { from, to } : null, REPLAY_DEFAULTS);
    expect(result.final).toEqual(expected.final);
    expect(result.steps.at(-1)?.phase).toBe('end');
    expect(result.steps.at(-1)?.toMs).toBe(evidence.nFrames * evidence.frameMs);
    expect(result.steps.some(s => s.trackerOutput.some(e => e.type === 'position' && e.trace))).toBe(true);
  });
  it('sends the same sounds to both components and full posterior rows to the checker', () => {
    const before = JSON.stringify(evidence);
    const result = replayComponents(index, phrases, evidence, null, REPLAY_DEFAULTS);
    for (const s of result.steps.filter(s => s.phase === 'push')) {
      expect(s.trackerInput?.phonemes.map(({ phonePosterior: _local, ...ph }) => ph)).toEqual(s.listenerOutput?.phonemes);
      expect(s.trackerInput?.posteriors).toBeUndefined();
      expect(s.checkerInput.evidence).toBe(s.listenerOutput);
      expect(s.checkerInput.evidence?.posteriors).toBeDefined();
      expect(s.checkerInput.trackerEvents).toBe(s.trackerOutput);
    }
    expect(result.steps.flatMap(s => s.listenerOutput?.phonemes ?? [])).toEqual(evidence.phonemes);
    expect(JSON.stringify(evidence)).toBe(before);
  });
  it('preserves a real skipped-passage report and its intermediate updates', () => {
    const evidence = JSON.parse(readFileSync(new URL('./fixtures/help/muo9h97r_bfad32ee.evidence.json', import.meta.url), 'utf8'));
    const expected = scoreClip(index, { ...clip, evidence, events: [{ errorClass: 'skipped_passage', state: 'suspected', location: { surah: 109, ayah: 3, word: 1 } }] }, { arm: 'declared', phrases });
    const result = replayComponents(index, phrases, evidence, { from, to }, REPLAY_DEFAULTS);
    expect(expected.final.length).toBeGreaterThan(0);
    expect(result.final).toEqual(expected.final);
    expect(result.steps.flatMap(s => s.checkerOutput).length).toBeGreaterThan(0);
  });
  it('rejects invalid controls and keeps defaults independent', () => {
    for (const bad of [{chunkMs:0},{minConfidence:2},{confirmMs:NaN},{localWindow:'true'},{extra:1},{constructor:1},[]]) expect(() => replaySettings(bad)).toThrow();
    const changed = replaySettings({ confirmMs: 0, localWindow: false });
    expect(changed.confirmMs).toBe(0);
    expect(changed.localWindow).toBe(false);
    expect(replaySettings()).toEqual(REPLAY_DEFAULTS);
  });
});
