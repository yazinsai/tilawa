import type { DecodedPhoneme, PhonemeEvidence } from "../evidence.js";
import type { ReferenceIndex } from "../reference.js";
import { syntheticFixture } from "../fixtures/index.js";
import type { Edit } from "../fixtures/index.js";
import { createTracker } from "../tracker.js";
import type { WordRef } from "../events.js";

/**
 * How much slower the device floor (2020 mid-range Android running Hermes, no JIT) is assumed to be than the Node
 * machine running the gate. An estimate until device measurements (E7) replace it.
 */
export const DEFAULT_SPEED_FACTOR = 20;
const FRAME_MS = 40;
/** Synthetic frames per phoneme: about 12 phonemes a second at the model's 40 ms stride, a brisk murattal pace. */
const FRAMES_PER_PHONEME = 2;

interface Segment {
  from: WordRef;
  to: WordRef;
  edits?: Edit[];
}

/**
 * The fixed replay: acquisition, long local tracking, a pronunciation slip, noise and `lost`, global re-acquisition,
 * a distant jump and a long silence. Every phase the tracker has, so a slow path anywhere shows up.
 */
const SCRIPT: Segment[] = [
  { from: { surah: 2, ayah: 255, word: 1 }, to: { surah: 2, ayah: 257, word: 10 }, edits: [{ kind: "sub", at: 60, to: "ق" }] },
  { from: { surah: 2, ayah: 1, word: 1 }, to: { surah: 2, ayah: 1, word: 1 }, edits: [{ kind: "noise", at: 0, length: 80, seed: 11 }, { kind: "del", at: 0, count: 99 }] },
  { from: { surah: 36, ayah: 1, word: 1 }, to: { surah: 36, ayah: 20, word: 3 } },
  { from: { surah: 67, ayah: 1, word: 1 }, to: { surah: 67, ayah: 12, word: 4 } },
];
const SILENCE_S = 20;

/** The replay as 1-second chunks of evidence, with its audio length. */
export function perfReplay(index: ReferenceIndex): { evidence: PhonemeEvidence[]; audioS: number } {
  const phonemes: DecodedPhoneme[] = [];
  let frame = 0;
  for (const [n, s] of SCRIPT.entries()) {
    const f = syntheticFixture(index, { id: "perf", from: s.from, to: s.to, edits: s.edits });
    for (const ph of f.evidence[0]?.phonemes ?? []) {
      phonemes.push({ ...ph, start: frame, end: frame + FRAMES_PER_PHONEME - 1 });
      frame += FRAMES_PER_PHONEME;
    }
    if (n === 2) frame += (SILENCE_S * 1000) / FRAME_MS;
  }
  const per = 1000 / FRAME_MS;
  const evidence: PhonemeEvidence[] = [];
  let p = 0;
  for (let a = 0; a < frame; a += per) {
    const chunk: DecodedPhoneme[] = [];
    while (p < phonemes.length && phonemes[p].start < a + per) chunk.push(phonemes[p++]);
    evidence.push({ inventory: index.inventoryId, frameMs: FRAME_MS, phonemes: chunk });
  }
  return { evidence, audioS: (frame * FRAME_MS) / 1000 };
}

export interface PerfResult {
  audioS: number;
  processS: number;
  /** Processing time over audio time on this machine. */
  rtf: number;
  speedFactor: number;
  /** The real-time factor this machine must beat for the device floor to keep up: `1 / speedFactor`. */
  budgetRtf: number;
  /** Slowest single push, in ms on this machine. */
  maxPushMs: number;
  pass: boolean;
}

/** Best of `runs` timed replays, so a scheduler hiccup can't fail the gate on its own. */
export function perfGate(index: ReferenceIndex, speedFactor = DEFAULT_SPEED_FACTOR, runs = 3): PerfResult {
  const { evidence, audioS } = perfReplay(index);
  let best = Infinity;
  let maxPushMs = 0;
  for (let r = 0; r < runs; r++) {
    const tracker = createTracker(index);
    let slowest = 0;
    const t0 = performance.now();
    for (const e of evidence) {
      const p0 = performance.now();
      tracker.push(e);
      slowest = Math.max(slowest, performance.now() - p0);
    }
    tracker.end();
    const ms = performance.now() - t0;
    if (ms < best) {
      best = ms;
      maxPushMs = slowest;
    }
  }
  const processS = best / 1000;
  const rtf = processS / audioS;
  const budgetRtf = 1 / speedFactor;
  return { audioS, processS, rtf, speedFactor, budgetRtf, maxPushMs, pass: rtf <= budgetRtf };
}
