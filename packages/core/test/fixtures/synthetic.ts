import { createChecker } from "../../src/engine/checker.js";
import type { CheckerOptions } from "../../src/engine/checker.js";
import type { CheckerEvent, EngineEvent, PositionEvent, TrackerState, WordRef, WordSpan } from "../../src/engine/events.js";
import type { DecodedPhoneme, PhonemeEvidence } from "../../src/engine/evidence.js";
import type { ReferenceIndex } from "../../src/engine/reference.js";
import { createTracker } from "../../src/engine/tracker.js";
import type { TrackerOptions } from "../../src/engine/tracker.js";

export type Edit =
  /** Offsets are into the reference phonemes of the fixture span, before any edit is applied. */
  | { kind: "sub"; at: number; to: string }
  | { kind: "del"; at: number; count: number }
  | { kind: "ins"; at: number; text: string }
  /** Re-recite reference phonemes [from, to) immediately after `to`. */
  | { kind: "repeat"; from: number; to: number }
  | { kind: "noise"; at: number; length: number; seed: number }
  | { kind: "gap"; at: number; frames: number };

export interface SyntheticProvenance {
  kind: "synthetic";
  from: WordRef;
  to: WordRef;
  edits: Edit[];
}

export interface TlogProvenance {
  kind: "tlog";
  shard: string;
  row: number;
  path: string;
  filenameLabel: { surah: number; ayah: number };
  /** True only after someone listened to the clip and confirmed `expect`. */
  audioVerified: boolean;
  reviewNote?: string;
}

export interface Fixture {
  id: string;
  provenance: SyntheticProvenance | TlogProvenance;
  evidence: PhonemeEvidence[];
  /** Asserted only for synthetic fixtures and audio-verified real clips. */
  /** `edgeWords`: span edges may differ by this many words, for reviewer spans whose half-word edges vary. */
  expect?: { state: TrackerState; span?: WordSpan; edgeWords?: number };
}

export function isAssertable(f: Fixture): boolean {
  return f.expect !== undefined && (f.provenance.kind === "synthetic" || f.provenance.audioVerified);
}

export function replay(index: ReferenceIndex, fixture: Fixture, options: TrackerOptions = {}): EngineEvent[] {
  const tracker = createTracker(index, options);
  const events: EngineEvent[] = [];
  for (const chunk of fixture.evidence) events.push(...tracker.push(chunk));
  events.push(...tracker.end());
  return events;
}

/** Replay through the tracker and a checker fed with its events; returns the checker events only. */
export function replayChecked(index: ReferenceIndex, fixture: Fixture, options: TrackerOptions = {}, checkerOptions: CheckerOptions = {}): CheckerEvent[] {
  const tracker = createTracker(index, options);
  const checker = createChecker(index, checkerOptions);
  const out: CheckerEvent[] = [];
  for (const chunk of fixture.evidence) out.push(...checker.push(chunk, tracker.push(chunk)));
  out.push(...checker.end(tracker.end()));
  return out;
}

export function finalPosition(events: EngineEvent[]): PositionEvent | undefined {
  return events.filter((e): e is PositionEvent => e.type === "position").at(-1);
}

export interface JumpReport {
  /** Time of the first locked event on the jump target; null if the tracker never got there. */
  committedMs: number | null;
  /** `committedMs` minus the onset of the jump in the audio. */
  latencyMs: number | null;
  /** Locked runs that reported a word off the recited path and away from it. */
  falseJumps: number;
  /** Locked runs off the path but `near` it: the local search briefly taking a skip or restart. */
  localFlashes: number;
}

/** Jump commit latency and false jumps, from events alone, given which words the recitation actually covered. */
export function jumpReport(
  events: EngineEvent[],
  o: { onsetMs: number; target: (w: WordRef) => boolean; path: (w: WordRef) => boolean; near?: (w: WordRef) => boolean },
): JumpReport {
  let committedMs: number | null = null;
  let falseJumps = 0;
  let localFlashes = 0;
  let off = false;
  for (const e of events) {
    if (e.type !== "position" || e.state !== "locked") continue;
    const word = e.observed[0].word;
    if (committedMs === null && o.target(word)) committedMs = e.timeMs;
    const now = !o.path(word) && !o.target(word);
    if (now && !off) {
      if (o.near?.(word)) localFlashes++;
      else falseJumps++;
    }
    off = now;
  }
  return { committedMs, latencyMs: committedMs === null ? null : committedMs - o.onsetMs, falseJumps, localFlashes };
}

const fmt = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
export const formatSpan = (s: WordSpan) => `${fmt(s.start)}–${fmt(s.end)}`;

/** Human-readable mismatches between a fixture's expectation and its replayed events. */
export function diffExpectation(fixture: Fixture, events: EngineEvent[], index?: ReferenceIndex): string[] {
  if (!fixture.expect) return [];
  const final = finalPosition(events);
  if (!final) return ["no position event emitted"];
  const problems: string[] = [];
  if (final.state !== fixture.expect.state) problems.push(`state: expected ${fixture.expect.state}, got ${final.state}`);
  if (fixture.expect.span) {
    const got = final.observed[0]?.span;
    const want = formatSpan(fixture.expect.span);
    if (!got) problems.push(`span: expected ${want}, got none`);
    else if (!spanWithin(got, fixture.expect.span, fixture.expect.edgeWords ?? 0, index)) {
      problems.push(`span: expected ${want}${fixture.expect.edgeWords ? ` ±${fixture.expect.edgeWords} words` : ""}, got ${formatSpan(got)}`);
    }
  }
  return problems;
}

function spanWithin(got: WordSpan, want: WordSpan, edgeWords: number, index?: ReferenceIndex): boolean {
  if (!edgeWords) return formatSpan(got) === formatSpan(want);
  if (!index) throw new Error("an edgeWords expectation needs the reference index");
  const at = (r: WordRef) => index.words.findIndex((w) => w.surah === r.surah && w.ayah === r.ayah && w.word === r.word);
  return Math.abs(at(got.start) - at(want.start)) <= edgeWords && Math.abs(at(got.end) - at(want.end)) <= edgeWords;
}

const FRAMES_PER_PHONEME = 3;

/**
 * Synthetic top-k posteriors for one chunk: each decoded phoneme's frames favour its token, and every other frame
 * favours the blank. `override(j)` replaces the distribution on phoneme `j`'s frames, as `[token, p]` pairs.
 */
export function withPosteriors(
  index: ReferenceIndex,
  evidence: PhonemeEvidence,
  override: (j: number) => [string, number][] | undefined = () => undefined,
  k = 5,
): PhonemeEvidence {
  const ids = [...index.tokens];
  const id = (t: string) => ids.indexOf(t) + 1;
  const startFrame = evidence.phonemes[0]?.start ?? 0;
  const nFrames = (evidence.phonemes.at(-1)?.end ?? startFrame) + 2 - startFrame;
  const rows: [number, number][][] = Array.from({ length: nFrames }, () => [[0, 0.95]]);
  evidence.phonemes.forEach((ph, j) => {
    const dist = override(j) ?? [[ph.p, 0.9], ["", 0.05]];
    for (let f = ph.start; f <= ph.end; f++) rows[f - startFrame] = dist.map(([t, p]) => [t ? id(t) : 0, p]);
  });
  const flatIds: number[] = [];
  const flatProbs: number[] = [];
  for (const row of rows) {
    const sorted = [...row].sort((a, b) => b[1] - a[1]);
    // Pad with distinct unlisted-looking filler so every row has k entries.
    for (let t = 1; sorted.length < k; t++) if (!sorted.some(([i]) => i === t)) sorted.push([t, 0]);
    for (const [i, p] of sorted.slice(0, k)) {
      flatIds.push(i);
      flatProbs.push(p);
    }
  }
  return { ...evidence, posteriors: { k, startFrame, nFrames, ids: flatIds, probs: flatProbs } };
}

function seededTokens(tokens: string, length: number, seed: number): string {
  let s = seed >>> 0 || 1;
  let out = "";
  for (let i = 0; i < length; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out += tokens[s % tokens.length];
  }
  return out;
}

export interface SyntheticOptions {
  id: string;
  from: WordRef;
  to: WordRef;
  edits?: Edit[];
  /** Phonemes per pushed chunk; the whole fixture is one chunk when omitted. */
  chunkSize?: number;
  expect?: Fixture["expect"];
}

export function syntheticFixture(index: ReferenceIndex, o: SyntheticOptions): Fixture {
  const find = (r: WordRef) => {
    const i = index.words.findIndex((w) => w.surah === r.surah && w.ayah === r.ayah && w.word === r.word);
    if (i === -1) throw new Error(`Unknown word ${fmt(r)}`);
    return index.words[i];
  };
  const ref = index.phonemes.slice(find(o.from).start, find(o.to).end);
  const edits = o.edits ?? [];
  const tokens = [...index.tokens].join("");

  // Each reference position becomes zero or more emitted phonemes plus a frame gap before it.
  const before: string[] = Array.from({ length: ref.length + 1 }, () => "");
  const replaced: (string | null)[] = [...ref];
  const gapBefore = new Array<number>(ref.length + 1).fill(0);
  for (const e of edits) {
    switch (e.kind) {
      case "sub":
        replaced[e.at] = e.to;
        break;
      case "del":
        for (let k = e.at; k < e.at + e.count; k++) replaced[k] = null;
        break;
      case "ins":
        before[e.at] += e.text;
        break;
      case "repeat":
        before[e.to] += ref.slice(e.from, e.to);
        break;
      case "noise":
        before[e.at] += seededTokens(tokens, e.length, e.seed);
        break;
      case "gap":
        gapBefore[e.at] += e.frames;
        break;
    }
  }

  const phonemes: DecodedPhoneme[] = [];
  let frame = 0;
  const emit = (p: string) => {
    phonemes.push({ p, start: frame, end: frame + 1, conf: 0.99 });
    frame += FRAMES_PER_PHONEME;
  };
  for (let i = 0; i <= ref.length; i++) {
    frame += gapBefore[i];
    for (const p of before[i]) emit(p);
    if (i < ref.length && replaced[i] !== null) for (const p of replaced[i]!) emit(p);
  }

  const size = o.chunkSize ?? phonemes.length;
  const evidence: PhonemeEvidence[] = [];
  for (let i = 0; i < phonemes.length; i += size) {
    evidence.push({ inventory: index.inventoryId, frameMs: 20, phonemes: phonemes.slice(i, i + size) });
  }
  return {
    id: o.id,
    provenance: { kind: "synthetic", from: o.from, to: o.to, edits },
    evidence,
    expect: o.expect,
  };
}
