import type { EngineEvent, PositionEvent, TrackerState, WordRef, WordSpan } from "../events.js";
import type { DecodedPhoneme, PhonemeEvidence } from "../evidence.js";
import type { ReferenceIndex } from "../reference.js";
import { createTracker } from "../tracker.js";
import type { TrackerOptions } from "../tracker.js";

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
}

export interface Fixture {
  id: string;
  provenance: SyntheticProvenance | TlogProvenance;
  evidence: PhonemeEvidence[];
  /** Asserted only for synthetic fixtures and audio-verified real clips. */
  expect?: { state: TrackerState; span?: WordSpan };
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

export function finalPosition(events: EngineEvent[]): PositionEvent | undefined {
  return events.filter((e): e is PositionEvent => e.type === "position").at(-1);
}

const fmt = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
export const formatSpan = (s: WordSpan) => `${fmt(s.start)}–${fmt(s.end)}`;

/** Human-readable mismatches between a fixture's expectation and its replayed events. */
export function diffExpectation(fixture: Fixture, events: EngineEvent[]): string[] {
  if (!fixture.expect) return [];
  const final = finalPosition(events);
  if (!final) return ["no position event emitted"];
  const problems: string[] = [];
  if (final.state !== fixture.expect.state) problems.push(`state: expected ${fixture.expect.state}, got ${final.state}`);
  if (fixture.expect.span) {
    const got = final.observed[0]?.span;
    const want = formatSpan(fixture.expect.span);
    if (!got) problems.push(`span: expected ${want}, got none`);
    else if (formatSpan(got) !== want) problems.push(`span: expected ${want}, got ${formatSpan(got)}`);
  }
  return problems;
}

const FRAMES_PER_PHONEME = 3;

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
