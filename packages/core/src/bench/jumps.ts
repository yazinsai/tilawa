/** Stitched long sessions (ticket 13): clean clips joined end to end, scored for false jumps, planted jumps and the start. */
import { createChecker } from "../checker.js";
import type { CheckerEvent, EngineEvent, PositionEvent, WordRef } from "../events.js";
import type { PhraseAsset, ReferenceIndex } from "../reference.js";
import { createTracker } from "../tracker.js";
import { chunk } from "./index.js";
import type { QudRecording } from "./index.js";

/** Posteriors are optional: a decode without them stitches to phonemes only. */
export type ClipEvidence = Pick<QudRecording, "inventory" | "frameMs" | "nFrames" | "phonemes"> & { posteriors?: QudRecording["posteriors"] };

export interface SessionPart {
  /** Source clip ID. */
  source: string;
  /** The ayahs the clip recites, as "surah:ayah", in order. */
  ayahs: string[];
  /** Silence or room tone before the clip. */
  gapBeforeMs: number;
  /** `room`: the gap repeats the clip's own lead-in before its first phoneme; `silence`: blank frames. */
  gapFill?: "silence" | "room";
  evidence: ClipEvidence;
}

export interface Segment {
  source: string;
  ayahs: string[];
  startMs: number;
  /** First decoded phoneme. */
  onsetMs: number;
  endMs: number;
  /** The first ayah is more than `LOCAL_AYAHS` from the previous segment's last: a deliberate jump. */
  planted: boolean;
}

export interface StitchedSession {
  evidence: ClipEvidence;
  segments: Segment[];
}

export interface FalseJump {
  timeMs: number;
  from: string;
  to: string;
  /** The ayahs being recited at the time. */
  truth: string[];
}

export interface SessionScore {
  audioS: number;
  firstCorrectLockMs: number | null;
  startWithin5s: boolean;
  /** Non-local lock changes. */
  jumps: number;
  falseJumps: number;
  falseJumpDetail: FalseJump[];
  planted: number;
  followed: number;
  /** Per planted jump, from the new passage's first phoneme to the first lock on it; null if never followed. */
  followLatencyMs: (number | null)[];
  /**
   * Wrong and added words a free-recitation checker confirmed: every one is false, the clips are clean. Omissions and
   * skips aren't counted: contiguous runs drop up to an ayah and clip spans can trim an edge word.
   */
  checkerWarnings: number | null;
}

/** Within this many ayahs a lock is local, as in `acceptance-v3`. */
const LOCAL_AYAHS = 2;
const WITHIN_MS = 5000;
/** A lock names what is being recited if it names an ayah from the last `LAG_MS`, or starting within `LEAD_MS`. Same as the QUD bench. */
const LAG_MS = 4000;
const LEAD_MS = 500;
const SILENCE = 0.99;

const ayahKey = (r: { surah: number; ayah: number }) => `${r.surah}:${r.ayah}`;
const ordinals = new WeakMap<ReferenceIndex, Map<string, number>>();
function ordinal(index: ReferenceIndex, ayah: string): number {
  let m = ordinals.get(index);
  if (!m) {
    m = new Map();
    for (const w of index.words) if (!m.has(ayahKey(w))) m.set(ayahKey(w), m.size);
    ordinals.set(index, m);
  }
  const o = m.get(ayah);
  if (o === undefined) throw new Error(`Unknown ayah ${ayah}`);
  return o;
}
const near = (index: ReferenceIndex, a: string, b: string) => Math.abs(ordinal(index, a) - ordinal(index, b)) <= LOCAL_AYAHS;

/** Joins clips into one recording, shifting phonemes and posteriors, with each part's gap in front of it. */
export function stitchSession(index: ReferenceIndex, parts: SessionPart[]): StitchedSession {
  const { inventory, frameMs } = parts[0].evidence;
  const withPosteriors = parts.every((p) => p.evidence.posteriors);
  const k = withPosteriors ? parts[0].evidence.posteriors!.k : 0;
  const ids: number[] = [];
  const probs: number[] = [];
  const phonemes: ClipEvidence["phonemes"] = [];
  const segments: Segment[] = [];
  let frames = 0;
  const silence = (n: number) => {
    for (let f = 0; f < n; f++) for (let j = 0; j < k; j++) {
      ids.push(j);
      probs.push(j ? (1 - SILENCE) / (k - 1) : SILENCE);
    }
  };
  for (const [i, p] of parts.entries()) {
    const e = p.evidence;
    if (e.inventory !== inventory || e.frameMs !== frameMs || (withPosteriors && e.posteriors!.k !== k)) throw new Error(`Incompatible evidence: ${p.source}`);
    const gap = Math.round(p.gapBeforeMs / frameMs);
    const lead = e.phonemes[0]?.start ?? 0;
    if (p.gapFill === "room" && lead >= 3) {
      for (let f = 0; f < gap; f++) {
        const at = (f % lead) * k;
        ids.push(...(e.posteriors?.ids.slice(at, at + k) ?? []));
        probs.push(...(e.posteriors?.probs.slice(at, at + k) ?? []));
      }
    } else silence(gap);
    const start = frames + gap;
    ids.push(...(e.posteriors?.ids.slice(0, e.nFrames * k) ?? []));
    probs.push(...(e.posteriors?.probs.slice(0, e.nFrames * k) ?? []));
    for (const ph of e.phonemes) phonemes.push({ ...ph, start: ph.start + start, end: ph.end + start });
    frames = start + e.nFrames;
    const prev = segments.at(-1);
    segments.push({
      source: p.source,
      ayahs: p.ayahs,
      startMs: start * frameMs,
      onsetMs: (start + lead) * frameMs,
      endMs: frames * frameMs,
      planted: i > 0 && !near(index, prev!.ayahs.at(-1)!, p.ayahs[0]),
    });
  }
  return { evidence: { inventory, frameMs, nFrames: frames, phonemes, ...(withPosteriors ? { posteriors: { k, ids, probs } } : {}) }, segments };
}

export function scoreSession(index: ReferenceIndex, s: StitchedSession, o: { phrases: PhraseAsset; chunkS?: number; posteriors?: boolean; checker?: boolean }): SessionScore {
  const tracker = createTracker(index, { mode: "memorization", phrases: o.phrases });
  const positions: PositionEvent[] = [];
  const keep = (out: EngineEvent[]) => positions.push(...out.filter((x): x is PositionEvent => x.type === "position"));
  const checker = o.checker ? createChecker(index, { phrases: o.phrases }) : null;
  const checks: CheckerEvent[] = [];
  for (const e of chunk(s.evidence as QudRecording, o.chunkS ?? 1, o.posteriors ?? false)) {
    const out = tracker.push(e);
    keep(out);
    if (checker) checks.push(...checker.push(e, out));
  }
  const out = tracker.end();
  keep(out);
  if (checker) checks.push(...checker.end(out));
  const latest = new Map(checks.map((e) => [e.id, e]));
  const locks = positions.filter((p) => p.state === "locked" && p.observed[0]).map((p) => ({ timeMs: p.timeMs, ayah: ayahKey(p.observed[0].word as WordRef) }));

  const segs = s.segments;
  const truth = (t: number) => {
    const live = segs.filter((g) => g.startMs - LEAD_MS <= t && t <= g.endMs + LAG_MS);
    const last = segs.filter((g) => g.startMs <= t).at(-1);
    return [...new Set([...live, ...(last ? [last] : [])].flatMap((g) => g.ayahs))];
  };
  // Runs: segments from one planted jump (or the start) to the next.
  const runStarts = segs.flatMap((g, i) => (i === 0 || g.planted ? [i] : []));
  const runs = runStarts.map((a, j) => {
    const b = runStarts[j + 1] ?? segs.length;
    return { first: segs[a], ayahs: new Set(segs.slice(a, b).flatMap((g) => g.ayahs)), untilMs: b < segs.length ? segs[b].startMs : Infinity };
  });
  const firstLockOn = (r: (typeof runs)[number]) => locks.find((l) => l.timeMs >= r.first.startMs && l.timeMs < r.untilMs && r.ayahs.has(l.ayah)) ?? null;

  let jumps = 0;
  const falseJumpDetail: FalseJump[] = [];
  for (let i = 1; i < locks.length; i++) {
    const [a, b] = [locks[i - 1], locks[i]];
    if (near(index, a.ayah, b.ayah)) continue;
    jumps++;
    const now = truth(b.timeMs);
    if (!now.some((x) => near(index, x, b.ayah))) falseJumpDetail.push({ timeMs: b.timeMs, from: a.ayah, to: b.ayah, truth: now });
  }
  const start = firstLockOn(runs[0]);
  const follows = runs.slice(1).map((r) => {
    const l = firstLockOn(r);
    return l ? l.timeMs - r.first.onsetMs : null;
  });
  return {
    audioS: (s.evidence.nFrames * s.evidence.frameMs) / 1000,
    firstCorrectLockMs: start?.timeMs ?? null,
    startWithin5s: !!start && start.timeMs - segs[0].onsetMs <= WITHIN_MS,
    jumps,
    falseJumps: falseJumpDetail.length,
    falseJumpDetail,
    planted: follows.length,
    followed: follows.filter((x) => x !== null).length,
    followLatencyMs: follows,
    checkerWarnings: !checker ? null : [...latest.values()].filter((e) => e.state === "confirmed" && (e.errorClass === "wrong_word" || e.errorClass === "added_word")).length,
  };
}
