/**
 * Lab scorer (.scratch/lab, ticket 08): the overall score, the three component scores and attribution on spliced
 * free-recitation sessions. Every wrong call is scored on one ladder of arms that differ in one input each:
 *
 *   oracle        checker on oracle evidence (what was said, decoded perfectly) with oracle tracker events
 *   truePosition  checker on the listener's evidence with oracle tracker events
 *   engine        tracker and checker on the listener's evidence (the product, free recitation)
 *
 * A wrong call of the engine goes to the first arm that makes it: checker, then listener, then tracker.
 */
import { createChecker } from "../checker.js";
import { EVENT_SCHEMA_VERSION } from "../events.js";
import type { CheckerEvent, EngineEvent, PositionEvent, WordRef } from "../events.js";
import type { DecodedPhoneme } from "../evidence.js";
import type { PhraseAsset, ReferenceIndex } from "../reference.js";
import { createTracker, trackerInput } from "../tracker.js";
import { chunk } from "./index.js";
import type { QudRecording } from "./index.js";
import { ERROR_CLASSES, ayahOrdinal, correctPosition, evidenceFits, inOccurrence, scoreSessionEvents, validateSession } from "./sessions.js";
import type { BenchSession, ErrorClass, Interval, SessionEvent, SessionEvidence } from "./sessions.js";

export const LAB_SCORE_VERSION = 1;
export const VARIANTS = ["phone", "studio"] as const;
export type Variant = (typeof VARIANTS)[number];
export type Split = "practice" | "exam";
export type Component = "listener" | "tracker" | "checker";
export const COMPONENTS: Component[] = ["listener", "tracker", "checker"];

/** Recited time bridges pauses up to this long between Quran audio (truth words and labelled mistakes). */
export const PAUSE_S = 2;
/** Tracker sampling step. */
export const SAMPLE_S = 0.1;
/** Exact-phoneme disambiguation can be one word early (2:255 vs 3:2), so the share counts from one word later. */
export const TOLERANCE_WORDS = 1;
/** Start points needing more further words than this are the strict slice when the manifest doesn't say. */
export const STRICT_WORDS = 5;
/** A false warning is the same call in another arm at the same word within this many seconds. */
export const MATCH_S = 6;
/** The realuser slice (real users, no planted mistakes) splits false warnings in a session's first seconds from the rest. */
export const EARLY_S = 10;
/** Window around a mistake over which the drill-down reports the tracker's on-place share. */
const AROUND_S = 1;
const CHUNK_S = 0.04;
const FRAME_MS = 20;
const NONE = { kind: "none" } as const;

export const DEFINITIONS = {
  recitedMinutes: `Recited time: the union of truth word and labelled-mistake intervals, bridging pauses up to ${PAUSE_S} s, minus non-Quran spans. One denominator for misses, false warnings, the checker score and tracker sampling.`,
  wrongCall: "A labelled mistake not confirmed by a matching warning (miss), or a confirmed warning matching no labelled mistake (false warning, including warnings on repeats and restarts). A warning on the mistake's word and occurrence with another error class still catches it, and is counted as misclassified. Free recitation, no context.",
  listener: "PER of the greedy decode against what was said (truth words with planted wrong and added words applied; non-Quran audio left out), one global alignment per session.",
  tracker: `Engine run on the listener's evidence. Share of recited time with a locked top candidate within one ayah of a word recited in the last 2 s, counted per entry (session start and every jump of more than one ayah) from ${TOLERANCE_WORDS} word after its disambiguation point. False jump: a run of locks on a wrong place, before or after the point. Lock delay: first on-place sample at or after the point.`,
  checker: "Wrong calls per 5 min of the checker on oracle evidence with oracle tracker events (locked on the true word).",
  teacher: "Teacher reference: the engine run on the teacher's evidence instead of the listener's, same scoring. A reference for listener quality through the engine, not a bound: the engine is tuned on the listener.",
  attribution: "Each engine wrong call goes to checker if the oracle arm makes it too, else to listener if the true-position arm (listener evidence, oracle tracker events) makes it, else to tracker. False warnings match across arms by word and time.",
} as const;

const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const round = (x: number, d = 3) => Math.round(x * 10 ** d) / 10 ** d;
const wordIndexes = new WeakMap<ReferenceIndex, Map<string, number>>();
export function refIndex(index: ReferenceIndex): (r: WordRef) => number {
  let m = wordIndexes.get(index);
  if (!m) { m = new Map(index.words.map((w, i) => [key(w), i])); wordIndexes.set(index, m); }
  const map = m;
  return (r: WordRef) => { const i = map.get(key(r)); if (i === undefined) throw new Error(`unknown word ${key(r)}`); return i; };
}

/** Recited time (see `DEFINITIONS.recitedMinutes`), sorted and disjoint. */
export function recitedIntervals(s: BenchSession): Interval[] {
  const items = [...s.words, ...s.events].map(x => ({ startS: x.startS, endS: Math.max(x.startS, x.endS) })).sort((a, b) => a.startS - b.startS);
  let out: Interval[] = [];
  for (const x of items) {
    const last = out.at(-1);
    if (last && x.startS - last.endS <= PAUSE_S) last.endS = Math.max(last.endS, x.endS);
    else out.push({ ...x });
  }
  for (const n of s.spans) if (n.kind === "non_quran") out = out.flatMap(iv => n.endS <= iv.startS || n.startS >= iv.endS ? [iv]
    : [...(n.startS > iv.startS ? [{ startS: iv.startS, endS: n.startS }] : []), ...(n.endS < iv.endS ? [{ startS: n.endS, endS: iv.endS }] : [])]);
  return out.filter(iv => iv.endS > iv.startS);
}
export const recitedSeconds = (s: BenchSession) => recitedIntervals(s).reduce((n, iv) => n + iv.endS - iv.startS, 0);
/** Sample times in recited time, every `SAMPLE_S`. */
function samples(s: BenchSession): number[] {
  const out: number[] = [];
  for (const iv of recitedIntervals(s)) for (let j = Math.ceil(iv.startS / SAMPLE_S - 1e-9); j * SAMPLE_S < iv.endS; j++) out.push(round(j * SAMPLE_S, 6));
  return out;
}

export interface SaidItem extends Interval { phonemes: string; at: WordRef | null; planted: boolean }
/** What was said, in time order: truth words (a planted wrong word's `said` in its slot) and planted added words. */
export function saidItems(s: BenchSession, index: ReferenceIndex): SaidItem[] {
  const ix = refIndex(index);
  const wrong = new Map<string, SessionEvent>();
  for (const e of s.events) {
    if (e.errorClass !== "wrong_word" && e.errorClass !== "added_word") continue;
    if (!e.said) throw new Error(`${s.id}: planted ${e.errorClass} at ${key(e.location)} lacks said phonemes`);
    if (e.errorClass === "wrong_word") wrong.set(`${e.span}|${key(e.location)}`, e);
  }
  const used = new Set<SessionEvent>();
  const items: SaidItem[] = s.words.map(w => {
    const e = wrong.get(`${w.span}|${key(w.at)}`);
    if (e) used.add(e);
    const r = index.words[ix(w.at)];
    return { startS: w.startS, endS: w.endS, at: w.at, planted: !!e, phonemes: e ? e.said! : index.phonemes.slice(r.start, r.end) };
  });
  for (const e of wrong.values()) if (!used.has(e)) throw new Error(`${s.id}: wrong_word at ${key(e.location)} has no truth word in its span`);
  for (const e of s.events) if (e.errorClass === "added_word") items.push({ startS: e.startS, endS: e.endS, at: null, planted: true, phonemes: e.said! });
  return items.sort((a, b) => a.startS - b.startS || (a.at ? 1 : 0) - (b.at ? 1 : 0));
}

/** Perfect evidence of what was said on the session clock: one confident spike per phoneme spread over its word, blank elsewhere. */
export function oracleEvidence(s: BenchSession, index: ReferenceIndex): SessionEvidence {
  const nFrames = Math.round(s.durationS * 1000 / FRAME_MS);
  const phonemes: DecodedPhoneme[] = [];
  let next = 0;
  for (const it of saidItems(s, index)) {
    const n = it.phonemes.length, f0 = Math.floor(it.startS * 1000 / FRAME_MS);
    const width = Math.max(1, Math.ceil(it.endS * 1000 / FRAME_MS) - f0);
    for (let j = 0; j < n; j++) {
      const f = Math.max(next, f0 + Math.floor(j * width / n));
      if (f >= nFrames) throw new Error(`${s.id}: oracle evidence overruns the audio clock`);
      phonemes.push({ p: it.phonemes[j], start: f, end: f, conf: 0.99 });
      next = f + 1;
    }
  }
  const K = 5, ids = new Array<number>(nFrames * K), probs = new Array<number>(nFrames * K).fill(0);
  const id = new Map([...index.tokens].map((t, i) => [t, i + 1]));
  for (let f = 0; f < nFrames; f++) for (let k = 0; k < K; k++) { ids[f * K + k] = k; }
  for (let f = 0; f < nFrames; f++) probs[f * K] = 0.95;
  for (const p of phonemes) {
    const t = id.get(p.p)!, row = p.start * K;
    const fill = [t, 0, ...[1, 2, 3, 4, 5].filter(x => x !== t)].slice(0, K);
    fill.forEach((x, k) => { ids[row + k] = x; probs[row + k] = k === 0 ? 0.9 : k === 1 ? 0.05 : 0; });
  }
  return { inventory: index.inventoryId, frameMs: FRAME_MS, nFrames, phonemes, posteriors: { k: K, ids, probs } };
}

/** Unit-cost edit distance, two rows. */
export function editDistance(a: string, b: string): number {
  let prev = Int32Array.from({ length: b.length + 1 }, (_, j) => j), row = new Int32Array(b.length + 1);
  for (let i = 1; i <= a.length; i++) {
    row[0] = i;
    const c = a.charCodeAt(i - 1);
    for (let j = 1; j <= b.length; j++) row[j] = Math.min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (c === b.charCodeAt(j - 1) ? 0 : 1));
    [prev, row] = [row, prev];
  }
  return prev[b.length];
}
/** The decode against what was said; decoded phonemes inside non-Quran spans are left out. */
export function listenerErrors(s: BenchSession, evidence: SessionEvidence, index: ReferenceIndex): { edits: number; phonemes: number } {
  const ref = saidItems(s, index).map(x => x.phonemes).join("");
  const noise = s.spans.filter(p => p.kind === "non_quran");
  const obs = evidence.phonemes.filter(p => { const t = p.start * evidence.frameMs / 1000; return !noise.some(n => n.startS <= t && t < n.endS); }).map(p => p.p).join("");
  return { edits: editDistance(ref, obs), phonemes: ref.length };
}

export interface TrackerEntry { startS: number; endS: number; word: WordRef; need: number; pointS: number | null; scoredFromS: number | null }
/**
 * Where the reciter (re-)enters the text: the first truth word and every word more than one ayah from the previous one
 * (a skipped passage, a restart further back). The disambiguation point is the end of the `need`-th further word of the
 * entry; null when the entry ends first or the text never becomes unique.
 */
export function trackerEntries(s: BenchSession, index: ReferenceIndex, disamb: ArrayLike<number>): TrackerEntry[] {
  const ay = ayahOrdinal(index), ix = refIndex(index);
  const starts = s.words.flatMap((w, i) => i === 0 || Math.abs(ay(w.at) - ay(s.words[i - 1].at)) > 1 ? [i] : []);
  return starts.map((k, n) => {
    const end = starts[n + 1] ?? s.words.length, need = disamb[ix(s.words[k].at)];
    const reached = need >= 0 && k + need < end;
    return { startS: s.words[k].startS, endS: n + 1 < starts.length ? s.words[end].startS : s.durationS, word: s.words[k].at, need,
      pointS: reached ? s.words[k + need].endS : null, scoredFromS: reached ? s.words[Math.min(end - 1, k + need + TOLERANCE_WORDS)].endS : null };
  });
}

const positionsOf = (events: EngineEvent[]) => events.filter((e): e is PositionEvent => e.type === "position").sort((a, b) => a.timeMs - b.timeMs);
/** The position shown at time `t`: the latest event at or before it. */
function shown(positions: PositionEvent[]): (t: number) => PositionEvent | undefined {
  return (t: number) => {
    let lo = 0, hi = positions.length;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (positions[mid].timeMs <= t * 1000 + 1e-6) lo = mid + 1; else hi = mid; }
    return positions[lo - 1];
  };
}

export interface TrackerScore {
  entries: number; reentries: number;
  /** Recited seconds counted (from each entry's point plus tolerance), and of those on the right place. */
  scoredS: number; onPlaceS: number;
  /** Recited seconds before the point, or in entries that never reach one. */
  unscoredS: number;
  falseJumps: number; falseJumpsBeforePoint: number;
  /** Per entry reaching its point: seconds from the point to the first on-place sample in the entry; censored when none. */
  delaysS: number[]; censored: number;
}
export function scoreTracker(s: BenchSession, events: EngineEvent[], index: ReferenceIndex, disamb: ArrayLike<number>): TrackerScore {
  const ay = ayahOrdinal(index), positions = positionsOf(events), at = shown(positions), entries = trackerEntries(s, index, disamb);
  const entryAt = (t: number) => { let e = entries[0]; for (const x of entries) if (x.startS <= t) e = x; return e; };
  const recited = recitedIntervals(s), inRecited = (t: number) => recited.some(iv => iv.startS <= t && t < iv.endS);
  let scored = 0, onPlace = 0, unscored = 0;
  const delay = new Map<TrackerEntry, number>();
  for (const t of samples(s)) {
    const e = entryAt(t), p = at(t), ok = p?.state === "locked" && correctPosition(s, p, t, ay);
    if (e.scoredFromS !== null && t >= e.scoredFromS) { scored++; if (ok) onPlace++; } else unscored++;
    if (ok && e.pointS !== null && t >= e.pointS && !delay.has(e)) delay.set(e, round(t - e.pointS));
  }
  let falseJumps = 0, beforePoint = 0, wrongAyah: number | null = null;
  for (const p of positions) {
    const t = p.timeMs / 1000;
    if (p.state !== "locked" || !p.observed[0] || !inRecited(t)) continue;
    if (correctPosition(s, p, t, ay)) { wrongAyah = null; continue; }
    const a = ay(p.observed[0].word);
    if (wrongAyah === null || Math.abs(a - wrongAyah) > 1) {
      falseJumps++;
      const e = entryAt(t);
      if (e.pointS === null || t < e.pointS) beforePoint++;
    }
    wrongAyah = a;
  }
  const reached = entries.filter(e => e.pointS !== null);
  return { entries: entries.length, reentries: entries.length - 1, scoredS: round(scored * SAMPLE_S), onPlaceS: round(onPlace * SAMPLE_S),
    unscoredS: round(unscored * SAMPLE_S), falseJumps, falseJumpsBeforePoint: beforePoint,
    delaysS: reached.flatMap(e => delay.has(e) ? [delay.get(e)!] : []), censored: reached.filter(e => !delay.has(e)).length };
}

/** The checker's true position: locked on the truth word being recited, a new run on every backward move or jump of more than one ayah. */
export function oraclePositions(s: BenchSession, index: ReferenceIndex) {
  const ay = ayahOrdinal(index), ix = refIndex(index);
  const runStart: WordRef[] = [];
  s.words.forEach((w, i) => {
    const prev = s.words[i - 1];
    runStart.push(!prev || ix(w.at) <= ix(prev.at) || Math.abs(ay(w.at) - ay(prev.at)) > 1 ? w.at : runStart[i - 1]);
  });
  let last = -1;
  return (timeMs: number): PositionEvent[] => {
    let i = last;
    while (i + 1 < s.words.length && s.words[i + 1].startS * 1000 < timeMs) i++;
    if (i === last) return [];
    last = i;
    const w = s.words[i].at;
    return [{ type: "position", v: EVENT_SCHEMA_VERSION, timeMs, referenceVersion: index.version, expected: w, state: "locked",
      observed: [{ word: w, span: { start: runStart[i], end: w }, confidence: 1, cost: 0, per: 0 }] }];
  };
}

function checkEvidence(s: BenchSession, e: SessionEvidence, index: ReferenceIndex, what: string) {
  if (!evidenceFits(s, e, index)) throw new Error(`Evidence mismatch: ${s.id} ${what}`);
}
export const chunks = (e: SessionEvidence) => chunk(e as QudRecording, CHUNK_S, !!e.posteriors?.ids?.length);
/** The product: free recitation, no context, tracker and checker on `evidence`. */
export function runEngine(s: BenchSession, evidence: SessionEvidence, index: ReferenceIndex, phrases?: PhraseAsset): EngineEvent[] {
  checkEvidence(s, evidence, index, "engine");
  const tracker = createTracker(index, { mode: "free", phrases }), checker = createChecker(index, { phrases }), out: EngineEvent[] = [];
  for (const e of chunks(evidence)) { const p = tracker.push(trackerInput(e)); out.push(...p, ...checker.push(e, p)); }
  const end = tracker.end(); out.push(...end, ...checker.end(end));
  return out;
}
/** The checker on `evidence` with oracle tracker events. */
export function runTruePosition(s: BenchSession, evidence: SessionEvidence, index: ReferenceIndex, phrases?: PhraseAsset): EngineEvent[] {
  checkEvidence(s, evidence, index, "true position");
  const checker = createChecker(index, { phrases }), oracle = oraclePositions(s, index), out: EngineEvent[] = [];
  for (const e of chunks(evidence)) { const p = oracle((e.decodedTo ?? 0) * e.frameMs); out.push(...p, ...checker.push(e, p)); }
  const end = oracle(Infinity); out.push(...end, ...checker.end(end));
  return out;
}
/** The tracker alone on `evidence` (the tracker on oracle evidence is a diagnostic). */
export function runTracker(s: BenchSession, evidence: SessionEvidence, index: ReferenceIndex, phrases?: PhraseAsset): EngineEvent[] {
  checkEvidence(s, evidence, index, "tracker");
  const tracker = createTracker(index, { mode: "free", phrases }), out: EngineEvent[] = [];
  for (const e of chunks(evidence)) out.push(...tracker.push(trackerInput(e)));
  return [...out, ...tracker.end()];
}

interface Warned { id: string; errorClass: string; at: WordRef; timeS: number; confirmedS: number; expected: string | null; observed: string | null }
interface ArmCalls { hit: Map<number, number>; misclassified: Set<number>; falseWarnings: Warned[] }
/** Hits (event index -> confirmation time) and false warnings of one arm, by the session scorer's matching rules. */
function armCalls(s: BenchSession, events: EngineEvent[], index: ReferenceIndex): ArmCalls {
  const score = scoreSessionEvents(s, events, index, NONE, { anyClass: true });
  const states = new Map<string, CheckerEvent[]>();
  for (const e of events) if (e.type === "checker") states.set(e.id, [...(states.get(e.id) ?? []), e]);
  const hit = new Map<number, number>(), misclassified = new Set<number>(), falseWarnings: Warned[] = [];
  for (const a of score.assignments) {
    if (!a.confirmed) continue;
    const list = [...states.get(a.warningId)!].sort((x, y) => x.timeMs - y.timeMs), confirmed = list.find(e => e.state === "confirmed")!;
    const confirmedS = confirmed.timeMs / 1000;
    if (a.eventIndex !== null) { hit.set(a.eventIndex, confirmedS); if (a.misclassified) misclassified.add(a.eventIndex); }
    else falseWarnings.push({ id: a.warningId, errorClass: list[0].errorClass, at: list[0].location, timeS: list[0].timeMs / 1000, confirmedS,
      expected: confirmed.expected ?? null, observed: confirmed.observed ?? null });
  }
  return { hit, misclassified, falseWarnings };
}
/** One-to-one matching of false warnings into another arm's: same word, nearest within `MATCH_S`. */
function matcher(pool: Warned[]): (w: Warned) => boolean {
  const used = new Set<number>();
  return (w: Warned) => {
    let best = -1;
    pool.forEach((x, i) => {
      if (used.has(i) || key(x.at) !== key(w.at) || Math.abs(x.timeS - w.timeS) > MATCH_S) return;
      if (best < 0 || Math.abs(x.timeS - w.timeS) < Math.abs(pool[best].timeS - w.timeS)) best = i;
    });
    if (best >= 0) used.add(best);
    return best >= 0;
  };
}

interface CaseBase { key: string; session: string; variant: Variant; split: Split; strict: boolean; errorClass: string; word: string;
  /** Component the wrong call is attributed to; null for a caught mistake. */
  component: Component | null;
  /** Share of recited samples within 1 s of the case where the engine's tracker was locked on the right place. */
  trackerOnPlace: number | null;
  /** The engine, the oracle arm and (when decoded) the teacher all make this wrong call: likely bad data first. */
  allFail: boolean;
  /** Listening window in the review queue's clip shape; `path` is relative to /lab/data when the manifest names the audio. */
  review: Interval & { path: string | null };
  /** Set when the test set's automated checks (ticket 04) failed the data under this case: review it as likely bad data. */
  dataCheck?: { pass: false; check: string; reasons: string[] } }
export interface MistakeCase extends CaseBase {
  /** `misclassified`: caught by a warning of another error class on the same word (or, for an added word, the word before it). */
  kind: "mistake"; startS: number; endS: number; outcome: "hit" | "misclassified" | "miss"; confirmedS: number | null; mutashabihat: boolean;
  oracle: "hit" | "miss"; truePosition: "hit" | "miss"; teacher: "hit" | "miss" | null }
export interface FalseWarningCase extends CaseBase {
  kind: "false_warning"; timeS: number; confirmedS: number; nonError: "repeat" | "restart" | "self_correction" | null;
  /** The checker's phonemes when it confirmed: what it expected at the word and what it heard instead (null when it gave none). */
  expected: string | null; observed: string | null;
  oracle: boolean; truePosition: boolean; teacher: boolean | null }
export type LabCase = MistakeCase | FalseWarningCase;
export const caseKey = (session: string, variant: Variant, type: string, word: WordRef, timeS: number) => `${session}/${variant}/${type}/${key(word)}/${timeS.toFixed(2)}`;

export type ClassCounts = { mistakes: number; missed: number; misclassified: number } & Record<Component, number>;
const zeroClass = (): ClassCounts => ({ mistakes: 0, missed: 0, misclassified: 0, listener: 0, tracker: 0, checker: 0 });
export interface VariantResult {
  mistakes: number; missed: number; misclassified: number; falseWarnings: number; nonErrorFalseWarnings: number;
  /** False warnings raised in the session's first `EARLY_S` seconds. */
  earlyFalseWarnings: number;
  attribution: Record<Component, { missed: number; falseWarnings: number }>;
  listener: { edits: number; phonemes: number };
  tracker: TrackerScore;
  byClass: Record<ErrorClass, ClassCounts>; mutashabihat: ClassCounts; falseWarningsByClass: Record<string, number>;
  teacher: { missed: number; falseWarnings: number; edits: number; phonemes: number } | null;
}
export interface SessionResult {
  id: string; split: Split; strict: boolean; speaker: string; durationS: number; recitedS: number;
  /** The manifest's `slices.source` ("recorded": a scripted-error take, tools/lab_recordings_core.py); unset for built sessions. */
  source?: string;
  /** The manifest's `slices.origin`: where a realuser session's audio came from (sobolev, retasy, tlog). */
  origin?: string;
  /** Recited seconds within the first `EARLY_S` seconds of the session. */
  earlyRecitedS: number;
  /** Further words the first word needs to become unique (-1: never). */
  startNeed: number;
  checker: { missed: number; misclassified: number; falseWarnings: number; nonErrorFalseWarnings: number };
  trackerOracle: TrackerScore | null;
  variants: Partial<Record<Variant, VariantResult>>;
  cases: LabCase[];
}
export interface VariantArms { evidence: SessionEvidence; engine: EngineEvent[]; truePosition: EngineEvent[]; teacher?: { evidence: SessionEvidence; engine: EngineEvent[] }; audio?: string }
export interface LabArms { oracle: EngineEvent[]; trackerOracle: EngineEvent[] | null; variants: Partial<Record<Variant, VariantArms>> }
export interface LabContext { index: ReferenceIndex; disamb: ArrayLike<number>; phrases?: PhraseAsset }

const nonErrorAt = (s: BenchSession, at: WordRef, t: number) =>
  s.nonErrors.find(n => key(n.at) === key(at) && inOccurrence(s, n, n.endS + 6, t))?.kind ?? null;

/**
 * Scores one session from its arms' events. Pure: every wrong call of every variant lands in exactly one component.
 * Mistakes overlapping an `exclude` window (flagged bad data) and false warnings raised inside one are not counted.
 */
export function scoreLabSession(s: BenchSession, split: Split, strictTag: boolean | undefined, arms: LabArms, ctx: LabContext, exclude: Interval[] = []): SessionResult {
  validateSession(s, ctx.index);
  const ay = ayahOrdinal(ctx.index), ix = refIndex(ctx.index);
  const startNeed = ctx.disamb[ix(s.words[0].at)];
  const strict = strictTag ?? (startNeed < 0 || startNeed > STRICT_WORDS);
  const out = (a: number, b = a) => exclude.some(w => a <= w.endS && b >= w.startS);
  const kept = s.events.map(e => !out(e.startS, e.endS)), nKept = kept.filter(Boolean).length;
  const keep = (a: ArmCalls): ArmCalls => ({ hit: new Map([...a.hit].filter(([i]) => kept[i])), misclassified: new Set([...a.misclassified].filter(i => kept[i])),
    falseWarnings: a.falseWarnings.filter(w => !out(w.timeS)) });
  const calls = (events: EngineEvent[]) => keep(armCalls(s, events, ctx.index));
  const oracle = calls(arms.oracle);
  const recited = recitedIntervals(s), all = samples(s);
  const result: SessionResult = { id: s.id, split, strict, speaker: s.speaker, durationS: s.durationS, recitedS: round(recited.reduce((n, iv) => n + iv.endS - iv.startS, 0)),
    earlyRecitedS: round(recited.reduce((n, iv) => n + Math.max(0, Math.min(iv.endS, EARLY_S) - iv.startS), 0)), startNeed, checker: { missed: nKept - oracle.hit.size, misclassified: oracle.misclassified.size, falseWarnings: oracle.falseWarnings.length,
      nonErrorFalseWarnings: oracle.falseWarnings.filter(w => nonErrorAt(s, w.at, w.timeS)).length },
    trackerOracle: arms.trackerOracle ? scoreTracker(s, arms.trackerOracle, ctx.index, ctx.disamb) : null, variants: {}, cases: [] };
  for (const variant of VARIANTS) {
    const v = arms.variants[variant];
    if (!v) continue;
    const engine = calls(v.engine), truePos = calls(v.truePosition);
    const teacher = v.teacher ? calls(v.teacher.engine) : null;
    const at = shown(positionsOf(v.engine));
    const onPlace = (a: number, b: number) => {
      const ts = all.filter(t => t >= a && t <= b);
      return ts.length ? round(ts.filter(t => { const p = at(t); return p?.state === "locked" && correctPosition(s, p, t, ay); }).length / ts.length) : null;
    };
    const r: VariantResult = { mistakes: nKept, missed: 0, misclassified: engine.misclassified.size, falseWarnings: engine.falseWarnings.length, nonErrorFalseWarnings: 0,
      earlyFalseWarnings: engine.falseWarnings.filter(w => w.timeS < EARLY_S).length,
      attribution: { listener: { missed: 0, falseWarnings: 0 }, tracker: { missed: 0, falseWarnings: 0 }, checker: { missed: 0, falseWarnings: 0 } },
      listener: listenerErrors(s, v.evidence, ctx.index), tracker: scoreTracker(s, v.engine, ctx.index, ctx.disamb),
      byClass: Object.fromEntries(ERROR_CLASSES.map(c => [c, zeroClass()])) as VariantResult["byClass"], mutashabihat: zeroClass(), falseWarningsByClass: {},
      teacher: v.teacher && teacher ? { missed: nKept - teacher.hit.size, falseWarnings: teacher.falseWarnings.length, ...listenerErrors(s, v.teacher.evidence, ctx.index) } : null };
    s.events.forEach((e, i) => {
      if (!kept[i]) return;
      const hit = engine.hit.has(i);
      const component: Component | null = hit ? null : !oracle.hit.has(i) ? "checker" : !truePos.hit.has(i) ? "listener" : "tracker";
      const slices = [r.byClass[e.errorClass as ErrorClass], ...(e.forms_other_ayah ? [r.mutashabihat] : [])].filter(Boolean);
      const misclassified = engine.misclassified.has(i);
      for (const c of slices) { c.mistakes++; if (misclassified) c.misclassified++; if (component) { c.missed++; c[component]++; } }
      if (component) { r.missed++; r.attribution[component].missed++; }
      result.cases.push({ key: caseKey(s.id, variant, `mistake:${e.errorClass}`, e.location, e.startS), kind: "mistake", session: s.id, variant, split, strict,
        errorClass: e.errorClass, word: key(e.location), startS: e.startS, endS: e.endS, outcome: misclassified ? "misclassified" : hit ? "hit" : "miss", component,
        confirmedS: engine.hit.get(i) ?? null, mutashabihat: !!e.forms_other_ayah, oracle: oracle.hit.has(i) ? "hit" : "miss",
        truePosition: truePos.hit.has(i) ? "hit" : "miss", teacher: teacher ? (teacher.hit.has(i) ? "hit" : "miss") : null,
        trackerOnPlace: onPlace(e.startS - AROUND_S, e.endS + AROUND_S), allFail: !hit && !oracle.hit.has(i) && (!teacher || !teacher.hit.has(i)),
        review: { path: v.audio ?? null, startS: round(Math.max(0, e.startS - 3), 2), endS: round(Math.min(s.durationS, e.endS + 3), 2) } });
    });
    const inOracle = matcher(oracle.falseWarnings), inTruePos = matcher(truePos.falseWarnings), inTeacher = teacher ? matcher(teacher.falseWarnings) : null;
    for (const w of engine.falseWarnings) {
      const o = inOracle(w), t = inTruePos(w), te = inTeacher ? inTeacher(w) : null;
      const component: Component = o ? "checker" : t ? "listener" : "tracker", nonError = nonErrorAt(s, w.at, w.timeS);
      r.attribution[component].falseWarnings++;
      if (nonError) r.nonErrorFalseWarnings++;
      r.falseWarningsByClass[w.errorClass] = (r.falseWarningsByClass[w.errorClass] ?? 0) + 1;
      result.cases.push({ key: caseKey(s.id, variant, `warning:${w.errorClass}`, w.at, w.timeS), kind: "false_warning", session: s.id, variant, split, strict,
        errorClass: w.errorClass, word: key(w.at), timeS: w.timeS, confirmedS: w.confirmedS, component, nonError, expected: w.expected, observed: w.observed,
        oracle: o, truePosition: t, teacher: te,
        trackerOnPlace: onPlace(w.timeS - AROUND_S, w.timeS + AROUND_S), allFail: o && te !== false,
        review: { path: v.audio ?? null, startS: round(Math.max(0, w.timeS - 4), 2), endS: round(Math.min(s.durationS, w.timeS + 2), 2) } });
    }
    result.variants[variant] = r;
  }
  return result;
}

/** Failed automated checks of a test set (`checks-v1.json`: `failed` planted edits, `failedClean` stretches). */
export interface DataChecks {
  failed: { id: string; session: string; type: string; location: WordRef; startS: number; endS: number; reasons: string[] }[];
  failedClean: { id: string; session: string; startS: number; endS: number; reasons: string[] }[];
}
/** Seconds around a failed planted edit or clean stretch within which a false warning is marked too. */
const DATA_CHECK_S = 1;
/** Marks the cases whose data failed an automated check: the planted mistake itself, and false warnings on or near failed data. Counts are unchanged. */
export function markDataChecks(r: SessionResult, checks: DataChecks): SessionResult {
  const edits = checks.failed.filter(f => f.session === r.id), clean = checks.failedClean.filter(f => f.session === r.id);
  for (const c of r.cases) {
    const near = (a: number, b: number) => c.kind === "false_warning" && c.timeS >= a - DATA_CHECK_S && c.timeS <= b + DATA_CHECK_S;
    const f = c.kind === "mistake"
      ? edits.find(f => f.type === c.errorClass && key(f.location) === c.word && Math.abs(f.startS - c.startS) < 0.05)
      : edits.find(f => near(f.startS, f.endS)) ?? clean.find(f => near(f.startS, f.endS));
    if (f) c.dataCheck = { pass: false, check: f.id, reasons: f.reasons };
  }
  return r;
}

export interface LabInput { session: BenchSession; split: Split; strict?: boolean; source?: string; origin?: string;
  evidence: Partial<Record<Variant, SessionEvidence>>; teacher?: Partial<Record<Variant, SessionEvidence>>; audio?: Partial<Record<Variant, string>>; trackerOracle?: boolean;
  exclude?: Interval[] }
/** Runs every arm of one session and scores it. */
export function labSession(input: LabInput, ctx: LabContext): SessionResult {
  const s = input.session;
  validateSession(s, ctx.index);
  const oracle = oracleEvidence(s, ctx.index);
  const variants: LabArms["variants"] = {};
  for (const v of VARIANTS) {
    const evidence = input.evidence[v];
    if (!evidence) continue;
    const t = input.teacher?.[v];
    variants[v] = { evidence, engine: runEngine(s, evidence, ctx.index, ctx.phrases), truePosition: runTruePosition(s, evidence, ctx.index, ctx.phrases),
      ...(t ? { teacher: { evidence: t, engine: runEngine(s, t, ctx.index, ctx.phrases) } } : {}), ...(input.audio?.[v] ? { audio: input.audio[v] } : {}) };
  }
  if (!Object.keys(variants).length) throw new Error(`${s.id}: no listener evidence`);
  const r = scoreLabSession(s, input.split, input.strict, { oracle: runTruePosition(s, oracle, ctx.index, ctx.phrases),
    trackerOracle: input.trackerOracle === false ? null : runTracker(s, oracle, ctx.index, ctx.phrases), variants }, ctx, input.exclude);
  return { ...r, ...(input.source ? { source: input.source } : {}), ...(input.origin ? { origin: input.origin } : {}) };
}

/**
 * A test set version's exclusion list (written by `tools/lab.py apply-flags` from bad-data review verdicts). An entry
 * with a window drops the mistakes and false warnings in it; an entry without one drops the whole session.
 */
export interface LabExclusions { version: 1; kind: "lab-flags"; testSet: string; base: string;
  excluded: ({ session: string; item?: string; verdict?: string; note?: string } & Partial<Interval>)[] }
export function exclusionsFor(x: LabExclusions | null, session: string): { session: boolean; windows: Interval[] } {
  const es = x?.excluded.filter(e => e.session === session) ?? [];
  return { session: es.some(e => e.startS === undefined || e.endS === undefined),
    windows: es.flatMap(e => e.startS !== undefined && e.endS !== undefined ? [{ startS: e.startS, endS: e.endS }] : []) };
}
/** The manifest entries one shard scores: those of `split` (all when unset), then every `shards`-th from `shard`. */
export function shardEntries<T extends { id: string; split: string }>(entries: T[], o: { split?: Split; only?: string; shard: number; shards: number }): T[] {
  return entries.filter(e => (!o.split || e.split === o.split) && (!o.only || e.id === o.only)).filter((_, i) => i % o.shards === o.shard);
}

export interface DelaySummary { n: number; p50: number | null; p90: number | null; max: number | null; censored: number; buckets: Record<string, number> }
const percentile = (xs: number[], q: number) => xs.length ? [...xs].sort((a, b) => a - b)[Math.ceil(q * xs.length) - 1] : null;
export function delaySummary(xs: number[], censored: number): DelaySummary {
  const buckets: Record<string, number> = { "0-1s": 0, "1-3s": 0, "3-5s": 0, "5-10s": 0, ">10s": 0, never: censored };
  for (const x of xs) buckets[x <= 1 ? "0-1s" : x <= 3 ? "1-3s" : x <= 5 ? "3-5s" : x <= 10 ? "5-10s" : ">10s"]++;
  return { n: xs.length, p50: percentile(xs, 0.5), p90: percentile(xs, 0.9), max: xs.length ? Math.max(...xs) : null, censored, buckets };
}
export interface TrackerTotals { onPlaceShare: number | null; onPlaceS: number; scoredS: number; unscoredS: number; falseJumps: number;
  falseJumpsBeforePoint: number; entries: number; reentries: number; lockDelayS: DelaySummary }
function trackerTotals(ts: TrackerScore[]): TrackerTotals {
  const sum = (f: (t: TrackerScore) => number) => ts.reduce((n, t) => n + f(t), 0);
  const scoredS = round(sum(t => t.scoredS)), onPlaceS = round(sum(t => t.onPlaceS));
  return { onPlaceShare: scoredS ? round(onPlaceS / scoredS, 4) : null, onPlaceS, scoredS, unscoredS: round(sum(t => t.unscoredS)),
    falseJumps: sum(t => t.falseJumps), falseJumpsBeforePoint: sum(t => t.falseJumpsBeforePoint), entries: sum(t => t.entries),
    reentries: sum(t => t.reentries), lockDelayS: delaySummary(ts.flatMap(t => t.delaysS), sum(t => t.censored)) };
}
const per5 = (n: number, seconds: number) => seconds ? round(n * 300 / seconds) : null;
export interface Totals {
  sessions: number; recitedMinutes: number;
  /** `misclassified` is part of `caught`. */
  overall: { wrongCalls: number; missed: number; falseWarnings: number; per5Min: number | null; mistakes: number; caught: number; misclassified: number; falseWarningsOnNonErrors: number };
  components: {
    listener: { per: number | null; edits: number; phonemes: number };
    tracker: TrackerTotals & { oracleEvidence: TrackerTotals | null };
    checker: { wrongCalls: number; missed: number; misclassified: number; falseWarnings: number; per5Min: number | null; falseWarningsOnNonErrors: number };
  };
  attribution: Record<Component, number> & { total: number; missed: Record<Component, number>; falseWarnings: Record<Component, number> };
  byClass: Record<ErrorClass, ClassCounts>; mutashabihat: ClassCounts; falseWarningsByClass: Record<string, number>;
  teacher: { wrongCalls: number; missed: number; falseWarnings: number; per5Min: number | null; per: number | null } | null;
}
/** Sums one variant over sessions; sessions without that variant are left out. */
export function totals(results: SessionResult[], variant: Variant): Totals {
  const rs = results.filter(r => r.variants[variant]), vs = rs.map(r => r.variants[variant]!);
  const sum = <T>(xs: T[], f: (x: T) => number) => xs.reduce((n, x) => n + f(x), 0);
  const recitedS = sum(rs, r => r.recitedS);
  const missed = sum(vs, v => v.missed), falseWarnings = sum(vs, v => v.falseWarnings), mistakes = sum(vs, v => v.mistakes);
  const edits = sum(vs, v => v.listener.edits), phonemes = sum(vs, v => v.listener.phonemes);
  const part = (k: "missed" | "falseWarnings") => Object.fromEntries(COMPONENTS.map(c => [c, sum(vs, v => v.attribution[c][k])])) as Record<Component, number>;
  const am = part("missed"), af = part("falseWarnings");
  const addClass = (cs: ClassCounts[]) => { const z = zeroClass(); for (const c of cs) for (const k of Object.keys(z) as (keyof ClassCounts)[]) z[k] += c[k]; return z; };
  const fwClass: Record<string, number> = {};
  for (const v of vs) for (const [c, n] of Object.entries(v.falseWarningsByClass)) fwClass[c] = (fwClass[c] ?? 0) + n;
  const teachers = vs.flatMap(v => v.teacher ? [v.teacher] : []);
  const cm = sum(rs, r => r.checker.missed), cf = sum(rs, r => r.checker.falseWarnings);
  const oracleTrackers = rs.flatMap(r => r.trackerOracle ? [r.trackerOracle] : []);
  return { sessions: rs.length, recitedMinutes: round(recitedS / 60),
    overall: { wrongCalls: missed + falseWarnings, missed, falseWarnings, per5Min: per5(missed + falseWarnings, recitedS), mistakes, caught: mistakes - missed,
      misclassified: sum(vs, v => v.misclassified), falseWarningsOnNonErrors: sum(vs, v => v.nonErrorFalseWarnings) },
    components: {
      listener: { per: phonemes ? round(edits / phonemes, 4) : null, edits, phonemes },
      tracker: { ...trackerTotals(vs.map(v => v.tracker)), oracleEvidence: oracleTrackers.length ? trackerTotals(oracleTrackers) : null },
      checker: { wrongCalls: cm + cf, missed: cm, misclassified: sum(rs, r => r.checker.misclassified), falseWarnings: cf, per5Min: per5(cm + cf, recitedS), falseWarningsOnNonErrors: sum(rs, r => r.checker.nonErrorFalseWarnings) },
    },
    attribution: { ...Object.fromEntries(COMPONENTS.map(c => [c, am[c] + af[c]])) as Record<Component, number>, total: missed + falseWarnings, missed: am, falseWarnings: af },
    byClass: Object.fromEntries(ERROR_CLASSES.map(c => [c, addClass(vs.map(v => v.byClass[c]))])) as Totals["byClass"],
    mutashabihat: addClass(vs.map(v => v.mutashabihat)), falseWarningsByClass: fwClass,
    teacher: teachers.length === vs.length && vs.length ? (() => {
      const tm = sum(teachers, t => t.missed), tf = sum(teachers, t => t.falseWarnings), tp = sum(teachers, t => t.phonemes);
      return { wrongCalls: tm + tf, missed: tm, falseWarnings: tf, per5Min: per5(tm + tf, recitedS), per: tp ? round(sum(teachers, t => t.edits) / tp, 4) : null };
    })() : null };
}

/** False warnings on real users' clean recitation (the realuser slice): per recited minute, per session, and in each session's first `EARLY_S` seconds against the rest. */
export interface CleanRate { sessions: number; recitedMinutes: number; falseWarnings: number; perRecitedMinute: number | null; perSession: number | null }
export interface CleanTotals extends CleanRate {
  firstSeconds: { seconds: number; falseWarnings: number; recitedMinutes: number; perRecitedMinute: number | null };
  rest: { falseWarnings: number; recitedMinutes: number; perRecitedMinute: number | null };
  falseWarningsOnNonErrors: number; byClass: Record<string, number>; byOrigin: Record<string, CleanRate>;
}
const perMinute = (n: number, seconds: number) => seconds ? round(n * 60 / seconds, 4) : null;
function cleanRate(rs: SessionResult[], variant: Variant): CleanRate {
  const vs = rs.filter(r => r.variants[variant]), recitedS = vs.reduce((n, r) => n + r.recitedS, 0);
  const fw = vs.reduce((n, r) => n + r.variants[variant]!.falseWarnings, 0);
  return { sessions: vs.length, recitedMinutes: round(recitedS / 60), falseWarnings: fw, perRecitedMinute: perMinute(fw, recitedS), perSession: vs.length ? round(fw / vs.length, 4) : null };
}
export function cleanTotals(results: SessionResult[], variant: Variant = "phone"): CleanTotals {
  const rs = results.filter(r => r.variants[variant]), vs = rs.map(r => r.variants[variant]!);
  const sum = <T>(xs: T[], f: (x: T) => number) => xs.reduce((n, x) => n + f(x), 0);
  const all = cleanRate(rs, variant), earlyS = sum(rs, r => r.earlyRecitedS ?? 0), early = sum(vs, v => v.earlyFalseWarnings ?? 0);
  const restS = sum(rs, r => r.recitedS) - earlyS, byClass: Record<string, number> = {};
  for (const v of vs) for (const [c, n] of Object.entries(v.falseWarningsByClass)) byClass[c] = (byClass[c] ?? 0) + n;
  const origins = [...new Set(rs.map(r => r.origin ?? "unknown"))].sort();
  return { ...all, firstSeconds: { seconds: EARLY_S, falseWarnings: early, recitedMinutes: round(earlyS / 60), perRecitedMinute: perMinute(early, earlyS) },
    rest: { falseWarnings: all.falseWarnings - early, recitedMinutes: round(restS / 60), perRecitedMinute: perMinute(all.falseWarnings - early, restS) },
    falseWarningsOnNonErrors: sum(vs, v => v.nonErrorFalseWarnings), byClass,
    byOrigin: Object.fromEntries(origins.map(o => [o, cleanRate(rs.filter(r => (r.origin ?? "unknown") === o), variant)])) };
}

export interface LabMeta { testSet: string; listener: string; teacher: string | null; sha: string; run: string | null; reference: string; manifest: string }
export interface LabReport extends LabMeta {
  version: typeof LAB_SCORE_VERSION; kind: "lab-score"; recordedAt: string; definitions: typeof DEFINITIONS;
  /** The split behind `overall`, `components` and `attribution`: exam on phone audio, else every session when there is no exam session. */
  headline: { split: Split | "all"; variant: Variant };
  shard: { index: number; count: number } | null;
  sessions: { scored: number; practice: number; exam: number; excluded: { id: string; reason: string }[] };
  splits: Record<Split | "all", Totals>;
  overall: Totals["overall"]; components: Totals["components"]; attribution: Totals["attribution"];
  /** `recorded`: sessions whose manifest `slices.source` is "recorded" (real takes of scripted mistakes), on phone audio.
   * `realuser`: real users' clean recitation (`slices.source` "realuser"), on phone audio. These sessions are left out of
   * `splits`, the headline and the other slices; `realuser` gives their false-warning rates. */
  slices: { studio: Record<Split | "all", Totals>; strict: Record<Split | "all", Totals>; recorded: Record<Split | "all", Totals>; realuser: Record<Split | "all", Totals> };
  realuser: Record<Split | "all", CleanTotals>;
  /** Practice cases only unless exam cases were asked for (exam: numbers only). */
  cases: LabCase[];
  /** Per-session counts behind every number above; merged shards re-aggregate from these. */
  results: SessionResult[];
}
export function labReport(results: SessionResult[], meta: LabMeta, o: { excluded?: { id: string; reason: string }[]; shard?: { index: number; count: number } | null; examCases?: boolean; recordedAt?: string } = {}): LabReport {
  const bySplit = (rs: SessionResult[], v: Variant) => ({ practice: totals(rs.filter(r => r.split === "practice"), v), exam: totals(rs.filter(r => r.split === "exam"), v), all: totals(rs, v) });
  const real = results.filter(r => r.source === "realuser"), planted = results.filter(r => r.source !== "realuser");
  const splits = bySplit(planted, "phone");
  const split: Split | "all" = splits.exam.sessions ? "exam" : "all";
  const head = splits[split];
  const visible = (r: SessionResult) => r.split === "practice" || !!o.examCases;
  return { version: LAB_SCORE_VERSION, kind: "lab-score", ...meta, recordedAt: o.recordedAt ?? new Date().toISOString(), definitions: DEFINITIONS,
    headline: { split, variant: "phone" }, shard: o.shard ?? null,
    sessions: { scored: results.length, practice: planted.filter(r => r.split === "practice").length, exam: planted.filter(r => r.split === "exam").length, excluded: o.excluded ?? [] },
    splits, overall: head.overall, components: head.components, attribution: head.attribution,
    slices: { studio: bySplit(planted, "studio"), strict: bySplit(planted.filter(r => r.strict), "phone"),
      recorded: bySplit(planted.filter(r => r.source === "recorded"), "phone"), realuser: bySplit(real, "phone") },
    realuser: { practice: cleanTotals(real.filter(r => r.split === "practice")), exam: cleanTotals(real.filter(r => r.split === "exam")), all: cleanTotals(real) },
    cases: results.filter(visible).flatMap(r => r.cases), results: results.map(r => visible(r) ? r : { ...r, cases: [] }) };
}

/** Combines shard reports of one run into one; every shard must be present. */
export function mergeReports(parts: LabReport[], o: { examCases?: boolean; recordedAt?: string } = {}): LabReport {
  if (!parts.length) throw new Error("no shard reports");
  const count = parts[0].shard?.count ?? 1, seen = new Set(parts.map(p => p.shard?.index ?? 0));
  for (let i = 0; i < count; i++) if (!seen.has(i)) throw new Error(`missing shard ${i} of ${count}`);
  for (const p of parts) for (const f of ["testSet", "listener", "teacher", "sha", "reference"] as const)
    if (p[f] !== parts[0][f]) throw new Error(`shards disagree on ${f}: ${p[f]} vs ${parts[0][f]}`);
  const results = parts.flatMap(p => p.results).sort((a, b) => a.id.localeCompare(b.id));
  if (new Set(results.map(r => r.id)).size !== results.length) throw new Error("a session was scored by two shards");
  const { testSet, listener, teacher, sha, run, reference, manifest } = parts[0];
  return labReport(results, { testSet, listener, teacher, sha, run, reference, manifest },
    { excluded: parts.flatMap(p => p.sessions.excluded), examCases: o.examCases, recordedAt: o.recordedAt });
}
