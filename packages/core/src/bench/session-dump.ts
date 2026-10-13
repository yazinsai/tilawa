import type { CheckerEvent, EngineEvent, PositionEvent, WordRef } from "../events.js";
import type { PhraseAsset, ReferenceAsset, ReferenceIndex } from "../reference.js";
import { createTracker } from "../tracker.js";
import { createChecker } from "../checker.js";
import { chunk } from "./index.js";
import type { QudRecording } from "./index.js";
import { contextAdapter } from "./sessions.js";
import type { BenchSession, SessionContext, SessionScore } from "./sessions.js";

/** Review-only session evidence for the scoreboard page. Never read by the scorer. */
export const SESSION_DUMP_VERSION = 3;
export interface SessionDumpWarning {
  id: string; errorClass: CheckerEvent["errorClass"]; at: string;
  states: { state: CheckerEvent["state"]; timeMs: number; confidence: number }[];
  eventIndex: number | null; confirmed: boolean; nonError: boolean;
}
export interface SessionDumpContext {
  context: SessionContext; effectiveContext: string; score: SessionScore;
  warnings: SessionDumpWarning[]; wrongPositions: { timeMs: number; observed: string; recited: string[] }[];
}
export interface SessionDumpRecord {
  id: string; track: BenchSession["track"]; source: BenchSession["source"]; speaker: string; audio: string; durationS: number;
  spans: BenchSession["spans"]; words: { span: string; at: string; text: string; startS: number; endS: number }[];
  events: (BenchSession["events"][number] & { at: string })[]; nonErrors: (BenchSession["nonErrors"][number] & { atKey: string })[];
  contexts: Record<string, SessionDumpContext>;
}

const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const ayahKey = (r: WordRef) => `${r.surah}:${r.ayah}`;
export const contextKey = (c: SessionContext) => c.kind === "surah" ? `surah ${c.surah}` : c.kind === "page" ? `page ${c.page}`
  : c.kind === "assignment" ? `assignment ${key(c.from)}-${key(c.to)}` : c.kind === "resume" ? `resume ${key(c.at)}` : "none";

/** Same engine loop as `runSession`, returning the raw events. */
export function sessionEvents(s: BenchSession, evidence: Pick<QudRecording, "inventory" | "frameMs" | "nFrames" | "phonemes" | "posteriors">,
    index: ReferenceIndex, context = s.context, phrases?: PhraseAsset): EngineEvent[] {
  const adapter = contextAdapter(context);
  const tracker = createTracker(index, { mode: s.track === "memorization" ? "memorization" : "free", phrases, ...adapter.tracker });
  const checker = createChecker(index, { phrases, ...adapter.checker });
  const out: EngineEvent[] = [];
  for (const e of chunk(evidence as QudRecording, 0.04, true)) {
    const positions = tracker.push({ ...e, posteriors: undefined });
    out.push(...positions, ...checker.push(e, positions));
  }
  const end = tracker.end(); out.push(...end, ...checker.end(end));
  return out;
}

export const quranText = (asset: ReferenceAsset) => new Map(asset.ayahs.flatMap(a => a.w.map((w, i) => [`${a.s}:${a.a}:${i + 1}`, w] as const)));

export function sessionDumpRecord(s: BenchSession, audio: string, text: ReadonlyMap<string, string>): SessionDumpRecord {
  return { id: s.id, track: s.track, source: s.source, speaker: s.speaker, audio, durationS: s.durationS, spans: s.spans,
    words: s.words.map(w => ({ span: w.span, at: key(w.at), text: text.get(key(w.at)) ?? "", startS: w.startS, endS: w.endS })),
    events: s.events.map(e => ({ ...e, at: key(e.location) })), nonErrors: s.nonErrors.map(n => ({ ...n, atKey: key(n.at) })), contexts: {} };
}

/** Per-warning detail behind `score`; the non-error and wrong-position rules mirror `scoreSessionEvents`. */
export function sessionDumpContext(s: BenchSession, events: EngineEvent[], score: SessionScore, index: ReferenceIndex): SessionDumpContext {
  const ord = new Map<string, number>();
  for (const w of index.words) if (!ord.has(ayahKey(w))) ord.set(ayahKey(w), ord.size);
  const ay = (w: WordRef) => ord.get(ayahKey(w))!;
  const recited = (t: number) => {
    const out: BenchSession["words"] = [];
    for (let i = s.words.length - 1; i >= 0; i--) if (s.words[i].startS <= t) {
      for (let j = i; j >= 0 && t - s.words[j].endS <= 2; j--) out.push(s.words[j]);
      break;
    }
    return out;
  };
  const quranAt = (t: number) => !s.spans.some(p => p.kind === "non_quran" && p.startS <= t && t < p.endS) && recited(t).length > 0;
  const states = new Map<string, CheckerEvent[]>();
  for (const e of events) if (e.type === "checker") states.set(e.id, [...(states.get(e.id) ?? []), e]);
  const assigned = new Map(score.assignments.map(a => [a.warningId, a]));
  const warnings = [...states.values()].map(list => {
    list.sort((a, b) => a.timeMs - b.timeMs);
    const first = list[0], t = first.timeMs / 1000, a = assigned.get(first.id);
    const confirmed = list.some(e => e.state === "confirmed");
    const nonError = confirmed && s.nonErrors.some(n => key(n.at) === key(first.location) && n.startS <= t && t <= n.endS + 6
      && !s.words.some(w => w.span !== n.span && key(w.at) === key(n.at) && w.startS > n.startS && w.startS <= t));
    return { id: first.id, errorClass: first.errorClass, at: key(first.location), states: list.map(e => ({ state: e.state, timeMs: e.timeMs, confidence: e.confidence })),
      eventIndex: a?.eventIndex ?? null, confirmed, nonError };
  }).sort((a, b) => a.states[0].timeMs - b.states[0].timeMs);
  const wrongPositions = events.filter((e): e is PositionEvent => e.type === "position").flatMap(p => {
    const t = p.timeMs / 1000, truth = recited(t);
    if (!quranAt(t) || p.state !== "locked") return [];
    if (p.observed[0] && truth.some(w => Math.abs(ay(w.at) - ay(p.observed[0].word)) <= 1)) return [];
    return [{ timeMs: p.timeMs, observed: p.observed[0] ? key(p.observed[0].word) : "", recited: [...new Set(truth.map(w => key(w.at)))] }];
  });
  return { context: score.context, effectiveContext: score.effectiveContext, score, warnings, wrongPositions };
}
