import type { CheckerEvent, EngineEvent, PositionEvent, WordRef } from "../events.js";
import { readFileSync } from "node:fs";
import type { ReferenceIndex, PhraseAsset, PageAsset } from "../reference.js";
import { pageRange } from "../reference.js";
import { createTracker } from "../tracker.js";
import type { TrackerOptions } from "../tracker.js";
import { createChecker } from "../checker.js";
import type { CheckerOptions } from "../checker.js";
import { chunk } from "./index.js";
import type { QudRecording } from "./index.js";
import type { LabelEvent } from "./acceptance.js";

export type SessionContext = { kind: "none" } | { kind: "surah"; surah: number } | { kind: "page"; page: number }
  | { kind: "assignment"; from: WordRef; to: WordRef } | { kind: "resume"; at: WordRef };
export interface Interval { startS: number; endS: number }
export type SessionSpan = Interval & ({ id: string; kind: "recitation" | "repeat" | "restart"; from: WordRef; to: WordRef }
  | { id: string; kind: "non_quran"; label: string });
export interface SessionEvent extends LabelEvent, Interval { span: string }
export interface BenchSession {
  version: 1; id: string; track: "recitation" | "memorization"; speaker: string;
  source: "stitched" | "recorded" | "qud"; audio: string; durationS: number; context: SessionContext;
  /** Paths and split provenance; stitched entries must carry the session speaker. */
  segments?: (Interval & { id: string; speaker: string; audio: string; split: "eval"; timing: "reviewed" | "alignment" })[];
  spans: SessionSpan[];
  words: (Interval & { span: string; at: WordRef })[];
  events: SessionEvent[];
  nonErrors: (Interval & { span: string; at: WordRef; kind: "repeat" | "restart" | "self_correction" })[];
}
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const ayahKey = (r: WordRef) => `${r.surah}:${r.ayah}`;
export function validateSession(s: BenchSession, index: ReferenceIndex): void {
  const fail = (why: string): never => { throw new Error(`Session ${s?.id ?? "?"}: ${why}`); };
  const refs = new Map(index.words.map((w, i) => [key(w), i]));
  const ref = (r: WordRef): number => {
    if (!r || ![r.surah, r.ayah, r.word].every(Number.isInteger)) return fail("invalid word reference");
    return refs.get(key(r)) ?? fail(`unknown word ${key(r)}`);
  };
  if (!s || s.version !== 1 || !s.id || !s.speaker || !s.audio || !["recitation", "memorization"].includes(s.track)
      || !["stitched", "recorded", "qud"].includes(s.source) || !(Number.isFinite(s.durationS) && s.durationS > 0)) fail("invalid header");
  if (![s.spans, s.words, s.events, s.nonErrors].every(Array.isArray)) fail("missing truth arrays");
  const time = (t: Interval, zero = false) => {
    if (![t.startS, t.endS].every(Number.isFinite) || t.startS < 0 || t.endS > s.durationS + 1e-6
      || (zero ? t.endS < t.startS : t.endS <= t.startS)) fail("invalid interval");
  };
  switch (s.context?.kind) {
    case "assignment": if (ref(s.context.from) > ref(s.context.to)) fail("reversed assignment"); break;
    case "resume": ref(s.context.at); break;
    case "surah": { const surah = s.context.surah; if (!Number.isInteger(surah) || !index.words.some(w => w.surah === surah)) fail("invalid surah"); break; }
    case "page": if (!Number.isInteger(s.context.page) || s.context.page < 1 || s.context.page > 604) fail("invalid Madani page"); break;
    case "none": break;
    default: fail("invalid context");
  }
  const spans = new Map<string, SessionSpan>();
  let last = 0;
  for (const span of s.spans) {
    time(span);
    if (!span.id || spans.has(span.id) || span.startS < last) fail("duplicate or overlapping/unordered spans");
    if (span.kind === "non_quran") { if (!span.label) fail("missing non-Quran label"); }
    else if (!["recitation", "repeat", "restart"].includes(span.kind) || ref(span.from) > ref(span.to)) fail("invalid recitation span");
    spans.set(span.id, span); last = span.endS;
  }
  if (!s.spans.length || !s.words.length) fail("empty recitation truth");
  const inside = (t: Interval & { span: string }) => {
    time(t, true); const span = spans.get(t.span);
    if (!span || span.kind === "non_quran" || t.startS < span.startS || t.endS > span.endS) fail("truth outside recitation occurrence");
    return span as Exclude<SessionSpan, { kind: "non_quran" }>;
  };
  last = 0;
  const previousWord = new Map<string, number>();
  for (const w of s.words) {
    const span = inside(w); time(w);
    if (w.startS < last || ref(w.at) < ref(span.from) || ref(w.at) > ref(span.to)) fail("unordered/overlapping words or word outside span range");
    if (ref(w.at) <= (previousWord.get(w.span) ?? -1)) fail("repeated/backward visit needs its own occurrence span");
    previousWord.set(w.span, ref(w.at));
    last = w.endS;
  }
  for (const span of s.spans) if (span.kind !== "non_quran" && !s.words.some(w => w.span === span.id)) fail("span lacks word timing");
  for (const e of s.events) {
    inside(e); ref(e.location);
    if (!["wrong_word", "omitted_word", "added_word", "skipped_passage"].includes(e.errorClass)
        || !["confirmed", "resolved", "suspected"].includes(e.state)) fail("unsupported event label");
  }
  for (const e of s.nonErrors) {
    inside(e); ref(e.at);
    if (!["repeat", "restart", "self_correction"].includes(e.kind)) fail("invalid non-error");
  }
  if (s.source === "stitched" && !s.segments?.length) fail("stitched session lacks speaker provenance");
  last = 0;
  for (const seg of s.segments ?? []) {
    time(seg);
    if (seg.speaker !== s.speaker || seg.split !== "eval" || !seg.audio || !seg.id || seg.startS < last
      || !["reviewed", "alignment"].includes(seg.timing)) fail("mixed speaker, split, or invalid segment provenance");
    last = seg.endS;
  }
}

export function weakerContexts(c: SessionContext): SessionContext[] {
  if (c.kind === "none") return [c];
  if (c.kind === "assignment") return [c, { kind: "surah", surah: c.from.surah }, { kind: "none" }];
  if (c.kind === "resume") return [c, { kind: "surah", surah: c.at.surah }, { kind: "none" }];
  return [c, { kind: "none" }];
}
const madaniPages: PageAsset = JSON.parse(readFileSync(new URL("../../assets/madani-pages-v1.json", import.meta.url), "utf8"));
export function contextAdapter(c: SessionContext): { supported: boolean; effective: string; tracker: TrackerOptions; checker: CheckerOptions } {
  const tracker: TrackerOptions = c.kind === "resume" ? { prior: { surah: c.at.surah, ayah: c.at.ayah, weight: 1 } }
    : c.kind === "surah" ? { priorRange: { ranges: [{ from: { surah: c.surah, ayah: 1 }, to: { surah: c.surah, ayah: Infinity } }], weight: 1 } }
    : c.kind === "page" ? { priorRange: { ranges: [pageRange(madaniPages, c.page)], weight: 1 } } : {};
  return { supported: true, effective: c.kind, tracker,
    checker: c.kind === "assignment" ? { passage: { from: c.from, to: c.to } } : {} };
}
export const ERROR_CLASSES = ["wrong_word", "omitted_word", "added_word", "skipped_passage"] as const;
export type ErrorClass = (typeof ERROR_CLASSES)[number];
export interface RecallSlice { errors: number; hits: number; recall: number | null }
/** Near-identical ayah phonemes (e.g. 94:5/94:6 differ only by a leading fa-): edit distance within 10% of the longer. */
export function isTwin(a: string, b: string): boolean {
  if (a === b) return true;
  const limit = Math.floor(Math.max(a.length, b.length) * 0.1);
  if (Math.abs(a.length - b.length) > limit) return false;
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const row = [i];
    for (let j = 1; j <= b.length; j++) row[j] = Math.min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    prev = row;
  }
  return prev[b.length] <= limit;
}
const ayahTexts = new WeakMap<ReferenceIndex, Map<number, string>>();
const recallSlice = (errors: number, hits: number): RecallSlice => ({ errors, hits, recall: errors ? hits / errors : null });
export interface SessionScore {
  id: string; track: BenchSession["track"]; source: BenchSession["source"]; speaker: string; context: SessionContext;
  supportedContext: boolean; effectiveContext: string; gating: false;
  errors: number; hits: number; misses: number; warnings: number; falseWarnings: number;
  precision: number | null; recall: number | null;
  cleanSeconds: number; cleanWarnings: number; cleanWarningsPer5Min: number | null;
  nonErrorWarnings: number; uncertainWarnings: number; suspectedOnlyWarnings: number;
  uncertainErrors: number; ambiguousErrors: number;
  wrongPositionEvents: number; positionSamples: number; locatedSamples: number; locatedShare: number | null;
  firstCorrectLockDelayS: number | null; within5S: boolean;
  suspectedDelayS: number[]; confirmedDelayS: number[];
  /** All labelled errors are denominators; abstentions count as misses. */
  byClass: Record<ErrorClass, RecallSlice>; mutashabihat: RecallSlice;
  /** Missed errors by cause, checker `uncertain` first, then tracker ambiguity between twin ayahs, then other ambiguity. */
  abstentions: { uncertain: number; twinAmbiguous: number; ambiguous: number };
  /** Entries into tracker `ambiguous`, split by whether every candidate is a twin of the others. */
  ambiguousEpisodes: { twin: number; general: number };
  assignments: { warningId: string; eventIndex: number | null; confirmed: boolean }[];
}
export function scoreSessionEvents(s: BenchSession, events: EngineEvent[], index: ReferenceIndex, context = s.context): SessionScore {
  validateSession(s, index);
  const ord = new Map<string, number>();
  for (const w of index.words) if (!ord.has(ayahKey(w))) ord.set(ayahKey(w), ord.size);
  const ay = (w: WordRef) => ord.get(ayahKey(w))!;
  const positions = events.filter((e): e is PositionEvent => e.type === "position").sort((a, b) => a.timeMs - b.timeMs);
  const warnings = new Map<string, CheckerEvent[]>();
  for (const e of events) if (e.type === "checker") { const list = warnings.get(e.id) ?? []; list.push(e); warnings.set(e.id, list); }
  const score: SessionScore = { id: s.id, track: s.track, source: s.source, speaker: s.speaker, context,
    supportedContext: contextAdapter(context).supported, effectiveContext: contextAdapter(context).effective, gating: false,
    errors: s.events.length, hits: 0, misses: 0, warnings: 0, falseWarnings: 0, precision: null, recall: null,
    cleanSeconds: 0, cleanWarnings: 0, cleanWarningsPer5Min: null, nonErrorWarnings: 0, uncertainWarnings: 0,
    suspectedOnlyWarnings: 0, uncertainErrors: 0, ambiguousErrors: 0, wrongPositionEvents: 0,
    positionSamples: 0, locatedSamples: 0, locatedShare: null, firstCorrectLockDelayS: null, within5S: false,
    suspectedDelayS: [], confirmedDelayS: [], byClass: {} as SessionScore["byClass"], mutashabihat: recallSlice(0, 0),
    abstentions: { uncertain: 0, twinAmbiguous: 0, ambiguous: 0 }, ambiguousEpisodes: { twin: 0, general: 0 }, assignments: [] };
  const windowEnd = (e: SessionEvent) => {
    if (e.errorClass === "omitted_word" || e.errorClass === "skipped_passage") {
      const next = s.words.find(w => w.startS >= e.endS && ay(w.at) === ay(e.location) + 2);
      return next ? s.spans.find(p => p.id === next.span)!.endS : s.durationS;
    }
    return Math.min(s.durationS, e.endS + 6);
  };
  const used = new Set<number>();
  const atTruth = (t: number) => {
    let lo = 0, hi = s.words.length;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (s.words[mid].startS <= t) lo = mid + 1; else hi = mid; }
    const out: BenchSession["words"] = [];
    for (let i = lo - 1; i >= 0 && t - s.words[i].endS <= 2; i--) out.push(s.words[i]);
    return out;
  };
  const quranAt = (t: number) => !s.spans.some(p => p.kind === "non_quran" && p.startS <= t && t < p.endS) && atTruth(t).length > 0;
  const correct = (p: PositionEvent, t: number) => !!p.observed[0] && atTruth(t).some(w => Math.abs(ay(w.at) - ay(p.observed[0].word)) <= 1);
  const cleanAt = (t: number) => quranAt(t) && atTruth(t).some(w => t < w.endS) && !s.events.some(e => e.startS <= t && t <= e.endS);
  for (const states of [...warnings.values()].sort((a, b) => a[0].timeMs - b[0].timeMs)) {
    states.sort((a, b) => a.timeMs - b.timeMs);
    const first = states[0], t = first.timeMs / 1000;
    const confirmed = states.some(e => e.state === "confirmed");
    const eligible = s.events.map((e, i) => ({ e, i })).filter(({ e, i }) => !used.has(i)
      && e.errorClass === first.errorClass && key(e.location) === key(first.location) && t >= e.startS && t <= windowEnd(e)
      && !s.words.some(w => w.span !== e.span && key(w.at) === key(e.location) && w.startS > e.startS && w.startS <= t))
      .sort((a, b) => a.e.endS - b.e.endS || a.i - b.i);
    const matched = eligible[0];
    if (matched) used.add(matched.i);
    score.assignments.push({ warningId: first.id, eventIndex: matched?.i ?? null, confirmed });
    if (confirmed) {
      score.warnings++; if (matched) score.hits++; else { score.falseWarnings++; if (cleanAt(t)) score.cleanWarnings++; }
      if (s.nonErrors.some(n => key(n.at) === key(first.location) && n.startS <= t && t <= n.endS + 6
        && !s.words.some(w => w.span !== n.span && key(w.at) === key(n.at) && w.startS > n.startS && w.startS <= t))) score.nonErrorWarnings++;
    } else if (states.some(e => e.state === "uncertain")) { score.uncertainWarnings++; if (matched) score.uncertainErrors++; }
    else score.suspectedOnlyWarnings++;
    if (matched) {
      const suspect = states.find(e => e.state === "suspected"), confirm = states.find(e => e.state === "confirmed");
      if (suspect) score.suspectedDelayS.push(Math.max(0, suspect.timeMs / 1000 - matched.e.endS));
      if (confirm) score.confirmedDelayS.push(Math.max(0, confirm.timeMs / 1000 - matched.e.endS));
    }
  }
  let pi = -1;
  for (let sample = 0; sample < Math.ceil(s.durationS * 10); sample++) {
    const t = sample / 10;
    while (pi + 1 < positions.length && positions[pi + 1].timeMs <= t * 1000) pi++;
    if (!quranAt(t)) continue;
    score.positionSamples++; if (cleanAt(t)) score.cleanSeconds += Math.min(0.1, s.durationS - t);
    const p = positions[pi];
    if (p?.state === "locked" && correct(p, t)) score.locatedSamples++;
  }
  for (const p of positions) {
    const t = p.timeMs / 1000;
    if (quranAt(t) && p.state === "locked") {
      if (!correct(p, t)) score.wrongPositionEvents++;
      else if (score.firstCorrectLockDelayS === null) score.firstCorrectLockDelayS = Math.max(0, t - s.words[0].startS);
    }
  }
  const hit = new Set(score.assignments.filter(a => a.confirmed && a.eventIndex !== null).map(a => a.eventIndex!));
  let ayahText = ayahTexts.get(index);
  if (!ayahText) {
    ayahText = new Map();
    for (const w of index.words) ayahText.set(ay(w), (ayahText.get(ay(w)) ?? "") + index.phonemes.slice(w.start, w.end));
    ayahTexts.set(index, ayahText);
  }
  const twin = (p: PositionEvent) => {
    const ayahs = [...new Set(p.observed.map(o => ay(o.word)))];
    return ayahs.length > 1 && ayahs.every(a => isTwin(ayahText!.get(a)!, ayahText!.get(ayahs[0])!));
  };
  const uncertainEvents = new Set(score.assignments.filter(a => !a.confirmed && a.eventIndex !== null
    && warnings.get(a.warningId)!.some(e => e.state === "uncertain")).map(a => a.eventIndex!));
  s.events.forEach((e, i) => {
    const before = positions.filter(p => p.timeMs / 1000 <= e.startS).at(-1);
    const amb = [...(before?.state === "ambiguous" ? [before] : []),
      ...positions.filter(p => p.state === "ambiguous" && p.timeMs / 1000 >= e.startS && p.timeMs / 1000 <= windowEnd(e))];
    if (amb.length) score.ambiguousErrors++;
    if (hit.has(i)) return;
    if (uncertainEvents.has(i)) score.abstentions.uncertain++;
    else if (amb.length && amb.every(twin)) score.abstentions.twinAmbiguous++;
    else if (amb.length) score.abstentions.ambiguous++;
  });
  positions.forEach((p, i) => {
    if (p.state === "ambiguous" && positions[i - 1]?.state !== "ambiguous") score.ambiguousEpisodes[twin(p) ? "twin" : "general"]++;
  });
  const slice = (keep: (e: SessionEvent) => boolean) => {
    const idx = s.events.map((e, i) => i).filter(i => keep(s.events[i]));
    return recallSlice(idx.length, idx.filter(i => hit.has(i)).length);
  };
  for (const c of ERROR_CLASSES) score.byClass[c] = slice(e => e.errorClass === c);
  score.mutashabihat = slice(e => !!e.forms_other_ayah);
  score.misses = score.errors - score.hits;
  score.precision = score.warnings ? score.hits / score.warnings : null;
  score.recall = score.errors ? score.hits / score.errors : null;
  score.cleanWarningsPer5Min = score.cleanSeconds ? score.cleanWarnings * 300 / score.cleanSeconds : null;
  score.locatedShare = score.positionSamples ? score.locatedSamples / score.positionSamples : null;
  score.within5S = score.firstCorrectLockDelayS !== null && score.firstCorrectLockDelayS <= 5;
  return score;
}

export interface DelaySummary { n: number; p50: number | null; p90: number | null; censoredByMiss: number }
export interface SessionSummary {
  key: string; track: BenchSession["track"]; context: SessionContext["kind"]; source: BenchSession["source"];
  sessions: number; speakers: number; unsupportedContexts: number;
  errors: number; hits: number; warnings: number; precision: number | null; recall: number | null;
  byClass: Record<ErrorClass, RecallSlice>; mutashabihat: RecallSlice;
  cleanPer5: number | null; located: number | null; within5s: number | null; wrongPositions: number; nonErrorWarnings: number;
  /** Matched warnings only; errors without that state are censored, not zero. */
  suspectedDelay: DelaySummary; confirmedDelay: DelaySummary;
  abstentionShare: Record<keyof SessionScore["abstentions"], number | null>;
  /** Sessions with no labelled errors: checker `uncertain` warnings and tracker ambiguity entries per session. */
  cleanSessions: number; perCleanSession: Record<keyof SessionScore["abstentions"], number | null>;
}
const percentile = (xs: number[], q: number) => xs.length ? [...xs].sort((a, b) => a - b)[Math.ceil(q * xs.length) - 1] : null;
export function summarizeSessions(rows: SessionScore[]): SessionSummary[] {
  const groups = new Map<string, SessionScore[]>();
  for (const r of rows) {
    const k = `${r.track}.${r.context.kind}.${r.source}`;
    groups.set(k, [...(groups.get(k) ?? []), r]);
  }
  return [...groups].map(([key, rs]) => {
    const total = (f: (r: SessionScore) => number) => rs.reduce((n, r) => n + f(r), 0);
    const ratio = (a: number, b: number) => b ? a / b : null;
    const errors = total(r => r.errors), hits = total(r => r.hits), warnings = total(r => r.warnings);
    const delay = (f: (r: SessionScore) => number[]): DelaySummary => {
      const xs = rs.flatMap(f);
      return { n: xs.length, p50: percentile(xs, 0.5), p90: percentile(xs, 0.9), censoredByMiss: errors - xs.length };
    };
    const sliceTotal = (f: (r: SessionScore) => RecallSlice) => recallSlice(total(r => f(r).errors), total(r => f(r).hits));
    const clean = rs.filter(r => r.errors === 0);
    const cleanSeconds = total(r => r.cleanSeconds);
    return { key, track: rs[0].track, context: rs[0].context.kind, source: rs[0].source,
      sessions: rs.length, speakers: new Set(rs.map(r => r.speaker)).size, unsupportedContexts: rs.filter(r => !r.supportedContext).length,
      errors, hits, warnings, precision: ratio(hits, warnings), recall: ratio(hits, errors),
      byClass: Object.fromEntries(ERROR_CLASSES.map(c => [c, sliceTotal(r => r.byClass[c])])) as SessionSummary["byClass"],
      mutashabihat: sliceTotal(r => r.mutashabihat),
      cleanPer5: cleanSeconds ? total(r => r.cleanWarnings) * 300 / cleanSeconds : null,
      located: ratio(total(r => r.locatedSamples), total(r => r.positionSamples)), within5s: ratio(rs.filter(r => r.within5S).length, rs.length),
      wrongPositions: total(r => r.wrongPositionEvents), nonErrorWarnings: total(r => r.nonErrorWarnings),
      suspectedDelay: delay(r => r.suspectedDelayS), confirmedDelay: delay(r => r.confirmedDelayS),
      abstentionShare: { uncertain: ratio(total(r => r.abstentions.uncertain), errors),
        twinAmbiguous: ratio(total(r => r.abstentions.twinAmbiguous), errors), ambiguous: ratio(total(r => r.abstentions.ambiguous), errors) },
      cleanSessions: clean.length,
      perCleanSession: { uncertain: ratio(clean.reduce((n, r) => n + r.uncertainWarnings, 0), clean.length),
        twinAmbiguous: ratio(clean.reduce((n, r) => n + r.ambiguousEpisodes.twin, 0), clean.length),
        ambiguous: ratio(clean.reduce((n, r) => n + r.ambiguousEpisodes.general, 0), clean.length) } };
  });
}

export function runSession(s: BenchSession, evidence: Pick<QudRecording, "inventory" | "frameMs" | "nFrames" | "phonemes" | "posteriors">,
    index: ReferenceIndex, context = s.context, phrases?: PhraseAsset): SessionScore {
  validateSession(s, index);
  if (evidence.inventory !== index.inventoryId || Math.abs(evidence.nFrames * evidence.frameMs / 1000 - s.durationS) > 0.1) throw new Error(`Evidence mismatch: ${s.id}`);
  const adapter = contextAdapter(context);
  const tracker = createTracker(index, { mode: s.track === "memorization" ? "memorization" : "free", phrases, ...adapter.tracker });
  const checker = createChecker(index, { phrases, ...adapter.checker });
  const out: EngineEvent[] = [];
  for (const e of chunk(evidence as QudRecording, 0.04, true)) {
    const positions = tracker.push({ ...e, posteriors: undefined });
    out.push(...positions, ...checker.push(e, positions));
  }
  const end = tracker.end(); out.push(...end, ...checker.end(end));
  return scoreSessionEvents(s, out, index, context);
}

/** Annotation rounding allowed past the audio clock; truth is clipped to the clock, larger overruns are rejected. */
const CLOCK_SLACK_S = 0.1;
/** Preserve every visit; never sort truth into Quran order. Special words become non-Quran spans. */
export function qudSession(rec: QudRecording, audio: string): BenchSession {
  const spans: SessionSpan[] = [], words: BenchSession["words"] = [];
  // Truth uses the original audio clock; rounded evidence frames can truncate the last word.
  const durationS: number = (rec.meta as any).duration_s ?? rec.nFrames * rec.frameMs / 1000;
  const clock = <T extends { start_s: number; end_s: number }>(t: T): T => {
    if (t.end_s > durationS + CLOCK_SLACK_S) throw new Error(`QUD ${rec.id}: truth at ${t.start_s}s overruns audio clock`);
    return t.end_s > durationS ? { ...t, end_s: durationS } : t;
  };
  const order = (r: WordRef) => (r.surah * 1000 + r.ayah) * 1000 + r.word;
  let last: WordRef | undefined;
  const truth = rec.truth.words.map(clock);
  // Non-Quran intervals are clipped to the gap between timed words, never stretched over them.
  const noise = rec.truth.non_quran.map(clock).map(n => {
    let { start_s, end_s } = n;
    for (const w of truth) {
      if (w.start_s >= end_s || w.end_s <= start_s) continue;
      if (w.start_s >= start_s && w.end_s <= end_s || w.start_s <= start_s && w.end_s >= end_s)
        throw new Error(`QUD ${rec.id}: non-Quran truth at ${n.start_s}s contradicts word ${w.word}`);
      if (w.start_s < start_s) start_s = w.end_s; else end_s = w.start_s;
    }
    return { start_s, end_s };
  });
  for (const w of truth) {
    const nums = /^(\d+):(\d+):(\d+)$/.exec(w.word);
    if (!nums) {
      spans.push({ id: `span-${spans.length}`, kind: "non_quran", label: w.word, startS: w.start_s, endS: w.end_s }); continue;
    }
    const at = { surah: +nums[1], ayah: +nums[2], word: +nums[3] };
    const prev = spans.at(-1);
    // One span per ayah occurrence. Within an ayah, backward/equal movement is a repeat; returning to an
    // earlier ayah is a restart when it lands on word 1, otherwise a repeat. Special words do not hide the return.
    if (prev && prev.kind !== "non_quran" && ayahKey(prev.to) === ayahKey(at) && at.word > prev.to.word && w.start_s - prev.endS <= 2
      && !noise.some(n => n.start_s < w.start_s && n.end_s > prev.endS)) {
      prev.to = at; prev.endS = w.end_s;
    } else {
      const back = last && order(at) <= order(last) ? last : undefined;
      spans.push({ id: `span-${spans.length}`, kind: !back ? "recitation" : ayahKey(back) !== ayahKey(at) && at.word === 1 ? "restart" : "repeat",
        from: at, to: at, startS: w.start_s, endS: w.end_s });
    }
    words.push({ span: spans.at(-1)!.id, at, startS: w.start_s, endS: w.end_s });
    last = at;
  }
  for (const n of noise) spans.push({ id: `noise-${spans.length}`, kind: "non_quran", label: "non_quran", startS: n.start_s, endS: n.end_s });
  spans.sort((a, b) => a.startS - b.startS);
  return { version: 1, id: rec.id, speaker: (rec.meta as any).reciter ?? rec.id, track: "recitation", source: "qud", audio,
    durationS, context: { kind: "none" }, spans, words, events: [],
    nonErrors: words.flatMap(w => {
      const kind = spans.find(s => s.id === w.span)?.kind;
      return kind === "repeat" || kind === "restart" ? [{ ...w, kind }] : [];
    }) };
}

export interface QudAudit {
  id: string; status: "converted" | "excluded"; reason: string | null;
  spans: number; repeats: number; restarts: number; nonQuran: number; surahJumps: number;
}
/** Convert and validate one recording; failures are excluded with their reason, never counted. */
export function qudAudit(rec: QudRecording, index: ReferenceIndex, audio = `${rec.id}.wav`): { session: BenchSession | null; audit: QudAudit } {
  try {
    const session = qudSession(rec, audio);
    validateSession(session, index);
    const count = (kind: SessionSpan["kind"]) => session.spans.filter(s => s.kind === kind).length;
    return { session, audit: { id: rec.id, status: "converted", reason: null, spans: session.spans.length, repeats: count("repeat"),
      restarts: count("restart"), nonQuran: count("non_quran"),
      surahJumps: session.words.filter((w, i) => i && w.at.surah !== session.words[i - 1].at.surah).length } };
  } catch (e) {
    return { session: null, audit: { id: rec.id, status: "excluded", reason: e instanceof Error ? e.message : String(e),
      spans: 0, repeats: 0, restarts: 0, nonQuran: 0, surahJumps: 0 } };
  }
}
