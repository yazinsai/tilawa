import type { EngineEvent, PositionEvent, WordRef } from "../events.js";
import type { DecodedPhoneme, PhonemeEvidence } from "../evidence.js";
import type { ReferenceIndex } from "../reference.js";
import { createTracker } from "../tracker.js";
import type { TrackerOptions } from "../tracker.js";

export const BENCH_REPORT_VERSION = 1;

/** One QUD alignment-benchmark recording after the teacher pass, as written by `tools/export_qud.py`. */
export interface QudRecording {
  id: string;
  meta: { style: string; content: string; noisy: boolean; passages: string; duration_s: number; repeat_events: number };
  inventory: string;
  frameMs: number;
  nFrames: number;
  phonemes: DecodedPhoneme[];
  posteriors: { k: number; ids: number[]; probs: number[] };
  truth: { words: { word: string; start_s: number; end_s: number }[]; non_quran: { start_s: number; end_s: number }[] };
}

/** Consecutive evidence chunks of about `seconds` each, never splitting a phoneme. */
export function chunk(rec: QudRecording, seconds: number, posteriors: boolean): PhonemeEvidence[] {
  const per = Math.max(1, Math.round((seconds * 1000) / rec.frameMs));
  const out: PhonemeEvidence[] = [];
  let p = 0;
  for (let a = 0; a < rec.nFrames; ) {
    let b = Math.min(rec.nFrames, a + per);
    const phonemes: DecodedPhoneme[] = [];
    while (p < rec.phonemes.length && rec.phonemes[p].start < b) {
      phonemes.push(rec.phonemes[p]);
      b = Math.max(b, rec.phonemes[p].end + 1);
      p++;
    }
    const e: PhonemeEvidence = { inventory: rec.inventory, frameMs: rec.frameMs, phonemes, decodedTo: b };
    if (posteriors) {
      const { k, ids, probs } = rec.posteriors;
      e.posteriors = { k, startFrame: a, nFrames: b - a, ids: ids.slice(a * k, b * k), probs: probs.slice(a * k, b * k) };
    }
    out.push(e);
    a = b;
  }
  return out;
}

interface TruthWord {
  ref: WordRef | null;
  start: number;
  end: number;
}

const parseWord = (w: string): WordRef | null => {
  const m = /^(\d+):(\d+):(\d+)$/.exec(w);
  return m ? { surah: +m[1], ayah: +m[2], word: +m[3] } : null;
};
const ayahKey = (r: { surah: number; ayah: number }) => `${r.surah}:${r.ayah}`;
/** Not a different passage: within this many ayahs in Quran order, across a surah boundary too. Matches the tracker's local window. */
const LOCAL_AYAHS = 2;
const ordinals = new WeakMap<ReferenceIndex, Map<string, number>>();
const ayahOrdinal = (index: ReferenceIndex, r: WordRef) => {
  let m = ordinals.get(index);
  if (!m) {
    m = new Map();
    for (const w of index.words) if (!m.has(ayahKey(w))) m.set(ayahKey(w), m.size);
    ordinals.set(index, m);
  }
  return m.get(ayahKey(r)) ?? -1e9;
};

/** A lock is right when it names an ayah recited in the last `LAG_S` seconds, or starting within `LEAD_S`. */
const LAG_S = 4;
const LEAD_S = 0.5;
const SAMPLE_S = 0.25;
/** A start must be located within this many seconds for the headline rate. */
const WITHIN_S = 5;

export interface StartResult {
  word: string;
  atS: number;
  /** Seconds from the start word's onset to the first correct lock; null if never before the next start. */
  latencyS: number | null;
  /** A wrong lock was shown between the onset and the first correct lock. */
  falseLock: boolean;
  /** Words from the start before its reference text is unique in the Quran; 1 is unique from the first word. */
  distinguishWords: number;
  /** Seconds from the end of the distinguishing word to the first correct lock, for shared openings. */
  distinguishLatencyS: number | null;
}

export interface RecordingResult {
  id: string;
  style: string;
  content: string;
  noisy: boolean;
  audioS: number;
  processMs: number;
  /** Processing time over audio time; real time is 1. */
  rtf: number;
  starts: StartResult[];
  lockedSamples: number;
  wrongLockedSamples: number;
  scoredSamples: number;
  jumps: number;
  falseJumps: number;
  shortlists: number;
  shortlistHits: number;
}

export function benchRecording(index: ReferenceIndex, rec: QudRecording, options: TrackerOptions & { posteriors: boolean; chunkS?: number }): RecordingResult {
  const evidence = chunk(rec, options.chunkS ?? 1, options.posteriors);
  const tracker = createTracker(index, { ...options, trace: true });
  const events: PositionEvent[] = [];
  const keep = (es: EngineEvent[]) => {
    for (const e of es) if (e.type === "position") events.push(e);
  };
  const t0 = performance.now();
  for (const e of evidence) keep(tracker.push(e));
  keep(tracker.end());
  const processMs = performance.now() - t0;
  const audioS = (rec.nFrames * rec.frameMs) / 1000;

  const isLocal = (a: WordRef, b: WordRef) => Math.abs(ayahOrdinal(index, a) - ayahOrdinal(index, b)) <= LOCAL_AYAHS;
  const words: TruthWord[] = rec.truth.words.map((w) => ({ ref: parseWord(w.word), start: w.start_s, end: w.end_s }));
  const quran = words.filter((w): w is TruthWord & { ref: WordRef } => w.ref !== null);
  const specialNear = (t: number) => words.some((w) => !w.ref && w.start <= t + LEAD_S && w.end >= t - 1);
  /** Ayahs recited around `t`; during a basmala or isti'adha, also the ayah that follows it. */
  const recitedNear = (t: number) => {
    const near = new Set(quran.filter((w) => w.start <= t + LEAD_S && w.end >= t - LAG_S).map((w) => ayahKey(w.ref)));
    if (specialNear(t)) {
      const next = quran.find((w) => w.start > t);
      if (next) near.add(ayahKey(next.ref));
    }
    return near;
  };
  const inNonQuran = (t: number) => rec.truth.non_quran.some((r) => r.start_s <= t && t <= r.end_s);
  const basmalaLock = (r: WordRef) => (r.surah === 1 && r.ayah === 1) || (r.surah === 27 && r.ayah === 30);

  // State at each sample time: the latest event at or before it.
  const samples: { t: number; e: PositionEvent | null }[] = [];
  let k = -1;
  for (let t = 0; t <= audioS; t += SAMPLE_S) {
    while (k + 1 < events.length && events[k + 1].timeMs / 1000 <= t) k++;
    samples.push({ t, e: k >= 0 ? events[k] : null });
  }
  const verdict = (s: (typeof samples)[number]): "right" | "wrong" | "unscored" | "unlocked" => {
    const near = recitedNear(s.t);
    if (!near.size || inNonQuran(s.t)) return "unscored";
    if (!s.e || s.e.state !== "locked") return "unlocked";
    const w = s.e.observed[0].word;
    if (near.has(ayahKey(w))) return "right";
    return basmalaLock(w) && specialNear(s.t) ? "unscored" : "wrong";
  };
  const verdicts = samples.map(verdict);

  const startIdx: number[] = [];
  quran.forEach((w, i) => {
    if (i === 0 || !isLocal(quran[i - 1].ref, w.ref)) startIdx.push(i);
  });
  const starts = startIdx.map((i, n): StartResult => {
    const at = quran[i].start;
    const until = n + 1 < startIdx.length ? quran[startIdx[n + 1]].start : Infinity;
    let latencyS: number | null = null;
    for (let j = 0; j < samples.length; j++) {
      const s = samples[j];
      if (s.t < at || s.t >= until) continue;
      const e = s.e;
      if (e?.state === "locked" && verdicts[j] === "right" && isLocal(e.observed[0].word, quran[i].ref)) {
        latencyS = s.t - at;
        break;
      }
    }
    // Still showing the previous passage is a stale bookmark, not a false lock: only a new wrong location counts.
    const previous = i > 0 ? quran[i - 1].ref : null;
    const settled = latencyS === null ? until : at + latencyS;
    const falseLock = events.some((e) => {
      const t = e.timeMs / 1000;
      if (t < at || t >= settled || e.state !== "locked") return false;
      const w = e.observed[0].word;
      return !isLocal(w, quran[i].ref) && !(previous && isLocal(w, previous)) && !(basmalaLock(w) && specialNear(t));
    });
    const distinguishWords = wordsToUnique(index, quran[i].ref);
    const distinguishing = quran[Math.min(quran.length - 1, i + distinguishWords - 1)];
    return {
      word: `${quran[i].ref.surah}:${quran[i].ref.ayah}:${quran[i].ref.word}`,
      atS: at,
      latencyS,
      falseLock,
      distinguishWords,
      distinguishLatencyS: latencyS === null || distinguishWords <= 1 ? null : Math.max(0, at + latencyS - distinguishing.end),
    };
  });

  let jumps = 0;
  let falseJumps = 0;
  let prev: WordRef | null = null;
  let shortlists = 0;
  let shortlistHits = 0;
  for (const e of events) {
    const t = e.timeMs / 1000;
    const near = recitedNear(t);
    const list = e.trace?.shortlist ?? [];
    if (list.length && near.size && !inNonQuran(t)) {
      shortlists++;
      if (list.some((c) => [...near].some((a) => covers(c.span.start, c.span.end, a)))) shortlistHits++;
    }
    if (e.state !== "locked") continue;
    const w = e.observed[0].word;
    if (prev && !isLocal(prev, w)) {
      jumps++;
      if (near.size && !near.has(ayahKey(w)) && !(basmalaLock(w) && specialNear(t))) falseJumps++;
    }
    prev = w;
  }

  return {
    id: rec.id,
    style: rec.meta.style,
    content: rec.meta.content,
    noisy: rec.meta.noisy,
    audioS,
    processMs,
    rtf: processMs / 1000 / audioS,
    starts,
    lockedSamples: verdicts.filter((v) => v === "right" || v === "wrong").length,
    wrongLockedSamples: verdicts.filter((v) => v === "wrong").length,
    scoredSamples: verdicts.filter((v) => v !== "unscored").length,
    jumps,
    falseJumps,
    shortlists,
    shortlistHits,
  };
}

const covers = (start: WordRef, end: WordRef, ayah: string) => {
  const [s, a] = ayah.split(":").map(Number);
  const at = s * 1000 + a;
  return start.surah * 1000 + start.ayah <= at && at <= end.surah * 1000 + end.ayah;
};

/** How many words from `from`, read on in reference order, until their phonemes occur exactly once in the Quran. */
export function wordsToUnique(index: ReferenceIndex, from: WordRef, max = 40): number {
  const i = index.words.findIndex((w) => w.surah === from.surah && w.ayah === from.ayah && w.word === from.word);
  if (i === -1) return 1;
  for (let n = 1; n <= max && i + n <= index.words.length; n++) {
    const text = index.phonemes.slice(index.words[i].start, index.words[i + n - 1].end);
    if (!text) continue;
    const first = index.phonemes.indexOf(text);
    if (index.phonemes.indexOf(text, first + 1) === -1) return n;
  }
  return max;
}

export interface BenchSummary {
  recordings: number;
  audioMinutes: number;
  starts: number;
  /** Starts located at the right ayah within 5 s of their first word. */
  within5s: number;
  /** Starts with a wrong lock shown before the right one. */
  falseLockStarts: number;
  /** Starts never located before the next start. */
  unresolved: number;
  /** Locked time that named the wrong ayah, over all locked scored time. */
  falseLockTimeShare: number;
  jumps: number;
  falseJumps: number;
  shortlistRecall: number;
  /** Median seconds from each recording's first Quran word to its first correct lock. */
  medianTimeToFirstLockS: number | null;
  sharedOpenings: number;
  /** Median seconds from the distinguishing word's end to the lock, over shared openings that locked. */
  medianDistinguishLatencyS: number | null;
  maxRtf: number;
  meanRtf: number;
}

const median = (xs: number[]) => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};

export function summarise(results: RecordingResult[]): BenchSummary {
  const starts = results.flatMap((r) => r.starts);
  const shared = starts.filter((s) => s.distinguishWords > 1);
  const locked = results.reduce((a, r) => a + r.lockedSamples, 0);
  const shortlists = results.reduce((a, r) => a + r.shortlists, 0);
  return {
    recordings: results.length,
    audioMinutes: results.reduce((a, r) => a + r.audioS, 0) / 60,
    starts: starts.length,
    within5s: starts.filter((s) => s.latencyS !== null && s.latencyS <= WITHIN_S).length,
    falseLockStarts: starts.filter((s) => s.falseLock).length,
    unresolved: starts.filter((s) => s.latencyS === null).length,
    falseLockTimeShare: locked ? results.reduce((a, r) => a + r.wrongLockedSamples, 0) / locked : 0,
    jumps: results.reduce((a, r) => a + r.jumps, 0),
    falseJumps: results.reduce((a, r) => a + r.falseJumps, 0),
    shortlistRecall: shortlists ? results.reduce((a, r) => a + r.shortlistHits, 0) / shortlists : 1,
    medianTimeToFirstLockS: median(results.map((r) => r.starts[0]?.latencyS).filter((x): x is number => x != null)),
    sharedOpenings: shared.length,
    medianDistinguishLatencyS: median(shared.map((s) => s.distinguishLatencyS).filter((x): x is number => x !== null)),
    maxRtf: Math.max(...results.map((r) => r.rtf)),
    meanRtf: results.reduce((a, r) => a + r.rtf, 0) / results.length,
  };
}
