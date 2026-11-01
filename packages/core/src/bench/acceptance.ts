/** Acceptance scoring for one labelled clip: checker events against planted errors, and the tracker's start. */
import { createChecker } from "../checker.js";
import type { CheckerPolicy } from "../checker.js";
import type { CheckerEvent, EngineEvent, PositionEvent, WordRef } from "../events.js";
import { EVENT_SCHEMA_VERSION } from "../events.js";
import type { PhraseAsset, ReferenceIndex } from "../reference.js";
import { createTracker, trackerInput } from "../tracker.js";
import { chunk } from "./index.js";
import type { QudRecording } from "./index.js";

/** `free`: the checker follows the tracker's lock (Decision 3). `declared`: the session declares the prompted passage. */
export type Arm = "free" | "declared";

export interface LabelEvent {
  errorClass: string;
  state: string;
  location: WordRef;
  /** The recitation with this error applied fits another ayah at least as well as the prompted one (a mutashabihat slip). */
  forms_other_ayah?: boolean;
}

export interface AcceptanceClip {
  id: string;
  /** The prompted ayahs, as "surah:ayah", in recitation order. */
  ayahs: string[];
  /** The final state a correct engine reaches for each planted error. */
  events: LabelEvent[];
  /** Planted repeats: correct recitation, so a confirmed event here is a false positive. */
  nonErrors: WordRef[];
  evidence: Pick<QudRecording, "inventory" | "frameMs" | "nFrames" | "phonemes" | "posteriors">;
}

export interface ClipScore {
  /** Each checker event's final state. */
  final: CheckerEvent[];
  confirmed: CheckerEvent[];
  hits: number;
  misses: LabelEvent[];
  falsePositives: CheckerEvent[];
  /** False positives on a planted repeat. */
  repeatFalsePositives: number;
  /** Labelled suspected errors, and how many of them the checker raised at all. */
  pending: number;
  pendingHits: number;
  gate: StartGate;
}

export interface StartGate {
  /** First decoded phoneme: the reciter's start. */
  onsetMs: number;
  /** First lock on a prompted ayah, from the start of the clip. */
  firstCorrectLockMs: number | null;
  correctWithin5s: boolean;
  /** A lock outside the prompted passage was shown before the first correct one. */
  wrongLockFirst: boolean;
  /** Any lock outside the prompted passage, at any time. */
  outsideLock: boolean;
  jumps: number;
  falseJumps: number;
}

export const MEMORIZATION_CLASSES = ["wrong_word", "omitted_word", "added_word", "skipped_passage"];
/** Within this many ayahs of the prompted passage a lock is local, as in the QUD bench. */
const LOCAL_AYAHS = 2;
const WITHIN_MS = 5000;

const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const ayahKey = (r: { surah: number; ayah: number }) => `${r.surah}:${r.ayah}`;
const signature = (e: { errorClass: string; location: WordRef }) => `${e.errorClass}/${key(e.location)}`;

/** The checker gets the clip's posteriors (for its local-window check); the tracker gets the decode only, as before. */
export function scoreClip(index: ReferenceIndex, clip: AcceptanceClip, o: { arm: Arm; phrases: PhraseAsset; chunkS?: number; policy?: CheckerPolicy }): ClipScore {
  const tracker = createTracker(index, { mode: "memorization", phrases: o.phrases, acquireMs: 1000 });
  const checker = createChecker(index, { phrases: o.phrases, policy: o.policy, ...(o.arm === "declared" ? { passage: declaredPassage(index, clip.ayahs) } : {}) });
  const checks: CheckerEvent[] = [];
  const positions: PositionEvent[] = [];
  const feed = (out: EngineEvent[]): EngineEvent[] => {
    positions.push(...out.filter((x): x is PositionEvent => x.type === "position"));
    return out;
  };
  const posteriors = !!clip.evidence.posteriors?.ids.length;
  for (const e of chunk(clip.evidence as QudRecording, o.chunkS ?? 0.32, posteriors)) {
    checks.push(...checker.push(e, feed(tracker.push(trackerInput(e)))));
  }
  checks.push(...checker.end(feed(tracker.end())));

  const latest = new Map<string, CheckerEvent>();
  for (const e of checks) latest.set(e.id, e);
  const confirmed = [...latest.values()].filter((e) => e.state === "confirmed");
  const expected = clip.events.filter((e) => MEMORIZATION_CLASSES.includes(e.errorClass));
  const wanted = expected.filter((e) => e.state === "confirmed");
  const pending = expected.filter((e) => e.state === "suspected");
  const used = new Set<number>();
  const misses = wanted.filter((e) => {
    const i = confirmed.findIndex((p, j) => !used.has(j) && signature(p) === signature(e));
    if (i < 0) return true;
    used.add(i);
    return false;
  });
  // A labelled skip may legitimately stay suspected; confirming it is not a false positive either.
  const pendingSignatures = new Set(pending.map(signature));
  const falsePositives = confirmed.filter((p, j) => !used.has(j) && !pendingSignatures.has(signature(p)));
  const repeats = new Set(clip.nonErrors.map(key));
  return {
    final: [...latest.values()],
    confirmed,
    hits: used.size,
    misses,
    falsePositives,
    repeatFalsePositives: falsePositives.filter((p) => repeats.has(key(p.location))).length,
    pending: pending.length,
    pendingHits: pending.filter((e) => [...latest.values()].some((p) => signature(p) === signature(e))).length,
    gate: startGate(index, clip, positions),
  };
}

/** The prompted passage's first and last words. */
function declaredPassage(index: ReferenceIndex, ayahs: string[]): { from: WordRef; to: WordRef } {
  const inAyah = (a: string) => index.words.filter((w) => ayahKey(w) === a && w.end > w.start);
  const ref = (w: WordRef): WordRef => ({ surah: w.surah, ayah: w.ayah, word: w.word });
  return { from: ref(inAyah(ayahs[0])[0]), to: ref(inAyah(ayahs.at(-1)!).at(-1)!) };
}

function startGate(index: ReferenceIndex, clip: AcceptanceClip, positions: PositionEvent[]): StartGate {
  const ordinal = new Map<string, number>();
  for (const w of index.words) if (!ordinal.has(ayahKey(w))) ordinal.set(ayahKey(w), ordinal.size);
  const prompted = clip.ayahs.map((a) => ordinal.get(a) ?? -1e9);
  const local = (r: WordRef) => prompted.some((p) => Math.abs((ordinal.get(ayahKey(r)) ?? 1e9) - p) <= LOCAL_AYAHS);
  const isLocal = (a: WordRef, b: WordRef) => Math.abs((ordinal.get(ayahKey(a)) ?? 0) - (ordinal.get(ayahKey(b)) ?? 0)) <= LOCAL_AYAHS;
  const locks = positions.filter((p) => p.state === "locked" && p.observed[0]);
  const firstCorrect = locks.find((p) => clip.ayahs.includes(ayahKey(p.observed[0].word)));
  const onsetMs = (clip.evidence.phonemes[0]?.start ?? 0) * clip.evidence.frameMs;
  let jumps = 0;
  let falseJumps = 0;
  let prev: WordRef | null = null;
  for (const p of locks) {
    const w = p.observed[0].word;
    if (prev && !isLocal(prev, w)) {
      jumps++;
      if (!local(w)) falseJumps++;
    }
    prev = w;
  }
  return {
    onsetMs,
    firstCorrectLockMs: firstCorrect?.timeMs ?? null,
    correctWithin5s: !!firstCorrect && firstCorrect.timeMs - onsetMs <= WITHIN_MS,
    wrongLockFirst: locks.some((p) => (!firstCorrect || p.timeMs < firstCorrect.timeMs) && !local(p.observed[0].word)),
    outsideLock: locks.some((p) => !local(p.observed[0].word)),
    jumps,
    falseJumps,
  };
}

export type Slice = "unambiguous" | "mutashabihat";

/** Memorization positives and hits per slice; every false positive counts against the unambiguous slice. */
export function slices(clip: AcceptanceClip, score: ClipScore): Record<Slice, { pos: number; tp: number; fp: number }> {
  const slice = (e: LabelEvent): Slice => (e.forms_other_ayah ? "mutashabihat" : "unambiguous");
  const out = { unambiguous: { pos: 0, tp: 0, fp: score.falsePositives.length }, mutashabihat: { pos: 0, tp: 0, fp: 0 } };
  for (const e of clip.events) if (e.state === "confirmed" && MEMORIZATION_CLASSES.includes(e.errorClass)) out[slice(e)].pos++;
  for (const s of ["unambiguous", "mutashabihat"] as const) out[s].tp = out[s].pos - score.misses.filter((e) => slice(e) === s).length;
  return out;
}

export type MissCause = "evidence" | "tracker" | "checker";

/**
 * E0: why a planted error was missed. `tracker` when the declared-passage arm catches it; otherwise `evidence` when the
 * teacher's decode shows the prompted word as said, and `checker` when the decode shows the error but it wasn't confirmed.
 */
export function attribute(index: ReferenceIndex, clip: AcceptanceClip, free: ClipScore, declared: ClipScore): { miss: LabelEvent; cause: MissCause }[] {
  const caught = new Set(declared.misses.map(signature));
  return free.misses.map((miss) => ({
    miss,
    cause: !caught.has(signature(miss)) ? "tracker" : visibleInEvidence(index, clip, miss.location) ? "checker" : "evidence",
  }));
}

const CONSONANTS = new Set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي");
/** Below this share of a word's consonants decoded as themselves, the decode shows the word was not said as written. */
const KEPT = 0.5;

/** Whether the greedy decode, aligned to the prompted passage, departs from the canonical word at `at`. */
export function visibleInEvidence(index: ReferenceIndex, clip: AcceptanceClip, at: WordRef): boolean {
  const ws = index.words.filter((w) => clip.ayahs.includes(ayahKey(w)));
  const lo = ws[0].start;
  const ref = index.phonemes.slice(lo, ws.at(-1)!.end);
  const obs = clip.evidence.phonemes.map((p) => p.p).join("");
  const aligned = alignRef(ref, obs);
  const w = ws.find((x) => key(x) === key(at));
  if (!w) return true;
  let consonants = 0;
  let kept = 0;
  for (let i = w.start - lo; i < w.end - lo; i++) {
    if (!CONSONANTS.has(ref[i])) continue;
    consonants++;
    if (aligned[i] !== null && obs[aligned[i]!] === ref[i]) kept++;
  }
  return consonants > 0 && kept / consonants < KEPT;
}

/** Unit-cost semi-global alignment: for each reference index, the observed index it aligns to, or null. */
export function alignRef(ref: string, obs: string): (number | null)[] {
  const n = ref.length;
  const m = obs.length;
  const d = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  // Leading and trailing observed phonemes (isti'adha, basmala, noise) are free.
  for (let i = 1; i <= n; i++) d[i][0] = i;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      d[i][j] = Math.min(d[i - 1][j - 1] + (ref[i - 1] === obs[j - 1] ? 0 : 1), d[i - 1][j] + 1, d[i][j - 1] + 1);
    }
  }
  let j = 0;
  for (let k = 1; k <= m; k++) if (d[n][k] < d[n][j]) j = k;
  const out: (number | null)[] = new Array(n).fill(null);
  let i = n;
  while (i > 0) {
    if (j > 0 && d[i][j] === d[i - 1][j - 1] + (ref[i - 1] === obs[j - 1] ? 0 : 1)) {
      out[--i] = --j;
    } else if (d[i][j] === d[i - 1][j] + 1) i--;
    else if (j > 0) j--;
    else i--;
  }
  return out;
}
