// Picks the few moments of a take worth asking about. Pure: no DOM, no clock, injectable randomness.
//
// Priority: confirmed warnings, then the reader's own bookmarks, then suspected/uncertain warnings, then moments
// where we lost or moved the reader's place. A warning only says the engine raised one; it is a question for the
// reader, never an answer. 1-2 regions with no warning and no bookmark are added at random as the bias sample:
// they are asked about like everything else and are never assumed to be clean.
import type { Incident } from "@tilawa/core/report";
import type { CheckerEvent, EngineEvent, PositionEvent, WordRef } from "./types";

export type MomentKind = "warning" | "stumble" | "position" | "sample" | "marked";
/** Stable, language-independent question ids; they are what `claim.question` stores. */
export type QuestionId = "made_mistake" | "wrong_ayah" | "reading_correct";

export interface Moment {
  /** Local id, unique within a take. */
  id: string;
  kind: MomentKind;
  question: QuestionId;
  /** Anchor on the session clock (ms). */
  atMs: number;
  /** The clip the reader hears, on the session clock, inside [0, durationMs]. */
  clip: { startMs: number; endMs: number };
  /** The word to show and highlight; null when the engine never placed the reader. */
  location: WordRef | null;
  /** Checker event ids behind this moment (warnings only). */
  eventIds: string[];
  /** Ids of the reader's bookmarks (stumble incidents) folded into this moment. */
  bookmarkIds: string[];
  /** Warnings: whether the engine confirmed it (vs suspected/uncertain). */
  confirmed?: boolean;
  errorClass?: string;
}

export interface SelectOptions {
  /** Length of the take on the session clock. */
  durationMs: number;
  /** Most moments to ask about, samples included. Default 5. */
  max?: number;
  /** Unflagged samples wanted (0-2). Default 2, or 1 when flagged moments would otherwise be cut. */
  samples?: number;
  /** Uniform [0, 1). Default Math.random. */
  rng?: () => number;
  /** Clip context around an anchor. Defaults 6000 before, 2000 after. */
  beforeMs?: number;
  afterMs?: number;
  /** Sample clip length. Default 8000. */
  sampleMs?: number;
  /** Candidates this close to an already chosen moment are folded into it. Default 3000. */
  mergeMs?: number;
  /** Clear audio a sample keeps from any warning or bookmark. Default 3000. */
  sampleGapMs?: number;
}

const QUESTION: Record<MomentKind, QuestionId> = {
  warning: "made_mistake",
  stumble: "made_mistake",
  marked: "made_mistake",
  position: "wrong_ayah",
  sample: "reading_correct",
};

export function questionFor(kind: MomentKind): QuestionId {
  return QUESTION[kind];
}

interface FoldedWarning {
  id: string;
  firstMs: number;
  final: CheckerEvent["state"];
  location: WordRef;
  errorClass: string;
}

/** One entry per checker id, in first-seen order, with its final state. */
export function foldWarnings(events: readonly EngineEvent[]): FoldedWarning[] {
  const byId = new Map<string, FoldedWarning>();
  for (const e of events) {
    if (e.type !== "checker") continue;
    const w = byId.get(e.id);
    if (!w) byId.set(e.id, { id: e.id, firstMs: e.timeMs, final: e.state, location: e.location, errorClass: e.errorClass });
    else {
      w.final = e.state;
      w.location = e.location;
      w.errorClass = e.errorClass;
    }
  }
  return [...byId.values()];
}

function positions(events: readonly EngineEvent[]): PositionEvent[] {
  return events.filter((e): e is PositionEvent => e.type === "position").sort((a, b) => a.timeMs - b.timeMs);
}

/** The word we were showing at `t`: the latest placed position event at or before it, else the first after it. */
export function wordAt(events: readonly EngineEvent[], t: number): WordRef | null {
  let best: WordRef | null = null;
  let after: WordRef | null = null;
  for (const p of positions(events)) {
    const w = p.observed[0]?.word;
    if (!w) continue;
    if (p.timeMs <= t) best = w;
    else if (!after) after = w;
  }
  return best ?? after;
}

/** Moments where the tracker moved or lost the reader after having placed them. */
function positionTroubles(events: readonly EngineEvent[]): { atMs: number; location: WordRef | null }[] {
  const out: { atMs: number; location: WordRef | null }[] = [];
  let shown: WordRef | null = null;
  let wasLocked = false;
  for (const p of positions(events)) {
    if (p.relocated) out.push({ atMs: p.timeMs, location: p.relocated.from });
    else if (wasLocked && (p.state === "lost" || p.state === "ambiguous")) out.push({ atMs: p.timeMs, location: shown });
    if (p.state === "locked") wasLocked = true;
    else if (p.state === "lost") wasLocked = false;
    if (p.observed[0]) shown = p.observed[0].word;
  }
  // One per 10 s is plenty: a wobble emits several events.
  return out.filter((x, i) => i === 0 || x.atMs - out[i - 1].atMs >= 10_000);
}

function clamp(v: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, v));
}

export function clipAround(atMs: number, durationMs: number, beforeMs = 6000, afterMs = 2000) {
  const startMs = Math.floor(clamp(atMs - beforeMs, 0, durationMs));
  const endMs = Math.ceil(clamp(atMs + afterMs, 0, durationMs));
  return { startMs, endMs: Math.max(startMs, endMs) };
}

/** A moment the reader marked while listening back (playback time on the session clock). */
export function markedMoment(id: string, atMs: number, events: readonly EngineEvent[], durationMs: number): Moment {
  return {
    id,
    kind: "marked",
    question: "made_mistake",
    atMs,
    clip: clipAround(atMs, durationMs),
    location: wordAt(events, atMs),
    eventIds: [],
    bookmarkIds: [],
  };
}

export function selectMoments(events: readonly EngineEvent[], bookmarks: readonly Incident[], options: SelectOptions): Moment[] {
  const { durationMs } = options;
  const max = Math.max(0, options.max ?? 5);
  const rng = options.rng ?? Math.random;
  const beforeMs = options.beforeMs ?? 6000;
  const afterMs = options.afterMs ?? 2000;
  const sampleMs = options.sampleMs ?? 8000;
  const mergeMs = options.mergeMs ?? 3000;
  const gapMs = options.sampleGapMs ?? 3000;
  if (durationMs <= 0 || max === 0) return [];

  const warnings = foldWarnings(events).filter((w) => w.final !== "resolved");
  type Candidate = Omit<Moment, "id" | "clip" | "question">;
  const confirmed: Candidate[] = [];
  const unsure: Candidate[] = [];
  for (const w of warnings) {
    const c: Candidate = {
      kind: "warning",
      atMs: w.firstMs,
      location: w.location,
      eventIds: [w.id],
      bookmarkIds: [],
      confirmed: w.final === "confirmed",
      errorClass: w.errorClass,
    };
    (c.confirmed ? confirmed : unsure).push(c);
  }
  const stumbles: Candidate[] = bookmarks
    .filter((b) => b.kind === "stumble")
    .map((b) => ({ kind: "stumble", atMs: b.markedAt.sessionMs, location: wordAt(events, b.markedAt.sessionMs), eventIds: [], bookmarkIds: [b.id] }));
  const troubles: Candidate[] = positionTroubles(events).map((t) => ({ kind: "position", atMs: t.atMs, location: t.location, eventIds: [], bookmarkIds: [] }));
  const byTime = (a: Candidate, b: Candidate) => a.atMs - b.atMs;
  const ordered = [...confirmed.sort(byTime), ...stumbles.sort(byTime), ...unsure.sort(byTime), ...troubles.sort(byTime)];

  // Fold near-duplicates into the higher-priority moment, so one stumble is asked about once.
  const flagged: Candidate[] = [];
  for (const c of ordered) {
    const near = flagged.find((f) => Math.abs(f.atMs - c.atMs) <= mergeMs);
    if (near) {
      near.eventIds.push(...c.eventIds.filter((id) => !near.eventIds.includes(id)));
      near.bookmarkIds.push(...c.bookmarkIds);
    } else flagged.push({ ...c, eventIds: [...c.eventIds], bookmarkIds: [...c.bookmarkIds] });
  }

  const wantSamples = clamp(options.samples ?? (flagged.length >= max - 1 ? 1 : 2), 0, Math.min(2, max));
  const chosen = flagged.slice(0, Math.max(0, max - wantSamples));
  const moments: Moment[] = chosen.map((c) => ({ ...c, question: QUESTION[c.kind], clip: clipAround(c.atMs, durationMs, beforeMs, afterMs) }) as Moment);

  // Busy spans: every warning (resolved ones too) and bookmark, plus every chosen clip.
  const busy: { startMs: number; endMs: number }[] = [
    ...foldWarnings(events).map((w) => ({ startMs: w.firstMs - beforeMs - gapMs, endMs: w.firstMs + afterMs + gapMs })),
    ...bookmarks.map((b) => ({ startMs: b.markedAt.sessionMs - beforeMs - gapMs, endMs: b.markedAt.sessionMs + afterMs + gapMs })),
    ...moments.map((m) => ({ startMs: m.clip.startMs, endMs: m.clip.endMs })),
  ];
  const placed = positions(events).filter((p) => p.observed.length > 0).map((p) => p.timeMs);
  for (let n = 0; n < wantSamples && moments.length < max; n++) {
    const s = pickSample(durationMs, sampleMs, busy, placed, rng);
    if (s === null) break;
    const clip = { startMs: s, endMs: Math.min(durationMs, s + sampleMs) };
    busy.push(clip);
    moments.push({
      id: "",
      kind: "sample",
      question: "reading_correct",
      atMs: Math.round((clip.startMs + clip.endMs) / 2),
      clip,
      location: wordAt(events, (clip.startMs + clip.endMs) / 2),
      eventIds: [],
      bookmarkIds: [],
    });
  }

  // Ask in the order they happened: it is easier to follow a recording forwards.
  moments.sort((a, b) => a.clip.startMs - b.clip.startMs);
  return moments.map((m, i) => ({ ...m, id: `m${i + 1}` }));
}

/**
 * A random sample start on a 250 ms grid, clear of every busy span. Starts where the engine had placed the reader
 * are preferred (we want recitation, not silence); if there are none, any clear start will do.
 */
function pickSample(durationMs: number, sampleMs: number, busy: { startMs: number; endMs: number }[], placed: number[], rng: () => number): number | null {
  const len = Math.min(sampleMs, durationMs);
  if (len < 2000) return null;
  const clear: number[] = [];
  const voiced: number[] = [];
  for (let s = 0; s + len <= durationMs; s += 250) {
    const e = s + len;
    if (busy.some((b) => s < b.endMs && e > b.startMs)) continue;
    clear.push(s);
    if (placed.some((t) => t >= s && t <= e)) voiced.push(s);
  }
  const pool = voiced.length ? voiced : clear;
  if (!pool.length) return null;
  return pool[Math.min(pool.length - 1, Math.floor(rng() * pool.length))];
}
