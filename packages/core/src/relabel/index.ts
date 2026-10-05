import type { TrackerState, WordSpan } from "../events.js";
import type { PhonemeEvidence } from "../evidence.js";
import type { ReferenceIndex } from "../reference.js";
import { finalPosition } from "../fixtures/index.js";
import { createTracker } from "../tracker.js";

/** One row of tlog pass1 output, as written by `tools/export_pass1.py`. */
export interface Pass1Clip {
  shard: string;
  row: number;
  path: string;
  status: "ok" | "decode_failed" | "too_long";
  filenameLabel: { surah: number; ayah: number } | null;
  /** Null unless `status` is "ok". */
  evidence: PhonemeEvidence | null;
}

export type Exclusion = "ambiguous" | "lost" | "insufficient_evidence" | "decode_failed" | "too_long";

export interface RelabelCandidate {
  span: WordSpan;
  cost: number;
  per: number;
}

/** A proposed label for review. Never a training label until accepted by 07. */
export interface ProposedLabel {
  labelKind: "proposed";
  shard: string;
  row: number;
  path: string;
  filenameLabel: { surah: number; ayah: number } | null;
  /** Set only when `state` is "locked". */
  proposed: WordSpan | null;
  /** Null when the tracker never ran or emitted nothing. */
  state: TrackerState | null;
  confidence: number | null;
  fit: number | null;
  separation: number | null;
  /** Plain edits the top candidate leads the best other fitting one by; null when uncontested. */
  leadEdits: number | null;
  cost: number | null;
  per: number | null;
  /** Distinct locations from the shortlist, best first by alignment cost (the prior is not applied here). */
  candidates: RelabelCandidate[];
  /** Exactly one reason for every clip that isn't locked; null when locked. */
  exclusion: Exclusion | null;
  /** How the proposed span relates to the filename label; null when nothing is proposed. */
  pattern: string | null;
  agrees: boolean | null;
}

export interface RelabelOptions {
  /** Pass top-k posteriors to the tracker; they're dropped otherwise. */
  posteriors?: boolean;
  /** Filename-ayah prior weight in 0..1. Default 1. */
  priorWeight?: number;
  maxCandidates?: number;
}

export function offsetPattern(label: { surah: number; ayah: number } | null, span: WordSpan, lastWordOf: (s: number, a: number) => number): string {
  const { start, end } = span;
  if (!label) return "no filename label";
  if (start.surah !== end.surah) return "crosses surahs";
  if (start.surah !== label.surah) return "other surah";
  const lo = start.ayah - label.ayah;
  const hi = end.ayah - label.ayah;
  const sign = (d: number) => (d >= 0 ? `+${d}` : `${d}`);
  if (lo !== hi) return `contains ${hi - lo + 1} ayahs (${sign(lo)}..${sign(hi)})`;
  if (lo !== 0) return `off by ${sign(lo)} ayah`;
  return start.word === 1 && end.word === lastWordOf(end.surah, end.ayah) ? "filename ayah" : "part of filename ayah";
}

export function relabelClip(index: ReferenceIndex, clip: Pass1Clip, options: RelabelOptions = {}): ProposedLabel {
  const base = {
    labelKind: "proposed" as const,
    shard: clip.shard,
    row: clip.row,
    path: clip.path,
    filenameLabel: clip.filenameLabel,
  };
  const empty = { proposed: null, state: null, confidence: null, fit: null, separation: null, leadEdits: null, cost: null, per: null, candidates: [], pattern: null, agrees: null };
  if (clip.status !== "ok" || !clip.evidence) return { ...base, ...empty, exclusion: clip.status === "too_long" ? "too_long" : "decode_failed" };

  const label = clip.filenameLabel;
  const tracker = createTracker(index, {
    trace: true,
    ...(label ? { prior: { ...label, weight: options.priorWeight ?? 1 } } : {}),
  });
  const { posteriors: _, ...decoded } = clip.evidence;
  const final = finalPosition([...tracker.push(options.posteriors ? clip.evidence : decoded), ...tracker.end()]);
  if (!final) return { ...base, ...empty, exclusion: "insufficient_evidence" };

  const shortlist = final.trace!.shortlist;
  const distinct: typeof shortlist = [];
  for (const c of shortlist) if (!distinct.some((k) => c.offset <= k.offset + k.length && k.offset <= c.offset + c.length)) distinct.push(c);
  const top = final.observed[0];
  const decision = final.trace!.decision;
  const proposed = final.state === "locked" ? top.span : null;
  const lastWordOf = (s: number, a: number) => {
    let n = 0;
    for (const w of index.words) if (w.surah === s && w.ayah === a) n = Math.max(n, w.word);
    return n;
  };
  const pattern = proposed ? offsetPattern(label, proposed, lastWordOf) : null;
  const exclusion: Exclusion | null =
    final.state === "locked" ? null : final.state === "ambiguous" ? "ambiguous" : final.state === "lost" ? "lost" : "insufficient_evidence";

  return {
    ...base,
    proposed,
    state: final.state,
    confidence: top?.confidence ?? null,
    fit: decision?.fit ?? null,
    separation: decision?.separation ?? null,
    leadEdits: decision && Number.isFinite(decision.leadEdits) ? decision.leadEdits : null,
    cost: top?.cost ?? null,
    per: top?.per ?? null,
    candidates: distinct.slice(0, options.maxCandidates ?? 5).map((c) => ({ span: c.span, cost: c.cost, per: c.per })),
    exclusion,
    pattern,
    agrees: pattern === null ? null : pattern === "filename ayah" || pattern === "part of filename ayah",
  };
}

export interface RelabelSummary {
  labelKind: "proposed";
  note: string;
  clips: number;
  /** "locked" or the exclusion reason. */
  outcomes: Record<string, number>;
  /** Of locked clips, how many lie within their filename ayah. */
  agreement: { locked: number; agree: number; rate: number };
  /** Offset patterns of locked clips, most common first. */
  patterns: Record<string, number>;
}

const countSorted = (keys: string[]): Record<string, number> => {
  const m = new Map<string, number>();
  for (const k of keys) m.set(k, (m.get(k) ?? 0) + 1);
  return Object.fromEntries([...m].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)));
};

export function summarize(labels: ProposedLabel[]): RelabelSummary {
  const locked = labels.filter((l) => l.exclusion === null);
  const agree = locked.filter((l) => l.agrees).length;
  return {
    labelKind: "proposed",
    note: "Proposed labels from a dry run. Not training labels: the training data is unchanged, and acceptance happens in ticket 07.",
    clips: labels.length,
    outcomes: countSorted(labels.map((l) => l.exclusion ?? "locked")),
    agreement: { locked: locked.length, agree, rate: locked.length ? agree / locked.length : 0 },
    patterns: countSorted(locked.map((l) => l.pattern!)),
  };
}
