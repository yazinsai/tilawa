import type { WordRef, WordSpan } from "../events.js";
import type { ProposedLabel } from "../relabel/index.js";

/**
 * Strata, assigned in this order so each clip lands in exactly one. Boundary and disagreement clips come first: they
 * decide the thresholds, and an outcome stratum would otherwise swallow them.
 */
export const STRATA = [
  "near_threshold",
  "posteriors_disagree",
  "agrees",
  "part_of_ayah",
  "offset",
  "multi_ayah",
  "other_surah",
  "ambiguous",
  "lost",
  "insufficient",
] as const;
export type Stratum = (typeof STRATA)[number];

/** Clips drawn per stratum by default; a stratum smaller than its quota is taken whole. About 300 in total. */
export const DEFAULT_QUOTA: Record<Stratum, number> = {
  near_threshold: 25,
  posteriors_disagree: 20,
  agrees: 60,
  part_of_ayah: 20,
  offset: 40,
  multi_ayah: 45,
  other_surah: 22,
  ambiguous: 25,
  lost: 30,
  insufficient: 20,
};

/** Locked clips with confidence in this range, and unlocked ones that came this close, are boundary cases. */
export const NEAR_THRESHOLD = { lo: 0.04, hi: 0.2 };

const sameSpan = (a: WordSpan | null, b: WordSpan | null) => JSON.stringify(a) === JSON.stringify(b);

export function stratumOf(label: ProposedLabel, withPosteriors?: ProposedLabel): Stratum | null {
  if (label.state === null) return null;
  const c = label.confidence ?? 0;
  if (c >= NEAR_THRESHOLD.lo && c <= NEAR_THRESHOLD.hi) return "near_threshold";
  if (withPosteriors && (withPosteriors.state !== label.state || !sameSpan(withPosteriors.proposed, label.proposed))) return "posteriors_disagree";
  if (label.state === "ambiguous") return "ambiguous";
  if (label.state === "lost") return "lost";
  if (label.state !== "locked") return "insufficient";
  const p = label.pattern ?? "";
  if (p === "filename ayah") return "agrees";
  if (p === "part of filename ayah") return "part_of_ayah";
  if (p.startsWith("off by")) return "offset";
  if (p.startsWith("contains")) return "multi_ayah";
  return "other_surah";
}

export interface SampleClip {
  row: number;
  shard: string;
  path: string;
  stratum: Stratum;
}

export interface Sample {
  seed: number;
  /** How many clips of the whole run fall in each stratum, and how many were drawn; used to weight estimates. */
  strata: Record<Stratum, { population: number; sampled: number }>;
  /** In review order: shuffled so strata are mixed and nothing about the order hints at the answer. */
  clips: SampleClip[];
}

/** FNV-1a: a stable pseudo-random order that only depends on the seed and the row. */
export function hash(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193) >>> 0;
  return h;
}

export function drawSample(labels: ProposedLabel[], options: { seed?: number; quota?: Partial<Record<Stratum, number>>; withPosteriors?: ProposedLabel[] } = {}): Sample {
  const seed = options.seed ?? 1;
  const quota = { ...DEFAULT_QUOTA, ...options.quota };
  const post = new Map((options.withPosteriors ?? []).map((l) => [`${l.shard}:${l.row}`, l]));
  const by = new Map<Stratum, ProposedLabel[]>(STRATA.map((s) => [s, []]));
  for (const l of labels) {
    const s = stratumOf(l, options.withPosteriors ? post.get(`${l.shard}:${l.row}`) : undefined);
    if (s) by.get(s)!.push(l);
  }
  const strata = {} as Sample["strata"];
  const clips: SampleClip[] = [];
  for (const s of STRATA) {
    const pool = by.get(s)!;
    const drawn = [...pool].sort((a, b) => hash(`${seed}:${a.shard}:${a.row}`) - hash(`${seed}:${b.shard}:${b.row}`)).slice(0, quota[s]);
    strata[s] = { population: pool.length, sampled: drawn.length };
    for (const l of drawn) clips.push({ row: l.row, shard: l.shard, path: l.path, stratum: s });
  }
  clips.sort((a, b) => hash(`order:${seed}:${a.shard}:${a.row}`) - hash(`order:${seed}:${b.shard}:${b.row}`));
  return { seed, strata, clips };
}

export interface ReviewAnswer {
  shard: string;
  row: number;
  /** `span`: the reviewer heard Quran and marked it. `not_recitation`: no usable recitation. `unsure`: left out of every metric. */
  verdict: "span" | "not_recitation" | "unsure";
  span: WordSpan | null;
  note?: string;
  /** Time spent on the clip. */
  ms: number;
}

/** The latest answer per clip wins, so going back and re-answering just appends. */
export function latestAnswers(answers: ReviewAnswer[]): Map<string, ReviewAnswer> {
  const out = new Map<string, ReviewAnswer>();
  for (const a of answers) out.set(`${a.shard}:${a.row}`, a);
  return out;
}

/** 95% Wilson interval for `k` of `n`. */
export function wilson(k: number, n: number): [number, number] {
  if (n === 0) return [0, 1];
  const z = 1.96;
  const p = k / n;
  const d = 1 + (z * z) / n;
  const mid = (p + (z * z) / (2 * n)) / d;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [Math.max(0, mid - half), Math.min(1, mid + half)];
}

export interface Rate {
  k: number;
  n: number;
  rate: number | null;
  ci: [number, number];
}
const rate = (k: number, n: number): Rate => ({ k, n, rate: n ? k / n : null, ci: wilson(k, n) });

/** Where a word sits in the mushaf, for measuring how far apart two span edges are. */
export interface WordOrder {
  lastWord(surah: number, ayah: number): number;
  ordinal(r: WordRef): number;
}

/**
 * Both edges within `edgeWords` words of the human's, counted across ayah boundaries. Clip edges are often a half
 * word that the listener drops or the reviewer includes either way, so an exact match is too strict a bar.
 */
export function closeSpan(a: WordSpan | null, b: WordSpan | null, order: WordOrder, edgeWords: number): boolean {
  if (!a || !b) return a === b;
  const d = (x: WordRef, y: WordRef) => Math.abs(order.ordinal(x) - order.ordinal(y));
  return d(a.start, b.start) <= edgeWords && d(a.end, b.end) <= edgeWords;
}

export interface StratumScore {
  stratum: Stratum;
  population: number;
  reviewed: number;
  /** Of locked clips: both edges within `edgeWords` of the human span. The headline number. */
  precision: Rate;
  /** Of locked clips: the proposed span equals the human span word for word. */
  exactPrecision: Rate;
  /** Of locked clips: same ayahs, words may differ. */
  ayahPrecision: Rate;
  /** Of reviewed clips: locked on a span not close to the human's, or on no recitation. */
  falseLock: Rate;
  /** Of reviewed clips: not locked. */
  exclusion: Rate;
  /** Of excluded clips: the human heard a clean span, so a usable clip was lost. */
  missed: Rate;
  /** Of reviewed clips: the filename ayah isn't what was recited. */
  filenameWrong: Rate;
}

const fullAyah = (label: ProposedLabel, lastWord: (s: number, a: number) => number): WordSpan | null => {
  const f = label.filenameLabel;
  return f ? { start: { surah: f.surah, ayah: f.ayah, word: 1 }, end: { surah: f.surah, ayah: f.ayah, word: lastWord(f.surah, f.ayah) } } : null;
};
const ayahs = (s: WordSpan) => [s.start, s.end].map((r: WordRef) => `${r.surah}:${r.ayah}`).join("-");

/** A clip accepted for training, with the span to train on. */
export interface EligibleClip {
  labelKind: "accepted";
  shard: string;
  row: number;
  path: string;
  span: WordSpan;
  /** `human`: a reviewer's blind span, which overrides the tracker. `tracker`: a lock at or above the adopted confidence. */
  source: "human" | "tracker";
  confidence: number | null;
}

/**
 * Locks at or above `minConfidence`, with reviewed clips taking the reviewer's answer instead: their span when they
 * heard one (even where the tracker didn't lock), and nothing when they heard no Quran. `unsure` falls back to the rule.
 */
export function eligibleClips(labels: ProposedLabel[], answers: ReviewAnswer[], minConfidence: number): EligibleClip[] {
  const latest = latestAnswers(answers);
  const out: EligibleClip[] = [];
  for (const l of labels) {
    const a = latest.get(`${l.shard}:${l.row}`);
    const base = { labelKind: "accepted" as const, shard: l.shard, row: l.row, path: l.path, confidence: l.confidence };
    if (a && a.verdict === "span" && a.span) out.push({ ...base, span: a.span, source: "human" });
    else if (a && a.verdict === "not_recitation") continue;
    else if (l.state === "locked" && l.proposed && (l.confidence ?? 0) >= minConfidence) out.push({ ...base, span: l.proposed, source: "tracker" });
  }
  return out;
}

export interface Score {
  run: string;
  edgeWords: number;
  strata: StratumScore[];
  /** Weighted by each stratum's share of the run, so oversampled strata don't skew it. Normal-approximation 95% CI. */
  weighted: { precision: WeightedRate; falseLock: WeightedRate; exclusion: WeightedRate };
  /** For each minimum confidence: weighted precision of the locks it keeps, and the share of the run it keeps. */
  sweep: { minConfidence: number; precision: WeightedRate; coverage: number }[];
}

export interface WeightedRate {
  rate: number | null;
  ci: [number, number];
}

export function weightedRate(parts: { weight: number; k: number; n: number }[]): WeightedRate {
  const used = parts.filter((p) => p.n > 0);
  const total = used.reduce((a, p) => a + p.weight, 0);
  if (!total) return { rate: null, ci: [0, 1] };
  let rate = 0;
  let variance = 0;
  for (const p of used) {
    const w = p.weight / total;
    const r = p.k / p.n;
    rate += w * r;
    // Agresti–Coull: an all-right or all-wrong stratum still contributes width.
    const adj = (p.k + 2) / (p.n + 4);
    variance += (w * w * adj * (1 - adj)) / (p.n + 4);
  }
  const half = 1.96 * Math.sqrt(variance);
  return { rate, ci: [Math.max(0, rate - half), Math.min(1, rate + half)] };
}

export const SWEEP = [0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7];

export function score(run: string, sample: Sample, labels: ProposedLabel[], answers: ReviewAnswer[], order: WordOrder, edgeWords = 1): Score {
  const lastWord = order.lastWord;
  const byKey = new Map(labels.map((l) => [`${l.shard}:${l.row}`, l]));
  const latest = latestAnswers(answers);
  const rows = sample.clips
    .map((c) => ({ clip: c, label: byKey.get(`${c.shard}:${c.row}`)!, answer: latest.get(`${c.shard}:${c.row}`) }))
    .filter((r) => r.label && r.answer && r.answer.verdict !== "unsure");

  const human = (r: (typeof rows)[number]) => (r.answer!.verdict === "span" ? r.answer!.span : null);
  const strata: StratumScore[] = STRATA.map((s) => {
    const rs = rows.filter((r) => r.clip.stratum === s);
    const locked = rs.filter((r) => r.label.state === "locked");
    const excluded = rs.filter((r) => r.label.state !== "locked");
    const correct = locked.filter((r) => closeSpan(r.label.proposed, human(r), order, edgeWords));
    const exact = locked.filter((r) => sameSpan(r.label.proposed, human(r)));
    const sameAyahs = locked.filter((r) => human(r) && ayahs(r.label.proposed!) === ayahs(human(r)!));
    return {
      stratum: s,
      population: sample.strata[s].population,
      reviewed: rs.length,
      precision: rate(correct.length, locked.length),
      exactPrecision: rate(exact.length, locked.length),
      ayahPrecision: rate(sameAyahs.length, locked.length),
      falseLock: rate(locked.length - correct.length, rs.length),
      exclusion: rate(excluded.length, rs.length),
      missed: rate(excluded.filter((r) => human(r)).length, excluded.length),
      filenameWrong: rate(rs.filter((r) => !sameSpan(fullAyah(r.label, lastWord), human(r))).length, rs.length),
    };
  });

  // A rate over a subset (e.g. locked clips) weighs each stratum by its estimated size of that subset.
  const weighted = (pick: (s: StratumScore) => Rate) =>
    weightedRate(strata.map((s) => ({ weight: s.reviewed ? (s.population * pick(s).n) / s.reviewed : 0, ...pick(s) })));
  const totalPopulation = STRATA.reduce((a, s) => a + sample.strata[s].population, 0);
  const sweep = SWEEP.map((t) => {
    const parts = STRATA.map((s) => {
      const rs = rows.filter((r) => r.clip.stratum === s);
      const kept = rs.filter((r) => r.label.state === "locked" && (r.label.confidence ?? 0) >= t);
      const population = sample.strata[s].population;
      const k = kept.filter((r) => closeSpan(r.label.proposed, human(r), order, edgeWords)).length;
      return { weight: rs.length ? (population * kept.length) / rs.length : 0, k, n: kept.length };
    });
    // The share of the run kept: each stratum's estimated kept count over the whole run.
    const coverage = parts.reduce((a, p) => a + p.weight, 0) / Math.max(1, totalPopulation);
    return { minConfidence: t, precision: weightedRate(parts), coverage };
  });

  return { run, edgeWords, strata, weighted: { precision: weighted((s) => s.precision), falseLock: weighted((s) => s.falseLock), exclusion: weighted((s) => s.exclusion) }, sweep };
}
