import { hash, weightedRate, wilson } from "../review/index.js";
import type { WeightedRate } from "../review/index.js";
import type { BucketCutoffs, BucketedClip } from "./index.js";

/** A bucketed clip with the numbers the cutoffs act on, so other cutoffs can be scored from the same review. */
export interface CutoffClip extends BucketedClip {
  shard: string;
  row: number;
  path: string;
  per: number | null;
  meanConf: number | null;
  confidence: number | null;
}

export const CUTOFF_STRATA = [
  "text_only",
  "clean_core",
  "clean_edge",
  "deviant_edge",
  "deviant_core",
  "ambiguous",
  "reject_high_per",
  "reject_low_teacher",
  "reject_other",
] as const;
export type CutoffStratum = (typeof CUTOFF_STRATA)[number];

export const CUTOFF_QUOTA: Record<CutoffStratum, number> = {
  text_only: 30,
  clean_core: 40,
  clean_edge: 50,
  deviant_edge: 50,
  deviant_core: 30,
  ambiguous: 30,
  reject_high_per: 30,
  reject_low_teacher: 25,
  reject_other: 15,
};

/** Null for clips with no audio worth hearing (failed decodes, over-long clips). */
export function cutoffStratum(c: CutoffClip, k: BucketCutoffs): CutoffStratum | null {
  if (c.label?.kind === "text") return "text_only";
  const per = c.per ?? Infinity;
  const conf = c.meanConf ?? 0;
  switch (c.bucket) {
    case "clean-match":
      return per > k.cleanMaxPer / 2 || conf < k.cleanMinTeacherConf + 0.03 ? "clean_edge" : "clean_core";
    case "deviant":
      return per <= 2 * k.cleanMaxPer || c.reason === "teacher_unsure" ? "deviant_edge" : "deviant_core";
    case "ambiguous":
      return "ambiguous";
    case "reject":
      if (c.reason === "high_per") return "reject_high_per";
      if (c.reason === "low_teacher_confidence") return "reject_low_teacher";
      if (c.reason === "lost" || c.reason === "insufficient_evidence") return "reject_other";
      return null;
  }
}

export interface CutoffSample {
  seed: number;
  cutoffs: BucketCutoffs;
  strata: Partial<Record<CutoffStratum, { population: number; sampled: number }>>;
  /** Clips left out of sampling, by reason. They count toward the population. */
  unsampled: Record<string, number>;
  /** Shuffled across strata. */
  clips: (CutoffClip & { stratum: CutoffStratum })[];
}

export function drawCutoffSample(clips: CutoffClip[], cutoffs: BucketCutoffs, o: { seed: number; quota?: Partial<Record<CutoffStratum, number>> }): CutoffSample {
  const quota = o.quota ?? CUTOFF_QUOTA;
  const by = new Map<CutoffStratum, CutoffClip[]>();
  const unsampled: Record<string, number> = {};
  for (const c of clips) {
    const s = cutoffStratum(c, cutoffs);
    if (s) (by.get(s) ?? by.set(s, []).get(s)!).push(c);
    else unsampled[c.reason] = (unsampled[c.reason] ?? 0) + 1;
  }
  const strata: CutoffSample["strata"] = {};
  const out: CutoffSample["clips"] = [];
  for (const s of CUTOFF_STRATA) {
    const pool = by.get(s);
    if (!pool) continue;
    const key = (c: CutoffClip) => hash(`${o.seed}:${c.shard}:${c.row}`);
    const drawn = [...pool].sort((a, b) => key(a) - key(b)).slice(0, quota[s] ?? 0);
    strata[s] = { population: pool.length, sampled: drawn.length };
    for (const c of drawn) out.push({ ...c, stratum: s });
  }
  out.sort((a, b) => hash(`order:${o.seed}:${a.shard}:${a.row}`) - hash(`order:${o.seed}:${b.shard}:${b.row}`));
  return { seed: o.seed, cutoffs, strata, unsampled, clips: out };
}

/** `exact`: the audio is the label's text with no mistakes. `mistakes`: right passage, with recitation errors. */
export type CutoffVerdict = "exact" | "mistakes" | "wrong_passage" | "no_recitation" | "unsure";

export interface CutoffAnswer {
  shard: string;
  row: number;
  verdict: CutoffVerdict;
  note?: string;
  ms?: number;
}

export interface GridRate extends WeightedRate {
  k: number;
  n: number;
}

export interface GridPoint {
  cleanMaxPer: number;
  cleanMinTeacherConf: number;
  /** Precision: judged exact. Share: estimated share of the whole run that lands in the bucket. */
  clean: { precision: GridRate; share: number };
  /** Right passage (exact or with mistakes), and with mistakes specifically. */
  deviant: { rightPassage: GridRate; mistakes: GridRate; share: number };
}

export interface CutoffScore {
  answered: number;
  unsure: number;
  /** Verdict counts per stratum. */
  strata: Partial<Record<CutoffStratum, { population: number; verdicts: Partial<Record<CutoffVerdict, number>> }>>;
  grid: GridPoint[];
}

const latest = (answers: CutoffAnswer[]) => {
  const m = new Map<string, CutoffAnswer>();
  for (const a of answers) m.set(`${a.shard}:${a.row}`, a);
  return m;
};

export function scoreCutoffs(sample: CutoffSample, answers: CutoffAnswer[], grid: { cleanMaxPer: number[]; cleanMinTeacherConf: number[] }): CutoffScore {
  const a = latest(answers);
  const k = sample.cutoffs;
  const total = Object.values(sample.strata).reduce((n, s) => n + s!.population, 0) + Object.values(sample.unsampled).reduce((n, x) => n + x, 0);
  const judged = sample.clips.flatMap((c) => {
    const v = a.get(`${c.shard}:${c.row}`)?.verdict;
    return v && v !== "unsure" ? [{ c, v }] : [];
  });
  const answeredIn = new Map<CutoffStratum, number>();
  for (const { c } of judged) answeredIn.set(c.stratum, (answeredIn.get(c.stratum) ?? 0) + 1);

  const strata: CutoffScore["strata"] = {};
  for (const [s, x] of Object.entries(sample.strata) as [CutoffStratum, { population: number }][]) strata[s] = { population: x.population, verdicts: {} };
  for (const c of sample.clips) {
    const v = a.get(`${c.shard}:${c.row}`)?.verdict;
    if (v) strata[c.stratum]!.verdicts[v] = (strata[c.stratum]!.verdicts[v] ?? 0) + 1;
  }

  const rateOf = (inBucket: (c: CutoffClip) => boolean, hit: (v: CutoffVerdict) => boolean) => {
    const parts: { weight: number; k: number; n: number }[] = [];
    let share = 0;
    for (const s of CUTOFF_STRATA) {
      const answered = answeredIn.get(s);
      if (!answered) continue;
      const mine = judged.filter((j) => j.c.stratum === s && inBucket(j.c));
      const weight = (sample.strata[s]!.population * mine.length) / answered;
      share += weight;
      parts.push({ weight, k: mine.filter((j) => hit(j.v)).length, n: mine.length });
    }
    const kk = parts.reduce((n, p) => n + p.k, 0);
    const nn = parts.reduce((n, p) => n + p.n, 0);
    const w = weightedRate(parts);
    return { rate: { ...w, ci: nn ? w.ci : wilson(0, 0), k: kk, n: nn }, share: total ? share / total : 0 };
  };

  const points: GridPoint[] = [];
  for (const p of grid.cleanMaxPer)
    for (const t of grid.cleanMinTeacherConf) {
      const located = (c: CutoffClip) => c.label !== null && (c.per ?? Infinity) <= k.deviantMaxPer;
      const isClean = (c: CutoffClip) => located(c) && (c.per ?? Infinity) <= p && (c.meanConf ?? 0) >= t;
      const isDeviant = (c: CutoffClip) => located(c) && !isClean(c);
      const clean = rateOf(isClean, (v) => v === "exact");
      const right = rateOf(isDeviant, (v) => v === "exact" || v === "mistakes");
      const mistakes = rateOf(isDeviant, (v) => v === "mistakes");
      points.push({
        cleanMaxPer: p,
        cleanMinTeacherConf: t,
        clean: { precision: clean.rate, share: clean.share },
        deviant: { rightPassage: right.rate, mistakes: mistakes.rate, share: right.share },
      });
    }
  return { answered: judged.length, unsure: sample.clips.filter((c) => a.get(`${c.shard}:${c.row}`)?.verdict === "unsure").length, strata, grid: points };
}
