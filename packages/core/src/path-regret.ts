import { UNIT, deletionCost, insertionCost, subCost } from "./fuzzy.js";

/** Shared acoustic categories; direction and the existing duration discounts remain explicit. */
export const EDIT_KINDS = ["sub-hard", "sub-soft", "sub-vowel", "ins-hard", "ins-vowel", "ins-duration",
  "del-hard", "del-vowel", "del-duration"] as const;
export const BASE_EDIT_COSTS = [UNIT, 5, 5, UNIT, 5, 2, UNIT, 5, 2] as const;
const vowels = new Set(["َ", "ِ", "ُ"]);

/** Exact diagonals have no acoustic cost. Movement is accounted for separately by local search. */
export function editKind(kind: "diagonal" | "insert" | "delete", ref: string, at: number, heard = ""): number | null {
  if (kind === "diagonal") {
    if (ref[at] === heard) return null;
    return vowels.has(ref[at]) && vowels.has(heard) ? 2 : subCost(ref[at], heard) < UNIT ? 1 : 0;
  }
  const c = kind === "insert" ? insertionCost(ref, at, heard) : deletionCost(ref, at);
  const base = kind === "insert" ? 3 : 6;
  return base + (c === 2 ? 2 : c === 5 ? 1 : 0);
}

export interface PathCostModel {
  /** Costs in the same UNITs as movement and the existing commitment margin. */
  costs: readonly number[];
}
export interface PathUncertaintySet {
  models: readonly PathCostModel[];
}
export interface PathCostWitness {
  counts: readonly number[];
  fixedCost: number;
  /** Observations consumed, including insertions; jumps never reset this interval. */
  observations: number;
}

export function validateUncertaintySet(set: PathUncertaintySet): void {
  if (!set.models.length || set.models.some(m => m.costs.length !== EDIT_KINDS.length
    || m.costs.some(c => !Number.isFinite(c) || c <= 0))) throw new Error("Invalid acoustic uncertainty set");
}

/** Max regret over the convex hull is attained at a fitted vertex: no independent parameter extremes. */
export function minimaxPathRegret(set: PathUncertaintySet, paths: readonly PathCostWitness[], preferred = 0): {
  selected: number; regrets: number[]; costs: number[][];
} {
  validateUncertaintySet(set);
  if (!paths.length || preferred < 0 || preferred >= paths.length || paths.some(p =>
    p.counts.length !== EDIT_KINDS.length || p.counts.some(n => !Number.isInteger(n) || n < 0)
    || !Number.isFinite(p.fixedCost) || p.fixedCost < 0 || !Number.isInteger(p.observations)
    || p.observations < 0 || p.observations !== paths[0].observations)) throw new Error("Paths must cover one heard interval");
  const costs = paths.map(p => set.models.map(m => p.fixedCost + p.counts.reduce((c, n, j) => c + n * m.costs[j], 0)));
  const minima = set.models.map((_, j) => Math.min(...costs.map(c => c[j])));
  const regrets = costs.map(c => Math.max(...c.map((v, j) => v - minima[j])));
  // An objective tie retains the original acoustic choice, including its nearest-progress tie break.
  let selected = preferred;
  for (let i = 0; i < paths.length; i++) if (regrets[i] < regrets[selected]) selected = i;
  return { selected, regrets, costs };
}

export interface EditTrainingRecord { written: string; heard: string }
export interface EditTrainingCounts { counts: number[]; matches: number }

/** Full-transcript directed alignment for none-labelled training clips, with the existing edit prices. */
export function trainingEditCounts(record: EditTrainingRecord): EditTrainingCounts {
  const { written: ref, heard: obs } = record;
  if (!ref.length || !obs.length) throw new Error("Empty training recording");
  const w = ref.length + 1, d = new Float64Array(w * (obs.length + 1));
  const kinds = new Int8Array(d.length);
  for (let i = 1; i < w; i++) { d[i] = d[i - 1] + deletionCost(ref, i - 1); kinds[i] = 2; }
  for (let j = 1; j <= obs.length; j++) for (let i = 0; i < w; i++) {
    const cell = j * w + i;
    d[cell] = d[cell - w] + insertionCost(ref, i, obs[j - 1]); kinds[cell] = 1;
    if (i > 0) {
      const s = d[cell - w - 1] + subCost(ref[i - 1], obs[j - 1]);
      if (s <= d[cell]) { d[cell] = s; kinds[cell] = 0; }
      const del = d[cell - 1] + deletionCost(ref, i - 1);
      if (del < d[cell]) { d[cell] = del; kinds[cell] = 2; }
    }
  }
  const counts = Array<number>(EDIT_KINDS.length).fill(0);
  let i = ref.length, j = obs.length, matches = 0;
  while (i || j) {
    const kind = kinds[j * w + i];
    const k = kind === 0 ? editKind("diagonal", ref, i - 1, obs[j - 1])
      : kind === 1 ? editKind("insert", ref, i, obs[j - 1]) : editKind("delete", ref, i - 1);
    if (k === null) matches++; else counts[k]++;
    if (kind !== 1) i--;
    if (kind !== 2) j--;
  }
  return { counts, matches };
}

/** Jeffreys-smoothed probability of each acoustic error versus exact recognition. All costs stay positive. */
export function fitAcousticCosts(records: readonly EditTrainingCounts[]): number[] {
  if (!records.length) throw new Error("No eligible training recordings");
  const matches = records.reduce((n, r) => n + r.matches, 0);
  if (!matches) throw new Error("Training contains no exact recognition");
  return EDIT_KINDS.map((_, j) => {
    const errors = records.reduce((n, r) => n + r.counts[j], 0);
    return -Math.log((errors + 0.5) / (errors + matches + 1));
  });
}

/** One shared conversion, fitted on training path totals; never independently rescale a partition vector. */
export function trainingCostScale(nats: readonly number[], records: readonly EditTrainingCounts[]): number {
  let xy = 0, xx = 0;
  for (const r of records) {
    const x = r.counts.reduce((s, n, j) => s + n * nats[j], 0);
    const y = r.counts.reduce((s, n, j) => s + n * BASE_EDIT_COSTS[j], 0);
    xy += x * y; xx += x * x;
  }
  if (!xx || !xy) throw new Error("No training acoustic edits for unit conversion");
  return xy / xx;
}
