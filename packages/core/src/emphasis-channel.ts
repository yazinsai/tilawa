import type { DecodedPhoneme, TopKPosteriors } from "./evidence.js";
import { subCost, UNIT } from "./fuzzy.js";

/** The four emphatic consonants; coloration is a shared state, not a phone substitution. */
export const isEmphatic = (phone: string): boolean => "صضطظ".includes(phone) && phone.length === 1;

export interface EmphasisModel {
  readonly inventory?: string;
  readonly phones: readonly string[];
  readonly kind: "persistent" | "independent";
  /** Persistent: retention after an ordinary phone. Control: independent coloration probability. */
  readonly persistence: number;
  readonly mass: number;
  readonly background: readonly number[];
  /** P(observed phone | reference phone, coloration state), two normalized rows per reference phone. */
  readonly emission: readonly (readonly (readonly number[])[])[];
}

export interface EmphasisRecording {
  written: string;
  /** Average posterior distribution on each emitted sound run, in inventory order. */
  observed: readonly (readonly number[])[];
}

export interface EmphasisUpdate { cost: number; colored: number }

/** Precompute emission likelihoods once per observation, independent of reference location. */
export function emphasisObservation(model: EmphasisModel, q?: readonly number[]): (colored: number, phone: string) => EmphasisUpdate {
  const valid = q?.length === model.phones.length && q.every(x => Number.isFinite(x) && x >= 0) && q.some(x => x > 0);
  const pooled = valid ? dot(model.background, q!) : 1;
  const emissions = new Map(model.phones.map((p, i) => [p, valid ? model.emission[i].map(row => dot(row, q!)) : [1, 1]]));
  return (colored, phone) => {
    const row = emissions.get(phone);
    if (!row) return { cost: 0, colored };
    const prior = isEmphatic(phone) ? 1 : model.kind === "persistent" ? colored * model.persistence : model.persistence;
    const off = (1 - prior) * row[0], on = prior * row[1], likelihood = off + on;
    return { cost: -UNIT / 3 * Math.log(likelihood / pooled), colored: on / likelihood };
  };
}

/** Average the run's unconditioned rows first; remove blank once, preserving residual top-k mass. */
export function posteriorObservation(p: TopKPosteriors, phones: readonly string[], ph: Pick<DecodedPhoneme, "start" | "end">): number[] | undefined {
  if (ph.start < p.startFrame || ph.end >= p.startFrame + p.nFrames || ph.end < ph.start) return undefined;
  const q = Array(phones.length).fill(0) as number[];
  for (let f = ph.start; f <= ph.end; f++) {
    const row = new Map<number, number>();
    let total = 0;
    for (let c = 0; c < p.k; c++) {
      const at = (f - p.startFrame) * p.k + c;
      const id = p.ids[at], prob = p.probs[at];
      if (!Number.isInteger(id) || id < 0 || id > phones.length || !Number.isFinite(prob) || prob < 0) return undefined;
      row.set(id, (row.get(id) ?? 0) + prob);
      total += prob;
    }
    const rest = Math.max(0, 1 - total) / Math.max(1, phones.length + 1 - row.size);
    for (let i = 0; i < phones.length; i++) q[i] += row.get(i + 1) ?? rest;
  }
  const sum = q.reduce((a, b) => a + b, 0);
  return sum > 0 ? q.map(v => v / sum) : undefined;
}

const normalize = (xs: readonly number[]): number[] => {
  const sum = xs.reduce((a, b) => a + b, 0);
  if (!(sum > 0)) throw new Error("Empty emphasis distribution");
  return xs.map(x => x / sum);
};
const dot = (a: readonly number[], b: readonly number[]) => a.reduce((sum, x, i) => sum + x * b[i], 0);
const logAdd = (a: number, b: number) => a === -Infinity ? b : b === -Infinity ? a : Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));

/** Reference-only deletions do not call this: an absent emphatic sound cannot activate coloration. */
export function emphasisTransition(model: EmphasisModel, phone: string): readonly (readonly number[])[] {
  if (isEmphatic(phone)) return [[0, 1], [0, 1]];
  const rho = model.persistence;
  return model.kind === "persistent" ? [[1, 0], [1 - rho, rho]] : [[1 - rho, rho], [1 - rho, rho]];
}

/** Marginalize coloration on a consuming diagonal; the returned belief belongs to that path. */
export function emphasisUpdate(model: EmphasisModel, colored: number, phone: string, q: readonly number[]): EmphasisUpdate {
  return emphasisObservation(model, q)(colored, phone);
}

/** Training pair channel. All sound placement and coloration boundaries are latent. */
function align(model: EmphasisModel, record: EmphasisRecording, collect?: (r: number, state: number, q: readonly number[], weight: number) => void): number {
  const r = [...record.written].map(p => model.phones.indexOf(p));
  const qs = record.observed;
  if (r.some(x => x < 0) || qs.some(q => q.length !== model.phones.length || q.some(x => !Number.isFinite(x) || x < 0) || !q.some(x => x > 0))) {
    throw new Error("Invalid emphasis training inventory or observation");
  }
  const n = r.length, m = qs.length, width = m + 1;
  const at = (i: number, j: number, s: number) => (i * width + j) * 2 + s;
  const t = [...record.written].map(p => emphasisTransition(model, p));
  const emissions = model.emission.map(states => states.map(row => qs.map(q => Math.log(dot(row, q)))));
  const insert = qs.map(q => Math.log(0.1) + Math.log(dot(model.background, q)));
  const del = Math.log(0.1), match = Math.log(0.8), stop = Math.log(0.9);
  const diagonal = (i: number, j: number, s: number, next: number) => match + Math.log(t[i][s][next]) + emissions[r[i]][next][j];
  const f = new Float64Array((n + 1) * width * 2).fill(-Infinity);
  f[0] = 0;
  for (let i = 0; i <= n; i++) for (let j = 0; j <= m; j++) for (let s = 0; s < 2; s++) {
    const v = f[at(i, j, s)];
    if (i < n && j < m) for (let next = 0; next < 2; next++) {
      const cell = at(i + 1, j + 1, next);
      f[cell] = logAdd(f[cell], v + diagonal(i, j, s, next));
    }
    if (i < n) f[at(i + 1, j, s)] = logAdd(f[at(i + 1, j, s)], v + del);
    if (j < m) f[at(i, j + 1, s)] = logAdd(f[at(i, j + 1, s)], v + insert[j]);
  }
  const logP = logAdd(f[at(n, m, 0)], f[at(n, m, 1)]) + stop;
  if (!collect) return logP;
  const b = new Float64Array(f.length).fill(-Infinity);
  b[at(n, m, 0)] = b[at(n, m, 1)] = stop;
  for (let i = n; i >= 0; i--) for (let j = m; j >= 0; j--) for (let s = 0; s < 2; s++) {
    const cell = at(i, j, s);
    if (i < n && j < m) for (let next = 0; next < 2; next++) b[cell] = logAdd(b[cell], diagonal(i, j, s, next) + b[at(i + 1, j + 1, next)]);
    if (i < n) b[cell] = logAdd(b[cell], del + b[at(i + 1, j, s)]);
    if (j < m) b[cell] = logAdd(b[cell], insert[j] + b[at(i, j + 1, s)]);
  }
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) for (let s = 0; s < 2; s++) for (let next = 0; next < 2; next++) {
    const weight = Math.exp(f[at(i, j, s)] + diagonal(i, j, s, next) + b[at(i + 1, j + 1, next)] - logP);
    if (weight > 0) collect(r[i], next, qs[j], weight);
  }
  return logP;
}
export const emphasisLogLikelihood = (model: EmphasisModel, recording: EmphasisRecording): number => align(model, recording);

/** Both models have two emission rows per phone and the same selection and smoothing capacity. */
export function fitEmphasisModel(records: readonly EmphasisRecording[], phones: readonly string[], kind: EmphasisModel["kind"], persistence: number, mass: number): EmphasisModel {
  if (!records.length || !phones.length || !(mass > 0) || !(persistence >= 0 && persistence <= 1)) throw new Error("Invalid emphasis fit");
  const counts = phones.map(() => 0.5);
  for (const record of records) for (const q of record.observed) {
    if (q.length !== phones.length || q.some(x => !Number.isFinite(x) || x < 0)) throw new Error("Invalid training posterior");
    const probabilities = normalize(q);
    probabilities.forEach((x, i) => counts[i] += x);
  }
  const background = normalize(counts);
  const prior = phones.map(p => normalize(phones.map(h => Math.exp(-3 * subCost(p, h) / UNIT))));
  let model: EmphasisModel = { phones: [...phones], kind, persistence, mass, background, emission: prior.map(row => [row, row]) };
  for (let iteration = 0; iteration < 20; iteration++) {
    const next = prior.map(row => [row.map(x => x * mass), row.map(x => x * mass)]);
    for (const record of records) align(model, record, (r, state, q, weight) => {
      const row = model.emission[r][state], likelihood = dot(row, q);
      q.forEach((v, h) => next[r][state][h] += weight * v * row[h] / likelihood);
    });
    const emission = next.map(states => states.map(normalize));
    const change = emission.reduce((sum, states, r) => sum + states.reduce((acc, row, s) => acc + row.reduce((a, x, h) => a + Math.abs(x - model.emission[r][s][h]), 0), 0), 0);
    model = { ...model, emission };
    if (change < 1e-6) break;
  }
  return model;
}
