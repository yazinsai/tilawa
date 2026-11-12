import type { DecodedPhoneme, TopKPosteriors } from "./evidence.js";
import { subCost, UNIT } from "./fuzzy.js";

/** Complete bundles are categorical states. We never multiply separate feature marginals. */
export function articulation(p: string): string {
  const groups: readonly [string, string][] = [
    ["ب", "bilabial/stop/voiced"], ["م۾", "bilabial/nasal/voiced"],
    ["ف", "labiodental/fricative/unvoiced"], ["ث", "dental/fricative/unvoiced"],
    ["ذظ", "dental/fricative/voiced"], ["تط", "alveolar/stop/unvoiced"],
    ["دض", "alveolar/stop/voiced"], ["سص", "alveolar/fricative/unvoiced"],
    ["ز", "alveolar/fricative/voiced"], ["ن", "alveolar/nasal/voiced"],
    ["ر", "alveolar/trill/voiced"], ["ل", "alveolar/lateral/voiced"],
    ["شڇ", "postalveolar/fricative/unvoiced"], ["ج", "postalveolar/affricate/voiced"],
    ["ك", "velar/stop/unvoiced"], ["ق", "uvular/stop/unvoiced"],
    ["خ", "uvular/fricative/unvoiced"], ["غ", "uvular/fricative/voiced"],
    ["ح", "pharyngeal/fricative/unvoiced"], ["ع", "pharyngeal/approximant/voiced"],
    ["ه", "glottal/fricative/unvoiced"], ["ء", "glottal/stop/unvoiced"],
    ["و", "labiovelar/approximant/voiced"], ["ي", "palatal/approximant/voiced"],
    ["اَٲ", "low/vowel/voiced"], ["ۦِ۪", "front-high/vowel/voiced"],
    ["ۥُ", "back-high/vowel/voiced"], ["ں", "unspecified/nasal/voiced"],
  ];
  return groups.find(([phones]) => phones.includes(p))?.[1] ?? `symbol-${p}/other/unspecified`;
}

export interface ArticulatoryModel {
  readonly inventory?: string;
  readonly phones: readonly string[];
  readonly states: readonly string[];
  /** State for each phone, sharing complete articulatory bundles in the joint model. */
  readonly phoneState: readonly number[];
  readonly background: readonly number[];
  /** Normalized P(observed state | reference state); last row is insertion. */
  readonly emission: readonly (readonly number[])[];
  readonly mass: number;
  readonly kind: "joint" | "identity";
}

export interface ArticulatoryRecording {
  written: string;
  /** Nonblank phone distributions, in model inventory order, for each retained token run. */
  observed: readonly (readonly number[])[];
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

const normalize = (xs: readonly number[]) => {
  const sum = xs.reduce((a, b) => a + b, 0);
  return xs.map(v => v / sum);
};
const dot = (a: readonly number[], b: readonly number[]) => a.reduce((sum, x, i) => sum + x * b[i], 0);
const logAdd = (a: number, b: number) => a === -Infinity ? b : b === -Infinity ? a : Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));

export function bundleObservation(model: ArticulatoryModel, probabilities: readonly number[]): number[] | undefined {
  if (probabilities.length !== model.phones.length || probabilities.some(x => !Number.isFinite(x) || x < 0)) return undefined;
  const q = Array(model.states.length).fill(0) as number[];
  probabilities.forEach((v, i) => q[model.phoneState[i]] += v);
  return q.some(v => v > 0) ? normalize(q) : undefined;
}

/** A single observation contributes the same likelihood factor to every local candidate's diagonal. */
export function articulatoryCosts(model: ArticulatoryModel, probabilities: readonly number[]): ReadonlyMap<string | null, number> {
  const q = bundleObservation(model, probabilities);
  if (!q) return new Map();
  const pooled = dot(model.background, q);
  const costs = new Map<string | null, number>(model.phones.map((p, i) => [p, -UNIT / 3 * Math.log(dot(model.emission[model.phoneState[i]], q) / pooled)]));
  costs.set(null, -UNIT / 3 * Math.log(dot(model.emission[model.states.length], q) / pooled));
  return costs;
}

/** Forward/backward pair channel: token placement is latent, including reference-only duration deletions. */
function align(model: ArticulatoryModel, recording: ArticulatoryRecording, collect?: (r: number, q: readonly number[], weight: number) => void): number {
  const ids = new Map(model.phones.map((p, i) => [p, model.phoneState[i]]));
  const r = [...recording.written].map(p => ids.get(p));
  const qs = recording.observed.map(q => bundleObservation(model, q));
  if (r.some(x => x === undefined) || qs.some(x => !x)) throw new Error("Invalid training inventory or observation");
  const n = r.length, m = qs.length, width = m + 1;
  const at = (i: number, j: number) => i * width + j;
  const ins = model.states.length;
  const emission = model.emission.map(row => qs.map(q => Math.log(dot(row, q!))));
  const diagonal = (i: number, j: number) => Math.log(0.8) + emission[r[i]!][j];
  const insertion = (_i: number, j: number) => Math.log(0.1) + emission[ins][j];
  const del = Math.log(0.1);
  const f = new Float64Array((n + 1) * width).fill(-Infinity);
  f[0] = 0;
  for (let i = 0; i <= n; i++) for (let j = 0; j <= m; j++) {
    const v = f[at(i, j)];
    if (i < n && j < m) f[at(i + 1, j + 1)] = logAdd(f[at(i + 1, j + 1)], v + diagonal(i, j));
    if (i < n) f[at(i + 1, j)] = logAdd(f[at(i + 1, j)], v + del);
    if (j < m) f[at(i, j + 1)] = logAdd(f[at(i, j + 1)], v + insertion(i, j));
  }
  const logP = f[at(n, m)] + Math.log(0.9);
  if (!collect) return logP;
  const back = new Float64Array(f.length).fill(-Infinity);
  back[at(n, m)] = Math.log(0.9);
  for (let i = n; i >= 0; i--) for (let j = m; j >= 0; j--) {
    if (i < n && j < m) back[at(i, j)] = logAdd(back[at(i, j)], diagonal(i, j) + back[at(i + 1, j + 1)]);
    if (i < n) back[at(i, j)] = logAdd(back[at(i, j)], del + back[at(i + 1, j)]);
    if (j < m) back[at(i, j)] = logAdd(back[at(i, j)], insertion(i, j) + back[at(i, j + 1)]);
  }
  for (let i = 0; i <= n; i++) for (let j = 0; j < m; j++) {
    if (i < n) collect(r[i]!, qs[j]!, Math.exp(f[at(i, j)] + diagonal(i, j) + back[at(i + 1, j + 1)] - logP));
    collect(ins, qs[j]!, Math.exp(f[at(i, j)] + insertion(i, j) + back[at(i, j + 1)] - logP));
  }
  return logP;
}

export const articulatoryLogLikelihood = (model: ArticulatoryModel, recording: ArticulatoryRecording) => align(model, recording);

/** Joint and identity controls use identical observations, priors, smoothing and placement marginalization. */
export function fitArticulatoryModel(records: readonly ArticulatoryRecording[], phones: readonly string[], kind: ArticulatoryModel["kind"], mass: number): ArticulatoryModel {
  if (!records.length || !phones.length || !(mass > 0)) throw new Error("Empty articulatory fit");
  const states = [...new Set(phones.map(p => kind === "joint" ? articulation(p) : p))];
  const phoneState = phones.map(p => states.indexOf(kind === "joint" ? articulation(p) : p));
  const seed: ArticulatoryModel = { phones: [...phones], states, phoneState, background: states.map(() => 1 / states.length), emission: [], mass, kind };
  const counts = states.map(() => 0.5);
  for (const record of records) for (const obs of record.observed) {
    const q = bundleObservation(seed, obs);
    if (!q) throw new Error("Invalid training observation");
    q.forEach((x, b) => counts[b] += x);
  }
  const background = normalize(counts);
  // Pool complete phone-pair priors into complete bundle pairs; no independent feature emissions.
  const prior = states.map((_, r) => {
    const row = states.map(() => 0);
    phones.forEach((p, i) => {
      if (phoneState[i] !== r) return;
      phones.forEach((h, j) => row[phoneState[j]] += Math.exp(-3 * subCost(p, h) / UNIT));
    });
    return normalize(row);
  });
  prior.push(background);
  let model: ArticulatoryModel = { ...seed, background, emission: prior };
  for (let iteration = 0; iteration < 20; iteration++) {
    const next = prior.map(row => row.map(x => x * mass));
    for (const record of records) align(model, record, (r, q, weight) => {
      const likelihood = dot(model.emission[r], q);
      q.forEach((v, b) => next[r][b] += weight * v * model.emission[r][b] / likelihood);
    });
    const emission = next.map(normalize);
    const change = emission.reduce((sum, row, r) => sum + row.reduce((s, v, b) => s + Math.abs(v - model.emission[r][b]), 0), 0);
    model = { ...model, emission };
    if (change < 1e-6) break;
  }
  return model;
}
