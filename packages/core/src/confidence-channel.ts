import { subCost, UNIT } from "./fuzzy.js";

/** A clean training recording. Confidence stays attached to its heard phone throughout alignment. */
export interface ConfidenceRecording {
  written: string;
  heard: readonly { p: string; conf: number }[];
}

export interface ConfidenceChannel {
  readonly phones: readonly string[];
  readonly bins: number;
  readonly mass: number;
  /** Normalized P(heard identity | reference identity); last row is the insertion channel. */
  readonly identity: readonly (readonly number[])[];
  /** Normalized P(confidence bin | reference/heard phonetic classes and identity relation). */
  readonly confidence: Readonly<Record<string, readonly number[]>>;
  /** Reference-independent P(confidence bin | heard identity), used as the likelihood-ratio denominator. */
  readonly background: readonly (readonly number[])[];
}

const NATS_PER_EDIT = 3;
const normalize = (row: number[]) => {
  const sum = row.reduce((a, b) => a + b, 0);
  return row.map(x => x / sum);
};
const logAdd = (a: number, b: number) => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = Math.max(a, b);
  return m + Math.log1p(Math.exp(-Math.abs(a - b)));
};

/** Manner classes share sparse confidence distributions without text or location features. */
export function confidencePhoneClass(p: string): string {
  const classes = ["َُِ۪", "اۦۥٲ", "ءبتدطضقك", "ثحخذزسشصظغفهڇ", "ج", "منں۾", "رل", "وي"];
  const k = classes.findIndex(c => c.includes(p));
  return k < 0 ? "other" : String(k);
}

const relation = (r: string | null, h: string) => r === null ? "insert" : r === h ? "same" : subCost(r, h) < UNIT ? "soft" : "different";
const key = (r: string | null, h: string) => `${r === null ? "insert" : confidencePhoneClass(r)}/${confidencePhoneClass(h)}/${relation(r, h)}`;
const binOf = (c: number, bins: number) => Number.isFinite(c) && c >= 0 && c <= 1 ? Math.min(bins - 1, Math.floor(c * bins)) : null;

/**
 * Confidence evidence alone, in existing edit units. Identity costs are supplied by the caller and kept intact.
 * An absent/invalid confidence, unknown phone, or untrained class contributes zero. Summing out confidence also
 * contributes zero because numerator and denominator are normalized distributions over the same bins.
 */
export function confidenceCost(model: ConfidenceChannel, reference: string | null, heard: string, conf?: number): number {
  const b = binOf(conf ?? NaN, model.bins);
  const h = model.phones.indexOf(heard);
  if (b === null || h < 0 || (reference !== null && !model.phones.includes(reference))) return 0;
  const row = model.confidence[key(reference, heard)];
  if (!row) return 0;
  const ratio = row[b] / model.background[h][b];
  return ratio === 1 ? 0 : -UNIT / NATS_PER_EDIT * Math.log(ratio);
}

/** Precompute the rows used by the local DP; all candidates consume the identical confidence observation. */
export function confidenceObservation(model: ConfidenceChannel, heard: string, conf?: number) {
  const costs = new Map(model.phones.map(r => [r, confidenceCost(model, r, heard, conf)]));
  return { substitution: (r: string) => costs.get(r) ?? 0, insertion: confidenceCost(model, null, heard, conf) };
}

// An EM iteration shares its immutable emission table across all recordings. Avoid logs and class lookups in
// the quadratic alignment loop; the weak cache releases old iterations when the estimator replaces the model.
const emissionTables = new WeakMap<ConfidenceChannel, { joint: Float64Array[]; identity: number[][] }>();
function emissions(model: ConfidenceChannel) {
  let table = emissionTables.get(model);
  if (table) return table;
  const identity = model.identity.map(row => row.map(Math.log));
  const joint = identity.map((row, r) => {
    const values = new Float64Array(model.phones.length * model.bins);
    model.phones.forEach((heard, h) => {
      const conf = model.confidence[key(r === model.phones.length ? null : model.phones[r], heard)] ?? model.background[h];
      conf.forEach((p, b) => values[h * model.bins + b] = row[h] + Math.log(p));
    });
    return values;
  });
  table = { joint, identity };
  emissionTables.set(model, table);
  return table;
}

/**
 * Normalized pair edit channel used only for estimation. Alignment is latent: diagonals, insertions and deletions
 * receive posterior counts rather than trusting a single greedy alignment. Each interior state chooses match,
 * deletion or insertion with fixed probabilities .8/.1/.1; boundaries still pay their chosen transition. Once
 * the reference is exhausted, the channel stops with probability .9 or emits another insertion with probability .1.
 */
function align(model: ConfidenceChannel, record: ConfidenceRecording, collect?: (r: number, h: number, b: number | null, weight: number) => void): number {
  const n = record.written.length, m = record.heard.length, width = m + 1;
  const at = (i: number, j: number) => i * width + j;
  const ids = new Map(model.phones.map((p, i) => [p, i]));
  const wi = [...record.written].map(p => ids.get(p));
  const hi = record.heard.map(p => ids.get(p.p));
  const bins = record.heard.map(p => binOf(p.conf, model.bins));
  if (wi.some(x => x === undefined) || hi.some(x => x === undefined)) throw new Error("Training phone outside inventory");
  const f = new Float64Array((n + 1) * width).fill(-Infinity);
  const back = new Float64Array(f.length).fill(-Infinity);
  const table = emissions(model);
  const emission = (r: number, j: number) => {
    const h = hi[j]!;
    const b = bins[j];
    return b === null ? table.identity[r][h] : table.joint[r][h * model.bins + b];
  };
  const diagonal = (i: number, j: number) => Math.log(0.8) + emission(wi[i]!, j);
  const insertion = (j: number) => Math.log(0.1) + emission(model.phones.length, j);
  const deletion = Math.log(0.1);
  f[0] = 0;
  for (let i = 0; i <= n; i++) for (let j = 0; j <= m; j++) {
    const v = f[at(i, j)];
    if (i < n && j < m) f[at(i + 1, j + 1)] = logAdd(f[at(i + 1, j + 1)], v + diagonal(i, j));
    if (i < n) f[at(i + 1, j)] = logAdd(f[at(i + 1, j)], v + deletion);
    if (j < m) f[at(i, j + 1)] = logAdd(f[at(i, j + 1)], v + insertion(j));
  }
  const logP = f[at(n, m)] + Math.log(0.9);
  if (!collect) return logP;
  back[at(n, m)] = Math.log(0.9);
  for (let i = n; i >= 0; i--) for (let j = m; j >= 0; j--) {
    if (i < n && j < m) back[at(i, j)] = logAdd(back[at(i, j)], diagonal(i, j) + back[at(i + 1, j + 1)]);
    if (i < n) back[at(i, j)] = logAdd(back[at(i, j)], deletion + back[at(i + 1, j)]);
    if (j < m) back[at(i, j)] = logAdd(back[at(i, j)], insertion(j) + back[at(i, j + 1)]);
  }
  for (let i = 0; i <= n; i++) for (let j = 0; j < m; j++) {
    const b = bins[j];
    if (i < n) collect(wi[i]!, hi[j]!, b, Math.exp(f[at(i, j)] + diagonal(i, j) + back[at(i + 1, j + 1)] - logP));
    collect(model.phones.length, hi[j]!, b, Math.exp(f[at(i, j)] + insertion(j) + back[at(i, j + 1)] - logP));
  }
  return logP;
}

/** Density likelihood makes different equal-width bin counts comparable during training-only selection. */
export const confidenceLogLikelihood = (model: ConfidenceChannel, record: ConfidenceRecording) =>
  align(model, record) + record.heard.filter(p => binOf(p.conf, model.bins) !== null).length * Math.log(model.bins);

/** Fit joint heard identity/confidence distributions; hyperparameters are chosen by training-speaker validation. */
export function fitConfidenceChannel(records: readonly ConfidenceRecording[], phones: readonly string[], bins: number, mass: number): ConfidenceChannel {
  if (!records.length || !phones.length || !Number.isInteger(bins) || bins < 1 || !(mass > 0)) throw new Error("Invalid confidence fit");
  const k = phones.length;
  const ids = new Map(phones.map((p, i) => [p, i]));
  const bg = Array.from({ length: k }, () => Array(bins).fill(0.5) as number[]);
  const heardCounts = Array(k).fill(0.5) as number[];
  for (const record of records) for (const p of record.heard) {
    const h = ids.get(p.p), b = binOf(p.conf, bins);
    if (h === undefined) throw new Error("Training phone outside inventory");
    heardCounts[h]++;
    if (b !== null) bg[h][b]++;
  }
  const background = bg.map(normalize);
  const identityPrior = phones.map(r => normalize(phones.map(h => Math.exp(-NATS_PER_EDIT * subCost(r, h) / UNIT))));
  identityPrior.push(normalize(heardCounts));
  // The class prior pools confidence by heard manner, keeping the class model independent of any reference text.
  const classBg: Record<string, number[]> = {};
  phones.forEach((h, i) => {
    const c = confidencePhoneClass(h);
    classBg[c] ??= Array(bins).fill(0.5) as number[];
    bg[i].forEach((v, b) => classBg[c][b] += v);
  });
  for (const c of Object.keys(classBg)) classBg[c] = normalize(classBg[c]);
  const prior: Record<string, number[]> = {};
  for (const r of [...phones, null]) for (const h of phones) prior[key(r, h)] = classBg[confidencePhoneClass(h)];
  let model: ConfidenceChannel = { phones: [...phones], bins, mass, identity: identityPrior, confidence: prior, background };
  for (let iteration = 0; iteration < 30; iteration++) {
    const identity = identityPrior.map(row => row.map(p => p * mass));
    const confidence = Object.fromEntries(Object.entries(prior).map(([c, row]) => [c, row.map(p => p * mass)]));
    for (const record of records) align(model, record, (r, h, b, weight) => {
      identity[r][h] += weight;
      if (b !== null) confidence[key(r === k ? null : phones[r], phones[h])][b] += weight;
    });
    const next: ConfidenceChannel = { ...model, identity: identity.map(normalize), confidence: Object.fromEntries(Object.entries(confidence).map(([c, row]) => [c, normalize(row)])) };
    const change = Object.keys(prior).reduce((sum, c) => sum + next.confidence[c].reduce((s, p, b) => s + Math.abs(p - model.confidence[c][b]), 0), 0);
    model = next;
    if (change < 1e-6) break;
  }
  return model;
}
