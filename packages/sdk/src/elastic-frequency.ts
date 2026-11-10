import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, ElasticEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

/** Compact spectral trajectories, fitted and selected using training speakers only. */
export interface ElasticModel {
  schema: 1;
  inventory: string;
  tokens: string;
  frameMs: 40;
  observationMs: 10;
  dimensions: 24;
  scale: number;
  means: number[][][];
  variances: number[][][];
  counts: number[];
  stay: number[];
  displacements: number[];
  regularization: number;
  maxFrames: number;
}

/** Keep the native 10 ms spectral progression, independent of microphone chunking.
 * Frame centering removes overall gain without independently normalizing frequency bands.
 */
export class ElasticObservations {
  readonly frames: number[][] = [];
  private readonly fbank = new Fbank();
  push(audio: Float32Array): void {
    for (const row of this.fbank.push(audio)) {
      const x = Array.from({ length: 24 }, (_, b) => {
        const lo = Math.floor(b * 80 / 24), hi = Math.floor((b + 1) * 80 / 24);
        let sum = 0;
        for (let i = lo; i < hi; i++) sum += row[i];
        return sum / (hi - lo);
      });
      const mean = x.reduce((a, b) => a + b, 0) / x.length;
      this.frames.push(x.map(v => v - mean));
    }
  }
}
export function elasticFeatures(audio: Float32Array): number[][] {
  const observations = new ElasticObservations();
  observations.push(audio);
  return observations.frames;
}
export const elasticLogadd = (a: number, b: number): number => {
  const m = Math.max(a, b);
  return Number.isFinite(m) ? m + Math.log(Math.exp(a - m) + Math.exp(b - m)) : -Infinity;
};
const logNormalize = (values: number[]) => {
  const z = values.reduce(elasticLogadd, -Infinity);
  return values.map(v => v - z);
};

/** Smooth monotonic correspondence, pinned at the spectrum's ends. The maximum
 * derivative perturbation is 2*pi/23 < 1, so no frequencies exchange order.
 */
export function warpSpectrum(row: number[], displacement: number): number[] {
  return row.map((_, b) => {
    const coordinate = b + displacement * Math.sin(Math.PI * b / (row.length - 1));
    const lo = Math.floor(coordinate), hi = Math.min(row.length - 1, lo + 1);
    return row[lo] + (row[hi] - row[lo]) * (coordinate - lo);
  });
}

/** Normalized deformation law shared by the complete competing pronunciations.
 * At most one band of displacement changes per observation, including across
 * phone boundaries. There is no phone-local warp initialization or exit reward.
 */
export function deformationPrior(model: ElasticModel, elastic: boolean) {
  const displacements = elastic ? model.displacements : [0];
  const initial = logNormalize(displacements.map(d => -model.regularization * d * d / 4));
  const transitions = displacements.map(from => logNormalize(displacements.map(to =>
    Math.abs(from - to) > 1 ? -Infinity : -model.regularization * ((to - from) ** 2 + to * to / 4))));
  return { displacements, initial, transitions };
}

/** Joint forward sum over ordered trajectory states, temporal placement and
 * continuous frequency correspondence. Three knots per phone are mandatory;
 * identical consecutive phones remain separate articulations. Subtract the
 * emission-free completion partition so extra timing paths are not evidence.
 */
export function scoreElasticSequence(model: ElasticModel, raw: number[][], pronunciation: string, elastic: boolean): number | null {
  const ids = [...pronunciation].map(p => model.tokens.indexOf(p) + 1);
  const n = ids.length * 3;
  if (!ids.length || ids.some(p => p < 1 || model.counts[p] <= 0) || raw.length < n || raw.length > model.maxFrames ||
    raw.some(r => r.length !== model.dimensions || r.some(v => !Number.isFinite(v)))) return null;
  const warp = deformationPrior(model, elastic), k = warp.displacements.length;
  const emissions = new Map<number, number[][]>();
  for (const p of new Set(ids)) {
    const mu = model.means[p], variance = model.variances[p];
    emissions.set(p, raw.map(row => {
      const x = row.map(v => v / model.scale);
      return Array.from({ length: 3 * k }, (_, qw) => {
        const q = Math.floor(qw / k), d = warp.displacements[qw % k];
        const mean = warpSpectrum(mu[q], d), v = warpSpectrum(variance[q], d);
        return x.reduce((sum, value, b) => sum - .5 * (Math.log(2 * Math.PI * v[b]) + (value - mean[b]) ** 2 / v[b]), 0);
      });
    }));
  }
  let scores = new Float64Array(n * k).fill(-Infinity);
  let prior = new Float64Array(n).fill(-Infinity);
  for (let t = 0; t < raw.length; t++) {
    const next = new Float64Array(n * k).fill(-Infinity), nextPrior = prior.slice();
    for (let s = 0; s < n; s++) {
      const q = s % 3, stay = Math.log(model.stay[q]), advance = Math.log1p(-model.stay[(s + 2) % 3]);
      nextPrior[s] = t === 0 ? (s === 0 ? 0 : -Infinity) : elasticLogadd(prior[s] + stay, s ? prior[s - 1] + advance : -Infinity);
      for (let w = 0; w < k; w++) {
        let incoming = t === 0 && s === 0 ? warp.initial[w] : -Infinity;
        if (t) for (let previous = 0; previous < k; previous++) {
          const temporal = elasticLogadd(scores[s * k + previous] + stay, s ? scores[(s - 1) * k + previous] + advance : -Infinity);
          incoming = elasticLogadd(incoming, temporal + warp.transitions[previous][w]);
        }
        next[s * k + w] = incoming + emissions.get(ids[Math.floor(s / 3)])![t][q * k + w];
      }
    }
    scores = next; prior = nextPrior;
  }
  const result = Array.from(scores.subarray((n - 1) * k)).reduce(elasticLogadd, -Infinity) - prior[n - 1];
  return Number.isFinite(result) ? result : null;
}

export function elasticComparison(model: ElasticModel, window: number[][], c: SequenceEvidenceCandidate): ElasticEvidence | null {
  const rigidWritten = scoreElasticSequence(model, window, c.written, false), rigidRival = scoreElasticSequence(model, window, c.rival, false);
  const elasticWritten = scoreElasticSequence(model, window, c.written, true), elasticRival = scoreElasticSequence(model, window, c.rival, true);
  if ([rigidWritten, rigidRival, elasticWritten, elasticRival].some(v => v === null)) return null;
  return { ...c, rigidWritten: rigidWritten!, rigidRival: rigidRival!, elasticWritten: elasticWritten!, elasticRival: elasticRival! };
}
export const elasticReversal = (c: ElasticEvidence): boolean =>
  [c.rigidWritten, c.rigidRival, c.elasticWritten, c.elasticRival].every(Number.isFinite) &&
  c.rigidWritten < c.rigidRival && c.elasticWritten > c.elasticRival;

export function validateElasticModel(value: unknown, inventory: string, tokens: string): ElasticModel {
  const m = value as ElasticModel, n = tokens.length + 1;
  const vector = (v: number[], len: number, positive = false) => Array.isArray(v) && v.length === len && v.every(x => Number.isFinite(x) && (!positive || x > 0));
  const tensor = (v: number[][][], positive = false) => Array.isArray(v) && v.length === n && v.every(p => Array.isArray(p) && p.length === 3 && p.every(r => vector(r, 24, positive)));
  if (!m || m.schema !== 1 || m.inventory !== inventory || m.tokens !== tokens || m.frameMs !== 40 || m.observationMs !== 10 || m.dimensions !== 24 ||
    !Number.isFinite(m.scale) || m.scale <= 0 || !tensor(m.means) || !tensor(m.variances, true) ||
    !vector(m.counts, n) || m.counts.some(c => c < 0) || !vector(m.stay, 3, true) || m.stay.some(p => p >= 1) ||
    !Array.isArray(m.displacements) || JSON.stringify(m.displacements) !== JSON.stringify([-2, -1, 0, 1, 2]) ||
    !Number.isFinite(m.regularization) || m.regularization <= 0 || !Number.isInteger(m.maxFrames) || m.maxFrames < 3 || m.maxFrames > 2048)
    throw new Error("Invalid elastic acoustic model or inventory");
  return m;
}

/** Attach only exact-window reversals; discovering comparisons cannot change any
 * primary token, posterior, tracker event or insertion ownership decision.
 */
export function addElasticEvidence(model: ElasticModel | undefined, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset, options: CheckerOptions = {}, mode: "free" | "memorization" = "free",
  inspect?: (c: ElasticEvidence | null, candidate: SequenceEvidenceCandidate) => void): PhonemeEvidence[] {
  if (!model || !audio.length) return chunks;
  const cells = insertionCells(chunks);
  if (!cells.length) return chunks;
  if (cells[0].frameMs !== model.frameMs) throw new Error("Elastic scorer clock differs from primary evidence");
  const candidates = new Map<string, SequenceEvidenceCandidate>();
  const tracker = createTracker(index, { mode, phrases, acquireMs: 1000 });
  const checker = createChecker(index, { ...options, phrases, sequenceEvidence(c) { candidates.set(JSON.stringify(c), c); return true; } });
  for (const e of cells) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const features = elasticFeatures(audio), certificates: ElasticEvidence[] = [];
  for (const c of candidates.values()) {
    const comparison = c.fromFrame >= 0 && c.toFrame * 4 <= features.length
      ? elasticComparison(model, features.slice(c.fromFrame * 4, c.toFrame * 4), c) : null;
    inspect?.(comparison, c);
    if (comparison && elasticReversal(comparison)) certificates.push(comparison);
  }
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e, i) => i !== first ? e : { ...e, phonemes: e.phonemes.map((p, j) => j ? p :
    { ...p, elasticVetoes: [...(p.elasticVetoes ?? []), ...certificates] }) });
}
