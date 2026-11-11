import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, TrainingSupportEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface EmissionFit {
  center: number[];
  scale: number[];
  phones: { mean: number[]; variance: number[]; count: number }[];
}
export interface TrainingSupportModel {
  schema: 1;
  inventory: string;
  tokens: string;
  frameMs: 40;
  observationMs: 10;
  dimensions: 8;
  stay: number;
  maxFrames: number;
  cutoff: number;
  full: EmissionFit;
  /** Each fit excludes every frame of exactly one source recording. */
  deletions: EmissionFit[];
}

/** Fixed eight-coordinate acoustic view; no trainable encoder or recognition output. */
export class SupportObservations {
  readonly frames: number[][] = [];
  private readonly fbank = new Fbank();
  push(audio: Float32Array): void {
    for (const row of this.fbank.push(audio)) {
      this.frames.push(Array.from({ length: 8 }, (_, d) => {
        let sum = 0;
        for (let j = 0; j < 10; j++) sum += row[10 * d + j];
        return sum / 10;
      }));
    }
  }
}
export function supportFeatures(audio: Float32Array): number[][] {
  const observations = new SupportObservations();
  observations.push(audio);
  return observations.frames;
}
function logadd(a: number, b: number): number {
  const m = Math.max(a, b);
  return Number.isFinite(m) ? m + Math.log(Math.exp(a - m) + Math.exp(b - m)) : -Infinity;
}
/** Complete ordered pronunciation; all phone boundaries marginalized. Timing prior conditioned on completion. */
export function boundaryLogP(values: number[][], stay: number): number {
  if (!values.length || !values[0].length || values.length < values[0].length) return -Infinity;
  const n = values[0].length, self = Math.log(stay), step = Math.log1p(-stay);
  let a = Array(n).fill(-Infinity), prior = a.slice();
  a[0] = values[0][0]; prior[0] = 0;
  for (let t = 1; t < values.length; t++) {
    a = a.map((v, s) => logadd(v + self, s ? a[s - 1] + step : -Infinity) + values[t][s]);
    prior = prior.map((v, s) => logadd(v + self, s ? prior[s - 1] + step : -Infinity));
  }
  return a[n - 1] - prior[n - 1];
}
export function supportScores(model: TrainingSupportModel, fit: EmissionFit, raw: number[][], sequences: string[]): number[] | null {
  if (!raw.length || raw.length > model.maxFrames || raw.some(r => r.length !== 8 || r.some(v => !Number.isFinite(v)))) return null;
  const ids = sequences.map(s => [...s].map(p => model.tokens.indexOf(p) + 1));
  if (ids.some(s => !s.length || s.length > raw.length || s.some(id => id < 1 || fit.phones[id].count <= 0))) return null;
  const x = raw.map(r => r.map((v, d) => (v - fit.center[d]) / fit.scale[d]));
  const emissions = new Map([...new Set(ids.flat())].map(p => {
    const phone = fit.phones[p];
    // Include the normalization Jacobian: deletions refit their own standardization.
    const normalizer = phone.variance.reduce((s, v, d) => s + Math.log(2 * Math.PI * v) + 2 * Math.log(fit.scale[d]), 0);
    return [p, x.map(r => -.5 * (normalizer + r.reduce((s, v, d) => s + (v - phone.mean[d]) ** 2 / phone.variance[d], 0)))];
  }));
  const scores = ids.map(s => boundaryLogP(raw.map((_, t) => s.map(p => emissions.get(p)![t])), model.stay));
  return scores.every(Number.isFinite) ? scores : null;
}
export function supportComparison(model: TrainingSupportModel, raw: number[][], candidate: SequenceEvidenceCandidate): TrainingSupportEvidence | null {
  const full = supportScores(model, model.full, raw, [candidate.written, candidate.rival]);
  if (!full) return null;
  const fullPreference = full[0] - full[1];
  let best: { scores: number[]; influence: number; recording: number } | null = null;
  for (let recording = 0; recording < model.deletions.length; recording++) {
    const scores = supportScores(model, model.deletions[recording], raw, [candidate.written, candidate.rival]);
    if (!scores) continue;
    const influence = (scores[0] - scores[1] - fullPreference) / raw.length;
    if (!best || Math.abs(influence) > Math.abs(best.influence)) best = { scores, influence, recording };
  }
  if (!best) return null;
  return { ...candidate, fullWritten: full[0], fullRival: full[1], removedWritten: best.scores[0],
    removedRival: best.scores[1], influence: best.influence, recording: best.recording, cutoff: model.cutoff };
}
export function supportReversal(c: TrainingSupportEvidence): boolean {
  return [c.fullWritten, c.fullRival, c.removedWritten, c.removedRival, c.influence, c.cutoff].every(Number.isFinite) &&
    c.fullWritten < c.fullRival && c.removedWritten > c.removedRival && c.cutoff >= 0 && c.influence > c.cutoff &&
    Number.isInteger(c.recording) && c.recording >= 0;
}
export function validateTrainingSupportModel(value: unknown, inventory: string, tokens: string): TrainingSupportModel {
  const m = value as TrainingSupportModel;
  const vector = (v: number[], positive = false) => Array.isArray(v) && v.length === 8 && v.every(x => Number.isFinite(x) && (!positive || x > 0));
  if (!m || m.schema !== 1 || m.inventory !== inventory || m.tokens !== tokens || m.frameMs !== 40 || m.observationMs !== 10 ||
    m.dimensions !== 8 || !(m.stay > 0 && m.stay < 1) || !Number.isInteger(m.maxFrames) || m.maxFrames < 1 ||
    !Number.isFinite(m.cutoff) || m.cutoff < 0 || !Array.isArray(m.deletions) || m.deletions.length < 2)
    throw new Error("Invalid recording-support model");
  for (const fit of [m.full, ...m.deletions]) {
    if (!fit || !vector(fit.center) || !vector(fit.scale, true) || !Array.isArray(fit.phones) || fit.phones.length !== tokens.length + 1)
      throw new Error("Invalid recording-support fit");
    for (const p of fit.phones) if (!p || !vector(p.mean) || !vector(p.variance, true) || !Number.isFinite(p.count) || p.count < 0)
      throw new Error("Invalid recording-support emission");
  }
  return m;
}
/** Discover comparisons with the unchanged tracker and confirmation logic, never from labels. */
export function supportCandidates(chunks: PhonemeEvidence[], index: ReferenceIndex, phrases?: PhraseAsset,
  options: CheckerOptions = {}, mode: "free" | "memorization" = "free"): SequenceEvidenceCandidate[] {
  const candidates = new Map<string, SequenceEvidenceCandidate>();
  const tracker = createTracker(index, { mode, phrases, acquireMs: 1000 });
  const checker = createChecker(index, { ...options, phrases, sequenceEvidence(c) { candidates.set(JSON.stringify(c), c); return true; } });
  for (const e of insertionCells(chunks)) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  return [...candidates.values()];
}
export function addTrainingSupportEvidence(model: TrainingSupportModel | undefined, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset, options: CheckerOptions = {}, mode: "free" | "memorization" = "free",
  inspect?: (result: TrainingSupportEvidence | null, candidate: SequenceEvidenceCandidate) => void): PhonemeEvidence[] {
  if (!model || !audio.length) return chunks;
  if (chunks.some(e => e.frameMs !== model.frameMs)) throw new Error("Recording-support clock differs from primary evidence");
  const features = supportFeatures(audio), certificates: TrainingSupportEvidence[] = [];
  for (const c of supportCandidates(chunks, index, phrases, options, mode)) {
    const result = c.fromFrame >= 0 && c.toFrame * 4 <= features.length ? supportComparison(model, features.slice(c.fromFrame * 4, c.toFrame * 4), c) : null;
    inspect?.(result, c);
    if (result && supportReversal(result)) certificates.push(result);
  }
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e, i) => i !== first ? e : { ...e, phonemes: e.phonemes.map((p, j) => j ? p : {
    ...p, trainingSupportVetoes: [...(p.trainingSupportVetoes ?? []), ...certificates],
  }) });
}
