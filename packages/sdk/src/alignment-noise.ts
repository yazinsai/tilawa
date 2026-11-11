import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { AlignmentNoiseEvidence, CheckerOptions, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface PhoneEmission { mean: number[]; variance: number[]; count: number }
/** Both heads have the same Gaussian emissions, features, timing and training assignments. */
export interface AlignmentNoiseModel {
  schema: 1; inventory: string; tokens: string; frameMs: 40; observationMs: 10; dimensions: 8;
  center: number[]; scale: number[]; control: PhoneEmission[]; noise: PhoneEmission[];
  contamination: number[]; stay: number; maxFrames: 2048;
}
/** Eight fixed mean log-mel coordinates; observations use the primary frontend's exact geometry. */
export class AlignmentObservations {
  readonly frames: number[][] = [];
  private readonly fbank = new Fbank();
  push(audio: Float32Array): void {
    for (const row of this.fbank.push(audio)) this.frames.push(Array.from({ length: 8 }, (_, d) => {
      let sum = 0;
      for (let j = 0; j < 10; j++) sum += row[10*d+j];
      return sum/10;
    }));
  }
}
export function alignmentFeatures(audio: Float32Array): number[][] {
  const observations = new AlignmentObservations();
  observations.push(audio);
  return observations.frames;
}
const logadd = (a: number, b: number): number => {
  const m = Math.max(a,b);
  return Number.isFinite(m) ? m + Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
/** Sum every complete ordered phone placement; normalize the same completion-conditioned timing prior. */
export function boundaryLogP(values: number[][], stay: number): number {
  if (!values.length || !values[0].length || values.length < values[0].length) return -Infinity;
  const states = values[0].length, self = Math.log(stay), step = Math.log1p(-stay);
  let a = Array(states).fill(-Infinity), prior = a.slice();
  a[0] = values[0][0]; prior[0] = 0;
  for (let t = 1; t < values.length; t++) {
    a = a.map((v,s) => logadd(v+self,s ? a[s-1]+step : -Infinity)+values[t][s]);
    prior = prior.map((v,s) => logadd(v+self,s ? prior[s-1]+step : -Infinity));
  }
  return a[states-1]-prior[states-1];
}
export function alignmentScores(model: AlignmentNoiseModel, raw: number[][], sequences: string[]): { control: number[]; noise: number[] } | null {
  if (!raw.length || raw.length > model.maxFrames || raw.some(r => r.length !== 8 || r.some(v => !Number.isFinite(v)))) return null;
  const ids = sequences.map(s => [...s].map(p => model.tokens.indexOf(p)+1));
  if (ids.some(s => !s.length || s.length > raw.length || s.some(p => p < 1 || model.control[p].count <= 0 || model.noise[p].count <= 0))) return null;
  const x = raw.map(r => r.map((v,d) => (v-model.center[d])/model.scale[d]));
  const used = [...new Set(ids.flat())];
  const score = (head: PhoneEmission[]) => {
    const columns = new Map(used.map(p => [p,x.map(r => {
      const e = head[p];
      return -.5*r.reduce((sum,v,d) => sum + Math.log(2*Math.PI*e.variance[d]) + (v-e.mean[d])**2/e.variance[d],0);
    })]));
    return ids.map(s => boundaryLogP(x.map((_,t) => s.map(p => columns.get(p)![t])),model.stay));
  };
  const control = score(model.control), noise = score(model.noise);
  return [...control,...noise].every(Number.isFinite) ? { control, noise } : null;
}
export function alignmentComparison(model: AlignmentNoiseModel, raw: number[][], c: SequenceEvidenceCandidate): AlignmentNoiseEvidence | null {
  const scores = alignmentScores(model,raw,[c.written,c.rival]);
  return scores ? { ...c,controlWritten:scores.control[0],controlRival:scores.control[1],noiseWritten:scores.noise[0],noiseRival:scores.noise[1] } : null;
}
export const alignmentReversal = (c: AlignmentNoiseEvidence): boolean =>
  [c.controlWritten,c.controlRival,c.noiseWritten,c.noiseRival].every(Number.isFinite) &&
  c.controlWritten < c.controlRival && c.noiseWritten > c.noiseRival;
export function validateAlignmentNoiseModel(value: unknown, inventory: string, tokens: string): AlignmentNoiseModel {
  const m = value as AlignmentNoiseModel;
  const vector = (v: number[],positive = false) => Array.isArray(v) && v.length === 8 && v.every(x => Number.isFinite(x) && (!positive || x > 0));
  if (!m || m.schema !== 1 || m.inventory !== inventory || m.tokens !== tokens || m.frameMs !== 40 || m.observationMs !== 10 || m.dimensions !== 8 ||
    !vector(m.center) || !vector(m.scale,true) || !(m.stay > 0 && m.stay < 1) || m.maxFrames !== 2048 ||
    !Array.isArray(m.contamination) || m.contamination.length !== tokens.length+1 || m.contamination.some(p => !Number.isFinite(p) || p < 0 || p >= 1)) throw new Error("Invalid alignment noise model");
  for (const head of [m.control,m.noise]) {
    if (!Array.isArray(head) || head.length !== tokens.length+1) throw new Error("Invalid alignment noise emissions");
    for (const p of head) if (!p || !vector(p.mean) || !vector(p.variance,true) || !Number.isFinite(p.count) || p.count < 0) throw new Error("Invalid phone emission");
  }
  return m;
}
/** Discover only baseline-supported string-path windows; attach finite, exact-pronunciation certificates. */
export function addAlignmentNoiseEvidence(model: AlignmentNoiseModel | undefined, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset, options: CheckerOptions = {}, mode: "free" | "memorization" = "free",
  inspect?: (comparison: AlignmentNoiseEvidence | null,candidate: SequenceEvidenceCandidate) => void): PhonemeEvidence[] {
  if (!model || !audio.length) return chunks;
  const cells = insertionCells(chunks);
  if (!cells.length) return chunks;
  if (cells[0].frameMs !== model.frameMs) throw new Error("Alignment scorer clock differs from primary evidence");
  const candidates = new Map<string,SequenceEvidenceCandidate>();
  const tracker = createTracker(index,{ mode,phrases,acquireMs:1000 });
  const checker = createChecker(index,{ ...options,phrases,sequenceEvidence(c) { candidates.set(JSON.stringify(c),c); return true; } });
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const features = alignmentFeatures(audio), certificates: AlignmentNoiseEvidence[] = [];
  for (const c of candidates.values()) {
    const result = c.fromFrame >= 0 && c.toFrame*4 <= features.length ? alignmentComparison(model,features.slice(c.fromFrame*4,c.toFrame*4),c) : null;
    inspect?.(result,c);
    if (result && alignmentReversal(result)) certificates.push(result);
  }
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e,i) => i !== first ? e : { ...e,phonemes:e.phonemes.map((p,j) => j ? p : { ...p,alignmentNoiseVetoes:[...(p.alignmentNoiseVetoes??[]),...certificates] }) });
}
