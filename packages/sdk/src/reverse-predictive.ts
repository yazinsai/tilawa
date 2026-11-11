import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, ReversePredictiveEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

interface Predictor {
  intercept: number[][];
  coefficient: number[][];
  variance: number[][];
  counts: number[];
}
export interface ReverseModel {
  schema: 1;
  inventory: string;
  tokens: string;
  frameMs: 40;
  observationMs: 10;
  dimensions: 80;
  center: number[];
  scale: number[];
  forward: Predictor;
  reverse: Predictor;
  maxFrames: number;
}

/** Retains the native log-mel trajectory, without recognition logits or time pooling. */
export class ReverseObservations {
  readonly frames: number[][] = [];
  private readonly fbank = new Fbank();
  push(audio: Float32Array): void {
    for (const row of this.fbank.push(audio)) this.frames.push(Array.from(row));
  }
}
export function reverseFeatures(audio: Float32Array): number[][] {
  const observations = new ReverseObservations();
  observations.push(audio);
  return observations.frames;
}
export const predictiveLogadd = (a: number, b: number): number => {
  const m = Math.max(a, b);
  return Number.isFinite(m) ? m + Math.log(Math.exp(a-m) + Math.exp(b-m)) : -Infinity;
};

/** Complete pronunciation chain with optional inter-phone background states. Every phone is
 * consumed, including repeated labels. Each emission predicts the current spectral observation
 * from the preceding observation in the chosen direction. Both arms condition on the same number
 * of frames and normalize by the probability of completing the same timing graph.
 */
export function scorePredictiveSequence(model: ReverseModel, raw: number[][], pronunciation: string, reverse: boolean): number | null {
  const arm = reverse ? model.reverse : model.forward;
  const ids = [...pronunciation].map(p => model.tokens.indexOf(p)+1);
  if (!ids.length || ids.some(p => p < 1 || arm.counts[p] <= 0) || raw.length < ids.length ||
    raw.length > model.maxFrames || raw.some(r => r.length !== 80 || r.some(v => !Number.isFinite(v)))) return null;
  if (reverse) ids.reverse();
  const labels = [0];
  for (const p of ids) labels.push(p, 0);
  const n = labels.length;
  let score = Array<number>(n).fill(-Infinity), prior = score.slice();
  let previous = Array<number>(80).fill(0);
  for (let t=0; t<raw.length; t++) {
    const x = raw[reverse ? raw.length-1-t : t].map((v,d) => (v-model.center[d])/model.scale[d]);
    const emission = new Map<number,number>();
    for (const p of new Set(labels)) emission.set(p, x.reduce((s,v,d) => {
      const residual = v-arm.intercept[p][d]-arm.coefficient[p][d]*previous[d];
      const variance=arm.variance[p][d];
      return s-.5*(Math.log(2*Math.PI*variance)+residual*residual/variance);
    },0));
    const next=score.slice(), nextPrior=prior.slice();
    for (let s=0; s<n; s++) {
      const skip=s>=2 && labels[s]!==0 && labels[s]!==labels[s-2];
      // A label divides its advance mass between the next background and next distinct phone.
      const step=s && labels[s-1]!==0 && s+1<n && labels[s-1]!==labels[s+1] ? .1 : .2;
      const incoming=(v: number[]) => predictiveLogadd(predictiveLogadd(v[s]+Math.log(.8),
        s ? v[s-1]+Math.log(step) : -Infinity),skip ? v[s-2]+Math.log(.1) : -Infinity);
      const start=s<2 ? -Math.log(2) : -Infinity;
      next[s]=(t ? incoming(score) : start)+emission.get(labels[s])!;
      nextPrior[s]=t ? incoming(prior) : start;
    }
    score=next; prior=nextPrior; previous=x;
  }
  const result=predictiveLogadd(score[n-1],score[n-2])-predictiveLogadd(prior[n-1],prior[n-2]);
  return Number.isFinite(result) ? result : null;
}
export function reverseComparison(model: ReverseModel, window: number[][], c: SequenceEvidenceCandidate): ReversePredictiveEvidence | null {
  const forwardWritten=scorePredictiveSequence(model,window,c.written,false);
  const forwardRival=scorePredictiveSequence(model,window,c.rival,false);
  const reverseWritten=scorePredictiveSequence(model,window,c.written,true);
  const reverseRival=scorePredictiveSequence(model,window,c.rival,true);
  if (forwardWritten===null || forwardRival===null || reverseWritten===null || reverseRival===null) return null;
  return { ...c, forwardWritten, forwardRival, reverseWritten, reverseRival };
}
export const reverseReversal = (c: ReversePredictiveEvidence): boolean =>
  [c.forwardWritten,c.forwardRival,c.reverseWritten,c.reverseRival].every(Number.isFinite) &&
  c.forwardWritten<c.forwardRival && c.reverseWritten>c.reverseRival;

export function validateReverseModel(value: unknown, inventory: string, tokens: string): ReverseModel {
  const m=value as ReverseModel, n=tokens.length+1;
  const vector=(v: number[],len: number,positive=false) => Array.isArray(v) && v.length===len && v.every(x=>Number.isFinite(x)&&(!positive||x>0));
  const matrix=(v: number[][],positive=false) => Array.isArray(v) && v.length===n && v.every(r=>vector(r,80,positive));
  const arm=(a: Predictor) => a && matrix(a.intercept) && matrix(a.coefficient) && matrix(a.variance,true) &&
    vector(a.counts,n) && a.counts.every(c=>c>=0);
  if (!m || m.schema!==1 || m.inventory!==inventory || m.tokens!==tokens || m.frameMs!==40 || m.observationMs!==10 || m.dimensions!==80 ||
    !vector(m.center,80) || !vector(m.scale,80,true) || !arm(m.forward) || !arm(m.reverse) ||
    !Number.isInteger(m.maxFrames) || m.maxFrames<1 || m.maxFrames>8192) throw new Error("Invalid reverse predictive model or inventory");
  return m;
}

/** Discover actual confirming comparisons, preserving complete pronunciations and original frames. */
export function addReverseEvidence(model: ReverseModel | undefined, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset, options: CheckerOptions = {}, mode: "free" | "memorization" = "free",
  inspect?: (c: ReversePredictiveEvidence | null, candidate: SequenceEvidenceCandidate) => void): PhonemeEvidence[] {
  if (!model || !audio.length) return chunks;
  const cells=insertionCells(chunks);
  if (!cells.length) return chunks;
  if (cells[0].frameMs!==model.frameMs) throw new Error("Reverse scorer clock differs from primary evidence");
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c) { candidates.set(JSON.stringify(c),c); return true; }});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const features=reverseFeatures(audio), certificates: ReversePredictiveEvidence[]=[];
  for (const c of candidates.values()) {
    const comparison=c.fromFrame>=0 && c.toFrame*4<=features.length
      ? reverseComparison(model,features.slice(c.fromFrame*4,c.toFrame*4),c) : null;
    inspect?.(comparison,c);
    if (comparison && reverseReversal(comparison)) certificates.push(comparison);
  }
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,reverseVetoes:[...(p.reverseVetoes??[]),...certificates]})});
}
