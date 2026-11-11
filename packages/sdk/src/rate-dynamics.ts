import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, PhonemeEvidence, PhraseAsset, ReferenceIndex, RateEvidence, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface RateParameters {
  targets: number[][];
  counts: number[];
  /** Treatment: retention of the preceding observed acoustic state per band.
   * Control: log variance scale for independent realizations per band.
   * Both arms have exactly phoneCount * dimensions + dimensions parameters. */
  shared: number[];
}
export interface RateModel {
  schema: 1;
  inventory: string;
  tokens: string;
  frameMs: 40;
  observationMs: 10;
  dimensions: 20;
  maxFrames: number;
  mean: number[];
  scale: number[];
  innovation: number[];
  dynamic: RateParameters;
  control: RateParameters;
}

/** Native 25 ms windows, 10 ms hop. Normalized spectral mass removes overall
 * recording gain. No playback speed or primary features are changed. */
export class RateObservations {
  readonly frames: number[][] = [];
  private readonly fbank = new Fbank();
  push(audio: Float32Array): void {
    for (const row of this.fbank.push(audio)) {
      const power = Array.from({ length: 20 }, (_, d) =>
        Math.exp(row[4*d])+Math.exp(row[4*d+1])+Math.exp(row[4*d+2])+Math.exp(row[4*d+3]));
      const mass = power.reduce((s,p)=>s+p,0);
      this.frames.push(power.map(p=>Math.log(p/mass)));
    }
  }
}
export function rateFeatures(audio: Float32Array): number[][] {
  const observations=new RateObservations();
  observations.push(audio);
  return observations.frames;
}
export function logadd(a: number,b: number): number {
  const m=Math.max(a,b);
  return Number.isFinite(m) ? m+Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
}

/** Conditional Markov acoustic dynamics approach each phone's target:
 * x[t] = retention*x[t-1] + (1-retention)*target[phone] + innovation.
 * After d frames its expected shortfall contracts by retention**d. The control
 * draws independent realizations about a target and learns dispersion instead
 * of retention, with the same number of parameters and training observations.
 * The acoustic state is the observed spectrum, not a listener spike or a phone
 * boundary annotation. Every complete segmentation is summed exactly. */
export function scoreRateSequence(model: RateModel, raw: number[][], pronunciation: string, dynamic: boolean,
  previous?: number[]): number|null {
  const m=dynamic ? model.dynamic : model.control;
  const ids=[...pronunciation].map(p=>model.tokens.indexOf(p)+1), n=ids.length;
  if (!n || ids.some(p=>p<1 || m.counts[p]<=0) || raw.length<n || raw.length>model.maxFrames ||
    raw.some(r=>r.length!==model.dimensions || r.some(v=>!Number.isFinite(v))) ||
    (previous && (previous.length!==model.dimensions || previous.some(v=>!Number.isFinite(v))))) return null;
  const x=raw.map(r=>r.map((v,d)=>(v-model.mean[d])/model.scale[d]));
  const initial=previous ? previous.map((v,d)=>(v-model.mean[d])/model.scale[d]) : Array<number>(20).fill(0);
  let prior=Array<number>(n).fill(-Infinity), score=prior.slice(); prior[0]=0;
  const density=(t:number,s:number)=>x[t].reduce((value,v,d)=> {
    const rho=dynamic ? m.shared[d] : 0;
    const mean=rho*(t ? x[t-1][d] : initial[d])+(1-rho)*m.targets[ids[s]][d];
    const variance=model.innovation[d]*(dynamic ? 1 : Math.exp(m.shared[d]));
    return value-.5*(Math.log(2*Math.PI*variance)+(v-mean)**2/variance);
  },0);
  score[0]=density(0,0);
  // Equal fixed self/advance weights cancel conditional on completion; the
  // partition below normalizes the uniform prior over all complete timings.
  for (let t=1;t<x.length;t++) {
    const next=Array<number>(n).fill(-Infinity), nextPrior=next.slice();
    for (let s=Math.max(0,n-x.length+t);s<=Math.min(n-1,t);s++) {
      nextPrior[s]=logadd(prior[s],s ? prior[s-1] : -Infinity);
      next[s]=logadd(score[s],s ? score[s-1] : -Infinity)+density(t,s);
    }
    score=next; prior=nextPrior;
  }
  const result=score[n-1]-prior[n-1];
  return Number.isFinite(result) ? result : null;
}
export function rateComparison(m: RateModel,raw: number[][],c: SequenceEvidenceCandidate,previous?:number[]): RateEvidence|null {
  const controlWritten=scoreRateSequence(m,raw,c.written,false,previous), controlRival=scoreRateSequence(m,raw,c.rival,false,previous);
  const dynamicWritten=scoreRateSequence(m,raw,c.written,true,previous), dynamicRival=scoreRateSequence(m,raw,c.rival,true,previous);
  if (controlWritten===null || controlRival===null || dynamicWritten===null || dynamicRival===null) return null;
  return {...c,controlWritten,controlRival,dynamicWritten,dynamicRival};
}
export const rateReversal=(c:RateEvidence):boolean=>
  [c.controlWritten,c.controlRival,c.dynamicWritten,c.dynamicRival].every(Number.isFinite) &&
  c.controlWritten<c.controlRival && c.dynamicWritten>c.dynamicRival;

export function validateRateModel(value:unknown,inventory:string,tokens:string):RateModel {
  const m=value as RateModel,n=tokens.length+1;
  const vector=(v:number[],len:number,positive=false)=>Array.isArray(v)&&v.length===len&&v.every(x=>Number.isFinite(x)&&(!positive||x>0));
  const matrix=(v:number[][])=>Array.isArray(v)&&v.length===n&&v.every(r=>vector(r,20));
  const valid=(p:RateParameters)=>p && matrix(p.targets) && vector(p.counts,n) && p.counts.every(v=>v>=0) && vector(p.shared,20);
  if (!m || m.schema!==1 || m.inventory!==inventory || m.tokens!==tokens || m.frameMs!==40 || m.observationMs!==10 || m.dimensions!==20 ||
    !Number.isInteger(m.maxFrames) || m.maxFrames<1 || m.maxFrames>8192 || !vector(m.mean,20) || !vector(m.scale,20,true) ||
    !vector(m.innovation,20,true) || !valid(m.dynamic) || !valid(m.control) || m.dynamic.shared.some(v=>v<0||v>=1) ||
    m.control.shared.some(v=>Math.abs(v)>10)) throw new Error("Invalid rate spectral model or inventory");
  return m;
}
/** Generate certificates from the actual confirming comparisons, after existing
 * gap and insertion enrichment. Only metadata is added; primary rows stay intact.
 */
export function addRateEvidence(m: RateModel | undefined, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset, options: CheckerOptions = {}, mode: "free" | "memorization" = "free",
  inspect?: (comparison: RateEvidence | null, candidate: SequenceEvidenceCandidate) => void): PhonemeEvidence[] {
  if (!m || !audio.length) return chunks;
  const cells=insertionCells(chunks);
  if (!cells.length) return chunks;
  if (cells[0].frameMs!==m.frameMs) throw new Error("Rate scorer clock differs from primary evidence");
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c) { candidates.set(JSON.stringify(c),c); return true; }});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const features=rateFeatures(audio), certificates:RateEvidence[]=[];
  for (const c of candidates.values()) {
    const comparison=c.fromFrame>=0 && c.toFrame*4<=features.length
      ? rateComparison(m,features.slice(c.fromFrame*4,c.toFrame*4),c,features[c.fromFrame*4-1]) : null;
    inspect?.(comparison,c);
    if (comparison && rateReversal(comparison)) certificates.push(comparison);
  }
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first ? e : {...e,phonemes:e.phonemes.map((p,j)=>j ? p :
    {...p,rateVetoes:[...(p.rateVetoes??[]),...certificates]})});
}
