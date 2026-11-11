import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, PhonemeEvidence, PhraseAsset, ReferenceIndex, GestureEvidence, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface GestureParameters {
  spectra: number[][];
  variances: number[][];
  counts: number[];
  stay: number[];
  weights: [number, number];
}
/** Two independently fitted, equally sized models on identical observations.
 * Each phone has ordered attack/sustain/release spectra. Only the source of the
 * kernel's adjacent contributions differs between overlap and control.
 */
export interface GestureModel {
  schema: 1;
  inventory: string;
  tokens: string;
  frameMs: 40;
  observationMs: 10;
  dimensions: 20;
  maxFrames: number;
  overlap: GestureParameters;
  control: GestureParameters;
}

/** Same analysis windows and mel powers as the primary frontend, retaining its
 * 10 ms observations. Sum groups of four mel bins and normalize spectral mass.
 */
export class GestureObservations {
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
export function gestureFeatures(audio: Float32Array): number[][] {
  const observations=new GestureObservations();
  observations.push(audio);
  return observations.frames;
}
export function logadd(a: number,b: number): number {
  const m=Math.max(a,b);
  return Number.isFinite(m) ? m+Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
}

/** Shared normalized gesture dynamics. Earlier gestures fade through the phone;
 * the following attack contributes increasingly before its own boundary. The
 * control uses the current phone's attack/release at those same kernel positions.
 * Both use the same neutral padding outside the complete candidate sequence.
 */
export function gestureMeans(m: GestureParameters, ids: number[], overlap: boolean): number[][] {
  return ids.flatMap((phone,i)=>[0,1,2].map(phase=> {
    const left=i===0 ? 0 : overlap ? ids[i-1]*3+2 : phone*3;
    const right=i===ids.length-1 ? 0 : overlap ? ids[i+1]*3 : phone*3+2;
    const a=m.weights[0]*(1-phase/2), b=m.weights[1]*phase/2, norm=1+a+b;
    return m.spectra[phone*3+phase].map((v,d)=>Math.log((v+a*m.spectra[left][d]+b*m.spectra[right][d])/norm));
  }));
}

/** Exact forward sum over all phone and phase boundaries, divided by the same
 * complete-sequence timing partition. No phone or phase can be erased, including
 * equal adjacent labels; different sequence lengths get normalized timing priors.
 */
export function scoreGestureSequence(model: GestureModel, raw: number[][], pronunciation: string, overlap: boolean): number|null {
  const m=overlap ? model.overlap : model.control;
  const ids=[...pronunciation].map(p=>model.tokens.indexOf(p)+1), n=ids.length*3;
  if (!n || ids.some(p=>p<1 || [0,1,2].some(k=>m.counts[p*3+k]<=0)) || raw.length<n || raw.length>model.maxFrames ||
    raw.some(r=>r.length!==model.dimensions || r.some(v=>!Number.isFinite(v)))) return null;
  const means=gestureMeans(m,ids,overlap);
  const states=ids.flatMap(p=>[p*3,p*3+1,p*3+2]);
  const stays=states.map(p=>Math.log(m.stay[p])), advances=states.map(p=>Math.log1p(-m.stay[p]));
  let prior=Array<number>(n).fill(-Infinity), score=prior.slice(); prior[0]=0;
  const density=(t:number,s:number)=>raw[t].reduce((value,x,d)=> {
    const v=m.variances[states[s]][d], residual=x-means[s][d];
    return value-.5*(Math.log(2*Math.PI*v)+residual*residual/v);
  },0);
  score[0]=density(0,0);
  for (let t=1;t<raw.length;t++) {
    const next=Array<number>(n).fill(-Infinity), nextPrior=next.slice();
    for (let s=Math.max(0,n-raw.length+t);s<=Math.min(n-1,t);s++) {
      nextPrior[s]=logadd(prior[s]+stays[s],s ? prior[s-1]+advances[s-1] : -Infinity);
      next[s]=logadd(score[s]+stays[s],s ? score[s-1]+advances[s-1] : -Infinity)+density(t,s);
    }
    score=next; prior=nextPrior;
  }
  const result=score[n-1]-prior[n-1];
  return Number.isFinite(result) ? result : null;
}
export function gestureComparison(m: GestureModel,raw: number[][],c: SequenceEvidenceCandidate): GestureEvidence|null {
  const controlWritten=scoreGestureSequence(m,raw,c.written,false), controlRival=scoreGestureSequence(m,raw,c.rival,false);
  const overlapWritten=scoreGestureSequence(m,raw,c.written,true), overlapRival=scoreGestureSequence(m,raw,c.rival,true);
  if (controlWritten===null || controlRival===null || overlapWritten===null || overlapRival===null) return null;
  return {...c,controlWritten,controlRival,overlapWritten,overlapRival};
}
export const gestureReversal=(c:GestureEvidence):boolean=>
  [c.controlWritten,c.controlRival,c.overlapWritten,c.overlapRival].every(Number.isFinite) &&
  c.controlWritten<c.controlRival && c.overlapWritten>c.overlapRival;

export function validateGestureModel(value:unknown,inventory:string,tokens:string):GestureModel {
  const m=value as GestureModel,n=(tokens.length+1)*3;
  const vector=(v:number[],len:number,positive=false)=>Array.isArray(v)&&v.length===len&&v.every(x=>Number.isFinite(x)&&(!positive||x>0));
  const matrix=(v:number[][])=>Array.isArray(v)&&v.length===n&&v.every(r=>vector(r,20,true));
  const valid=(p:GestureParameters)=>p && matrix(p.spectra) && p.spectra.every(r=>Math.abs(r.reduce((s,v)=>s+v,0)-1)<1e-5) &&
    matrix(p.variances) && vector(p.counts,n) && p.counts.every(v=>v>=0) &&
    vector(p.stay,n,true) && p.stay.every(v=>v<1) && vector(p.weights,2,true);
  if (!m || m.schema!==1 || m.inventory!==inventory || m.tokens!==tokens || m.frameMs!==40 || m.observationMs!==10 || m.dimensions!==20 ||
    !Number.isInteger(m.maxFrames) || m.maxFrames<3 || m.maxFrames>8192 || !valid(m.overlap) || !valid(m.control))
    throw new Error("Invalid gesture spectral model or inventory");
  return m;
}
/** Generate certificates from the actual confirming comparisons, after existing
 * gap and insertion enrichment. Only metadata is added; primary rows stay intact.
 */
export function addGestureEvidence(m: GestureModel | undefined, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset, options: CheckerOptions = {}, mode: "free" | "memorization" = "free",
  inspect?: (comparison: GestureEvidence | null, candidate: SequenceEvidenceCandidate) => void): PhonemeEvidence[] {
  if (!m || !audio.length) return chunks;
  const cells=insertionCells(chunks);
  if (!cells.length) return chunks;
  if (cells[0].frameMs!==m.frameMs) throw new Error("Gesture scorer clock differs from primary evidence");
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c) { candidates.set(JSON.stringify(c),c); return true; }});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const features=gestureFeatures(audio), certificates:GestureEvidence[]=[];
  for (const c of candidates.values()) {
    const comparison=c.fromFrame>=0 && c.toFrame*4<=features.length
      ? gestureComparison(m,features.slice(c.fromFrame*4,c.toFrame*4),c) : null;
    inspect?.(comparison,c);
    if (comparison && gestureReversal(comparison)) certificates.push(comparison);
  }
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first ? e : {...e,phonemes:e.phonemes.map((p,j)=>j ? p :
    {...p,gestureVetoes:[...(p.gestureVetoes??[]),...certificates]})});
}
