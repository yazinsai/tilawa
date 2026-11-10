import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, GestureEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface GestureModel {
  schema: 1;
  inventory: string;
  tokens: string;
  frameMs: 40;
  bands: number;
  order: number;
  center: number[];
  scale: number[];
  maxSegment: number;
  maxFrames: number;
  counts: number[];
  signature: { means: number[][]; variances: number[][] };
  control: { means: number[][]; variances: number[][] };
}

/** Keep the primary frontend clock; coordinate pooling is independent of the candidate text. */
export class GestureObservations {
  readonly frames: number[][] = [];
  private readonly fbank = new Fbank();
  private at = 0;
  push(audio: Float32Array): void {
    for (const row of this.fbank.push(audio)) {
      if (this.at++ % 4 === 0) this.frames.push(Array.from(row));
    }
  }
}
export function gestureFeatures(audio: Float32Array): number[][] {
  const o = new GestureObservations();
  o.push(audio);
  return o.frames;
}

const tensor = (a: number[], b: number[]) => a.flatMap(x => b.map(y => x*y));
const logadd = (a: number, b: number): number => {
  const m = Math.max(a,b);
  return Number.isFinite(m) ? m + Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
export const gestureDimensions = (bands: number, order: number) =>
  bands + Array.from({length: order}, (_,k) => bands**(k+1)).reduce((a,b)=>a+b,0);

/** Chen's product of segment exponentials. The start coordinate retains absolute articulation;
 * there is no time channel. Collinear subdivision and monotone traversal speed leave this unchanged.
 */
export function pathSignature(path: number[][], order: number): number[] {
  const d = path[0].length;
  let levels = Array.from({length: order+1}, (_,k) => Array<number>(d**k).fill(k ? 0 : 1));
  for (let t=1; t<path.length; t++) {
    const dx = path[t].map((v,j) => v-path[t-1][j]);
    const exp: number[][] = [[1]];
    for (let k=1;k<=order;k++) exp.push(tensor(exp[k-1],dx).map(v=>v/k));
    const next = levels.map(a=>a.slice());
    for (let k=1;k<=order;k++) for (let j=0;j<k;j++) {
      const term=tensor(levels[j],exp[k-j]);
      for (let i=0;i<term.length;i++) next[k][i]+=term[i];
    }
    levels=next;
  }
  return [...path[0],...levels.slice(1).flat()];
}

/** Exactly as many coordinates and fitted Gaussian parameters as the signature arm.
 * Sample at uniformly spaced traversal times; differing within-segment pacing changes this arm.
 */
export function timeTrajectory(path: number[][], order: number): number[] {
  const d=path[0].length, n=gestureDimensions(d,order)/d;
  return Array.from({length:n},(_,k)=> {
    const t=k*(path.length-1)/(n-1), a=Math.floor(t), b=Math.min(path.length-1,a+1);
    return path[a].map((v,j)=>v+(path[b][j]-v)*(t-a));
  }).flat();
}

export function normalizedGesture(model: GestureModel, raw: number[][]): number[][] {
  return raw.map(row=>Array.from({length:model.bands},(_,d)=> {
    const a=Math.floor(d*80/model.bands), b=Math.floor((d+1)*80/model.bands);
    const value=row.slice(a,b).reduce((s,v)=>s+v,0)/(b-a);
    return (value-model.center[d])/model.scale[d];
  }));
}

/** Semi-Markov forward sum over every positive-duration phone boundary, interleaved with
 * optional silence segments. Both arms use the same uniform, completion-conditioned timing law.
 * Every written or rival phone must occur; silence can never substitute for a missing phone.
 */
export function scoreGestureSequence(model: GestureModel, raw: number[][], pronunciation: string, signature: boolean): number | null {
  const ids=[...pronunciation].map(p=>model.tokens.indexOf(p)+1);
  if (!ids.length || ids.some(p=>p<1 || model.counts[p]<=0) || raw.length<ids.length || raw.length>model.maxFrames ||
    raw.some(r=>r.length!==80 || r.some(v=>!Number.isFinite(v)))) return null;
  const path=normalizedGesture(model,raw), tmax=path.length, arm=signature?model.signature:model.control;
  const unique=new Set([0,...ids]);
  const emissions=new Map<number,number[][]>();
  for (const p of unique) emissions.set(p,Array.from({length:tmax},()=>[]));
  for (let a=0;a<tmax;a++) for (let len=1;len<=model.maxSegment && a+len<=tmax;len++) {
    const y=signature?pathSignature(path.slice(a,a+len),model.order):timeTrajectory(path.slice(a,a+len),model.order);
    for (const p of unique) {
      const mu=arm.means[p], v=arm.variances[p];
      emissions.get(p)![a][len]=y.reduce((s,x,d)=>s-.5*(Math.log(2*Math.PI*v[d])+(x-mu[d])**2/v[d]),0);
    }
  }
  let alpha=Array<number>(tmax+1).fill(-Infinity), prior=alpha.slice();
  alpha[0]=0; prior[0]=0;
  for (const [s,p] of [0,...ids.flatMap(p=>[p,0])].entries()) {
    const next=Array<number>(tmax+1).fill(-Infinity), nextPrior=next.slice();
    for (let end=0;end<=tmax;end++) {
      if (p===0) { next[end]=alpha[end]; nextPrior[end]=prior[end]; }
      for (let len=1;len<=model.maxSegment && len<=end;len++) {
        const start=end-len, e=emissions.get(p)![start][len];
        next[end]=logadd(next[end],alpha[start]+e);
        nextPrior[end]=logadd(nextPrior[end],prior[start]);
      }
    }
    alpha=next; prior=nextPrior;
  }
  const result=alpha[tmax]-prior[tmax];
  return Number.isFinite(result)?result:null;
}

export function gestureComparison(model: GestureModel, raw: number[][], c: SequenceEvidenceCandidate): GestureEvidence | null {
  const controlWritten=scoreGestureSequence(model,raw,c.written,false), controlRival=scoreGestureSequence(model,raw,c.rival,false);
  const orderedWritten=scoreGestureSequence(model,raw,c.written,true), orderedRival=scoreGestureSequence(model,raw,c.rival,true);
  if ([controlWritten,controlRival,orderedWritten,orderedRival].some(v=>v===null)) return null;
  return {...c,controlWritten:controlWritten!,controlRival:controlRival!,orderedWritten:orderedWritten!,orderedRival:orderedRival!};
}
export const gestureReversal=(c: GestureEvidence): boolean =>
  [c.controlWritten,c.controlRival,c.orderedWritten,c.orderedRival].every(Number.isFinite) &&
  c.controlWritten<c.controlRival && c.orderedWritten>c.orderedRival;

export function validateGestureModel(value: unknown, inventory: string, tokens: string): GestureModel {
  const m=value as GestureModel, n=tokens.length+1;
  const vector=(v: number[],len: number,positive=false)=>Array.isArray(v)&&v.length===len&&v.every(x=>Number.isFinite(x)&&(!positive||x>0));
  if (!m || m.schema!==1 || m.inventory!==inventory || m.tokens!==tokens || m.frameMs!==40 ||
    ![2,3].includes(m.bands) || ![2,3].includes(m.order) || !vector(m.center,m.bands) || !vector(m.scale,m.bands,true) ||
    !vector(m.counts,n) || m.counts.some(c=>c<0) || !Number.isInteger(m.maxSegment) || m.maxSegment<1 || m.maxSegment>64 ||
    !Number.isInteger(m.maxFrames) || m.maxFrames<1 || m.maxFrames>2048) throw new Error("Invalid gesture acoustic model");
  const dimensions=gestureDimensions(m.bands,m.order);
  for (const arm of [m.signature,m.control]) if (!arm || !Array.isArray(arm.means) || arm.means.length!==n ||
    !arm.means.every(v=>vector(v,dimensions)) || !Array.isArray(arm.variances) || arm.variances.length!==n ||
    !arm.variances.every(v=>vector(v,dimensions,true))) throw new Error("Invalid gesture emission parameters");
  return m;
}

/** Replay primary evidence to discover the exact complete comparisons the checker actually used. */
export function addGestureEvidence(model: GestureModel | undefined, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset, options: CheckerOptions = {}, mode: "free" | "memorization" = "free",
  inspect?: (c: GestureEvidence | null, candidate: SequenceEvidenceCandidate) => void): PhonemeEvidence[] {
  if (!model || !audio.length) return chunks;
  const cells=insertionCells(chunks);
  if (!cells.length) return chunks;
  if (cells[0].frameMs!==model.frameMs) throw new Error("Gesture scorer clock differs from primary evidence");
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c) {candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const features=gestureFeatures(audio), certificates: GestureEvidence[]=[];
  for (const c of candidates.values()) {
    const comparison=c.fromFrame>=0 && c.toFrame<=features.length
      ? gestureComparison(model,features.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(comparison,c);
    if (comparison && gestureReversal(comparison)) certificates.push(comparison);
  }
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,gestureVetoes:[...(p.gestureVetoes??[]),...certificates]})});
}
