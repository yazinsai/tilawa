import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, RegularityEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { insertionCells } from "./seam-insertion.js";

export interface EmissionModel {
  center: number[];
  scale: number[];
  means: number[][];
  variances: number[][];
  counts: number[];
}
export interface RegularityModel {
  schema: 1;
  inventory: string;
  tokens: string;
  frameMs: 40;
  observationMs: 10;
  dimensions: 12;
  scales: number[];
  stay: number;
  maxFrames: number;
  regularity: EmissionModel;
  energy: EmissionModel;
}
export interface RegularityObservation { regularity: number[]; energy: number[] }

/** Symmetric extension of a single analysis window; no unseen audio is synthesized. */
const reflect = (i: number, n: number): number => {
  const j = ((i % (2*n)) + 2*n) % (2*n);
  return j < n ? j : 2*n-1-j;
};

/** Local Haar coefficient scaling, normalized by the same window RMS in both arms.
 * Pointwise log-magnitude slopes retain whether interruptions persist across scales.
 * The control pools coefficient power first, discarding that local scaling structure.
 */
export function waveletObservation(wave: Float32Array, scales: number[]): RegularityObservation {
  const n=wave.length;
  const mean=wave.reduce((a,b)=>a+b,0)/n;
  const x=Array.from(wave,v=>v-mean);
  const rms=Math.sqrt(x.reduce((a,b)=>a+b*b,0)/n);
  if (!Number.isFinite(rms)) return {regularity:Array(12).fill(NaN),energy:Array(12).fill(NaN)};
  if (rms===0) return {regularity:Array(12).fill(0),energy:Array(12).fill(0)};
  const radius=Math.max(...scales);
  const prefix=Array<number>(n+2*radius+1).fill(0);
  for (let i=0;i<n+2*radius;i++) prefix[i+1]=prefix[i]+x[reflect(i-radius,n)];
  const coefficients=scales.map(s=> {
    const values=Array<number>(n);
    for (let i=0;i<n;i++) {
      const center=i+radius;
      const left=prefix[center]-prefix[center-s],right=prefix[center+s]-prefix[center];
      values[i]=(right-left)/(Math.sqrt(2*s)*rms);
    }
    return values;
  });
  const regularity: number[]=[];
  for (let k=0;k<scales.length-1;k++) {
    const denominator=Math.log(scales[k+1]/scales[k]);
    const slopes=coefficients[k].map((v,i)=>(Math.log(Math.abs(coefficients[k+1][i])+1e-3)-Math.log(Math.abs(v)+1e-3))/denominator);
    const average=slopes.reduce((a,b)=>a+b,0)/n;
    regularity.push(average,Math.sqrt(slopes.reduce((a,b)=>a+(b-average)**2,0)/n));
  }
  // Twelve independent temporal coordinates give the control the same emission
  // capacity. Pool across scales before observing power; no local scaling slope remains.
  const energy=Array.from({length:12},(_,d)=> {
    const from=Math.floor(d*n/12),to=Math.floor((d+1)*n/12);
    let power=0;
    for (const c of coefficients) for (let i=from;i<to;i++) power+=c[i]*c[i];
    return Math.log(power/((to-from)*scales.length)+1e-6);
  });
  return {regularity,energy};
}

/** 25 ms waveform windows at the native 10 ms observation clock, independent of pushes. */
export class RegularityObservations {
  readonly frames: RegularityObservation[]=[];
  private held=new Float32Array(0);
  constructor(readonly scales: number[]) {}
  push(audio: Float32Array): void {
    const joined=new Float32Array(this.held.length+audio.length);
    joined.set(this.held); joined.set(audio,this.held.length);
    let offset=0;
    for (;offset+400<=joined.length;offset+=160) this.frames.push(waveletObservation(joined.subarray(offset,offset+400),this.scales));
    this.held=joined.slice(offset);
  }
}
export function regularityFeatures(audio: Float32Array, scales: number[]): RegularityObservation[] {
  const observations=new RegularityObservations(scales); observations.push(audio); return observations.frames;
}
export const regularityLogadd=(a: number,b: number): number=> {
  const m=Math.max(a,b); return Number.isFinite(m)?m+Math.log(Math.exp(a-m)+Math.exp(b-m)):-Infinity;
};

/** All phones are mandatory. Self loops sum all phone boundaries; an emission-free
 * partition conditions the common timing prior on completing the entire pronunciation.
 * Repeated labels remain distinct articulations in both independently fitted arms.
 */
export function scoreRegularitySequence(model: RegularityModel, raw: RegularityObservation[], pronunciation: string,
  arm: "regularity" | "energy"): number | null {
  const ids=[...pronunciation].map(p=>model.tokens.indexOf(p)+1), m=model[arm];
  if (!ids.length || ids.some(p=>p<1||m.counts[p]<=0) || raw.length<ids.length || raw.length>model.maxFrames ||
    raw.some(r=>r[arm].length!==12||r[arm].some(v=>!Number.isFinite(v)))) return null;
  let scores=Array<number>(ids.length).fill(-Infinity),prior=scores.slice();
  const stay=Math.log(model.stay),step=Math.log1p(-model.stay);
  for (let t=0;t<raw.length;t++) {
    const x=raw[t][arm].map((v,d)=>(v-m.center[d])/m.scale[d]);
    const emissions=new Map<number,number>();
    for (const p of new Set(ids)) emissions.set(p,m.means[p].reduce((sum,mu,d)=> {
      const variance=m.variances[p][d]; return sum-.5*(Math.log(2*Math.PI*variance)+(x[d]-mu)**2/variance);
    },0));
    const next=scores.slice(),nextPrior=prior.slice();
    for (let s=0;s<ids.length;s++) {
      next[s]=(t===0?(s===0?0:-Infinity):regularityLogadd(scores[s]+stay,s?scores[s-1]+step:-Infinity))+emissions.get(ids[s])!;
      nextPrior[s]=t===0?(s===0?0:-Infinity):regularityLogadd(prior[s]+stay,s?prior[s-1]+step:-Infinity);
    }
    scores=next; prior=nextPrior;
  }
  const value=scores.at(-1)!-prior.at(-1)!;
  return Number.isFinite(value)?value:null;
}
export function regularityComparison(model: RegularityModel, window: RegularityObservation[], candidate: SequenceEvidenceCandidate): RegularityEvidence | null {
  const controlWritten=scoreRegularitySequence(model,window,candidate.written,"energy");
  const controlRival=scoreRegularitySequence(model,window,candidate.rival,"energy");
  const regularityWritten=scoreRegularitySequence(model,window,candidate.written,"regularity");
  const regularityRival=scoreRegularitySequence(model,window,candidate.rival,"regularity");
  if (controlWritten===null||controlRival===null||regularityWritten===null||regularityRival===null) return null;
  return {...candidate,controlWritten,controlRival,regularityWritten,regularityRival};
}
export const regularityReversal=(c: RegularityEvidence): boolean=>
  [c.controlWritten,c.controlRival,c.regularityWritten,c.regularityRival].every(Number.isFinite)&&
  c.controlWritten<c.controlRival&&c.regularityWritten>c.regularityRival;

export function validateRegularityModel(value: unknown,inventory: string,tokens: string): RegularityModel {
  const m=value as RegularityModel,n=tokens.length+1;
  const vector=(v: number[],len: number,positive=false)=>Array.isArray(v)&&v.length===len&&v.every(x=>Number.isFinite(x)&&(!positive||x>0));
  const emission=(e: EmissionModel)=>e&&vector(e.center,12)&&vector(e.scale,12,true)&&vector(e.counts,n)&&e.counts.every(c=>c>=0)&&
    Array.isArray(e.means)&&e.means.length===n&&e.means.every(r=>vector(r,12))&&
    Array.isArray(e.variances)&&e.variances.length===n&&e.variances.every(r=>vector(r,12,true));
  if (!m||m.schema!==1||m.inventory!==inventory||m.tokens!==tokens||m.frameMs!==40||m.observationMs!==10||m.dimensions!==12||
    !vector(m.scales,7,true)||m.scales.some((s,i)=>!Number.isInteger(s)||s>256||(i>0&&s<=m.scales[i-1]))||
    !Number.isFinite(m.stay)||m.stay<=0||m.stay>=1||!emission(m.regularity)||!emission(m.energy)||
    !Number.isInteger(m.maxFrames)||m.maxFrames<1||m.maxFrames>2048) throw new Error("Invalid multiscale regularity model or inventory");
  return m;
}

/** Replay only to discover the original engine comparisons. Primary rows and tokens
 * remain untouched; serialized certificates carry both complete sequences and endpoints.
 */
export function addRegularityEvidence(model: RegularityModel | undefined,audio: Float32Array,chunks: PhonemeEvidence[],
  index: ReferenceIndex,phrases?: PhraseAsset,options: CheckerOptions={},mode: "free"|"memorization"="free",
  inspect?: (comparison: RegularityEvidence|null,candidate: SequenceEvidenceCandidate)=>void): PhonemeEvidence[] {
  if (!model||!audio.length) return chunks;
  const cells=insertionCells(chunks);
  if (!cells.length) return chunks;
  if (cells[0].frameMs!==model.frameMs) throw new Error("Regularity scorer clock differs from primary evidence");
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c) {candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const features=regularityFeatures(audio,model.scales),certificates: RegularityEvidence[]=[];
  for (const c of candidates.values()) {
    const comparison=c.fromFrame>=0&&c.toFrame*4<=features.length?regularityComparison(model,features.slice(c.fromFrame*4,c.toFrame*4),c):null;
    inspect?.(comparison,c);
    if (comparison&&regularityReversal(comparison)) certificates.push(comparison);
  }
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,regularityVetoes:[...(p.regularityVetoes??[]),...certificates]})});
}
