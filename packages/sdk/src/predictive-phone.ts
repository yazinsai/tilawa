import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, PredictivePhoneEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

/** Normal/inverse-Wishart posterior. Covariance is full, shared between all occurrences of a phone in a draw. */
export interface PhonePosterior { mean: number[]; kappa: number; nu: number; psi: number[][]; count: number }
export interface PredictiveModel {
  schema: 1; inventory: string; tokens: string; frameMs: 40; observationMs: 10; dimensions: 8;
  center: number[]; scale: number[]; phones: PhonePosterior[]; stay: number;
  draws: 128; maxFrames: 2048;
}
/** Frozen native frontend; ten adjacent mel bins form each of eight coordinates. */
export class PredictiveObservations {
  readonly frames: number[][] = [];
  private readonly fbank = new Fbank();
  push(audio: Float32Array): void {
    for (const row of this.fbank.push(audio)) this.frames.push(Array.from({length:8},(_,d)=> {
      let sum=0; for (let j=0;j<10;j++) sum+=row[10*d+j]; return sum/10;
    }));
  }
}
export function predictiveFeatures(audio: Float32Array): number[][] {
  const o=new PredictiveObservations(); o.push(audio); return o.frames;
}
export const logadd = (a: number,b: number): number => {
  const m=Math.max(a,b); return Number.isFinite(m) ? m+Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
export function cholesky(a: number[][]): number[][] {
  const n=a.length,l=Array.from({length:n},()=>Array(n).fill(0));
  for (let i=0;i<n;i++) for (let j=0;j<=i;j++) {
    let v=a[i][j]; for (let k=0;k<j;k++) v-=l[i][k]*l[j][k];
    if (i===j) { if (!(v>0)) throw new Error("Covariance is not positive definite"); l[i][j]=Math.sqrt(v); }
    else l[i][j]=v/l[j][j];
  }
  return l;
}
function inverseLower(l: number[][]): number[][] {
  const n=l.length,out=Array.from({length:n},()=>Array(n).fill(0));
  for (let j=0;j<n;j++) for (let i=j;i<n;i++) {
    let v=i===j?1:0; for (let k=j;k<i;k++) v-=l[i][k]*out[k][j]; out[i][j]=v/l[i][i];
  }
  return out;
}
/** Deterministic Monte Carlo, independent of words, windows, readers and candidate order. */
export class PosteriorRandom {
  constructor(private state=1729) {}
  uniform(): number {
    this.state=(Math.imul(1664525,this.state)+1013904223)>>>0; return (this.state+.5)/4294967296;
  }
  normal(): number { return Math.sqrt(-2*Math.log(this.uniform()))*Math.cos(2*Math.PI*this.uniform()); }
  gamma(shape: number): number {
    if (shape<1) return this.gamma(shape+1)*this.uniform()**(1/shape);
    const d=shape-1/3,c=1/Math.sqrt(9*d);
    for (;;) {
      const x=this.normal(), b=1+c*x; if (b<=0) continue;
      const v=b*b*b,u=this.uniform();
      if (u<1-.0331*x**4 || Math.log(u)<.5*x*x+d*(1-v+Math.log(v))) return d*v;
    }
  }
}
interface Emission { mean: number[]; precision: number[][]; logDet: number }
function emission(mean: number[],cov: number[][]): Emission {
  const l=cholesky(cov); return { mean,precision:inverseLower(l),logDet:2*l.reduce((s,r,i)=>s+Math.log(r[i]),0) };
}
function point(p: PhonePosterior): Emission {
  return emission(p.mean,p.psi.map(r=>r.map(v=>v/(p.nu-p.mean.length-1))));
}
/** Bartlett decomposition: sigma^-1 ~ Wishart(psi^-1,nu), mu|sigma ~ N(mean,sigma/kappa). */
export function samplePosterior(p: PhonePosterior,rng: PosteriorRandom): Emission {
  const n=p.mean.length, lp=cholesky(p.psi), inv=inverseLower(lp);
  const scale=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>inv.reduce((s,r)=>s+r[i]*r[j],0)));
  const l=cholesky(scale), a=Array.from({length:n},()=>Array(n).fill(0));
  for (let i=0;i<n;i++) for (let j=0;j<=i;j++) a[i][j]=i===j?Math.sqrt(2*rng.gamma((p.nu-i)/2)):rng.normal();
  const b=l.map((r,i)=>r.map((_,j)=>r.reduce((s,v,k)=>s+v*a[k][j],0))), bi=inverseLower(b);
  const cov=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>bi.reduce((s,r)=>s+r[i]*r[j],0)));
  const lc=cholesky(cov),z=Array.from({length:n},()=>rng.normal());
  const mean=p.mean.map((m,i)=>m+lc[i].reduce((s,v,j)=>s+v*z[j],0)/Math.sqrt(p.kappa));
  return emission(mean,cov);
}
function gaussian(x: number[],e: Emission): number {
  let q=0; for (const row of e.precision) { let v=0; for (let d=0;d<x.length;d++) v+=row[d]*(x[d]-e.mean[d]); q+=v*v; }
  return -.5*(x.length*Math.log(2*Math.PI)+e.logDet+q);
}
/** All ordered phone boundaries, with every complete phone required. Duration prior is conditioned on completion. */
export function boundaryLogP(values: number[][],stay: number): number {
  if (!values.length || !values[0].length || values.length<values[0].length) return -Infinity;
  const states=values[0].length, self=Math.log(stay),step=Math.log1p(-stay);
  let a=Array(states).fill(-Infinity),prior=a.slice(); a[0]=values[0][0]; prior[0]=0;
  for (let t=1;t<values.length;t++) {
    a=a.map((v,s)=>logadd(v+self,s?a[s-1]+step:-Infinity)+values[t][s]);
    prior=prior.map((v,s)=>logadd(v+self,s?prior[s-1]+step:-Infinity));
  }
  return a[states-1]-prior[states-1];
}
// Lanczos log-gamma, used by the exact conjugate predictive for a single emission class.
function logGamma(z: number): number {
  const c=[676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,
    12.507343278686905,-.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];
  let x=.99999999999980993; z-=1;
  for (let i=0;i<c.length;i++) x+=c[i]/(z+i+1);
  const t=z+7.5; return .5*Math.log(2*Math.PI)+(z+.5)*Math.log(t)-t+Math.log(x);
}
/** Exact joint NIW predictive, sharing the emission parameters over every observed frame. */
export function integratedEmission(p: PhonePosterior,x: number[][]): number {
  const n=x.length,d=p.mean.length,k=p.kappa+n,nu=p.nu+n;
  const mean=Array.from({length:d},(_,j)=>x.reduce((s,r)=>s+r[j],0)/n);
  const psi=p.psi.map((r,i)=>r.map((v,j)=>v+x.reduce((s,row)=>s+(row[i]-mean[i])*(row[j]-mean[j]),0)+
    p.kappa*n/k*(mean[i]-p.mean[i])*(mean[j]-p.mean[j])));
  const determinant=(a:number[][])=>2*cholesky(a).reduce((s,r,i)=>s+Math.log(r[i]),0);
  let gamma=0; for (let j=0;j<d;j++) gamma+=logGamma((nu-j)/2)-logGamma((p.nu-j)/2);
  return -n*d/2*Math.log(Math.PI)+d/2*Math.log(p.kappa/k)+p.nu/2*determinant(p.psi)-nu/2*determinant(psi)+gamma;
}
function validWindow(model: PredictiveModel,raw: number[][],sequences: string[]): number[][] | null {
  if (!raw.length || raw.length>model.maxFrames || raw.some(r=>r.length!==8||r.some(v=>!Number.isFinite(v)))) return null;
  const ids=sequences.map(s=>[...s].map(p=>model.tokens.indexOf(p)+1));
  if (ids.some(s=>!s.length || s.length>raw.length || s.some(id=>id<1 || model.phones[id].count<=0))) return null;
  return ids;
}
/** Integrate the likelihood, never average log likelihoods. One draw supplies every frame and repeated phone. */
export function predictiveScores(model: PredictiveModel,raw: number[][],sequences: string[]): {point:number[]; half:number[]; predictive:number[]} | null {
  const ids=validWindow(model,raw,sequences); if (!ids) return null;
  const x=raw.map(r=>r.map((v,d)=>(v-model.center[d])/model.scale[d]));
  const used=[...new Set(ids.flat())].sort((a,b)=>a-b);
  const score=(table: Map<number,Emission>)=> {
    const rows=new Map(used.map(p=>[p,x.map(r=>gaussian(r,table.get(p)!))]));
    return ids.map(s=>boundaryLogP(x.map((_,t)=>s.map(p=>rows.get(p)![t])),model.stay));
  };
  const control=score(new Map(used.map(p=>[p,point(model.phones[p])])));
  if (ids.every(s=>new Set(s).size===1)) {
    const predictive=ids.map(s=>integratedEmission(model.phones[s[0]],x));
    return [...control,...predictive].every(Number.isFinite) ? {point:control,half:predictive.slice(),predictive} : null;
  }
  let sums=sequences.map(()=>-Infinity),half=sums.slice();
  // Each phone has its own fixed random stream; adding or exchanging a candidate changes no shared draw.
  const random=new Map(used.map(p=>[p,new PosteriorRandom(1729+p)]));
  for (let draw=0;draw<model.draws;draw++) {
    const values=score(new Map(used.map(p=>[p,samplePosterior(model.phones[p],random.get(p)!)])));
    sums=sums.map((v,i)=>logadd(v,values[i]));
    if (draw+1===model.draws/2) half=sums.map(v=>v-Math.log(model.draws/2));
  }
  const predictive=sums.map(v=>v-Math.log(model.draws));
  // When every phone state uses the same class, boundaries cancel and conjugate integration is exact.
  for (let i=0;i<ids.length;i++) if (new Set(ids[i]).size===1) {
    predictive[i]=half[i]=integratedEmission(model.phones[ids[i][0]],x);
  }
  return [...control,...half,...predictive].every(Number.isFinite) ? {point:control,half,predictive} : null;
}
export function predictiveComparison(model: PredictiveModel,raw: number[][],c: SequenceEvidenceCandidate): PredictivePhoneEvidence | null {
  const scores=predictiveScores(model,raw,[c.written,c.rival]); if (!scores) return null;
  return {...c,pointWritten:scores.point[0],pointRival:scores.point[1],
    predictiveWritten:scores.predictive[0],predictiveRival:scores.predictive[1],
    halfWritten:scores.half[0],halfRival:scores.half[1]};
}
export const predictiveReversal=(c: PredictivePhoneEvidence): boolean=>
  [c.pointWritten,c.pointRival,c.predictiveWritten,c.predictiveRival,c.halfWritten,c.halfRival].every(Number.isFinite) &&
  c.pointWritten<c.pointRival && c.predictiveWritten>c.predictiveRival && c.halfWritten>c.halfRival;
export function validatePredictiveModel(value: unknown,inventory: string,tokens: string): PredictiveModel {
  const m=value as PredictiveModel;
  const vector=(v: number[],positive=false)=>Array.isArray(v)&&v.length===8&&v.every(x=>Number.isFinite(x)&&(!positive||x>0));
  if (!m || m.schema!==1 || m.inventory!==inventory || m.tokens!==tokens || m.frameMs!==40 || m.observationMs!==10 || m.dimensions!==8 ||
    !vector(m.center)||!vector(m.scale,true)||!Array.isArray(m.phones)||m.phones.length!==tokens.length+1||
    !(m.stay>0&&m.stay<1)||m.draws!==128||m.maxFrames!==2048) throw new Error("Invalid predictive phone model");
  for (const p of m.phones) {
    if (!vector(p.mean)||!(p.kappa>0)||!Number.isFinite(p.kappa)||!(p.nu>9)||!Number.isFinite(p.nu)||
      !Number.isFinite(p.count)||p.count<0||!Array.isArray(p.psi)||p.psi.length!==8||!p.psi.every(r=>vector(r))||
      p.psi.some((r,i)=>r.some((v,j)=>Math.abs(v-p.psi[j][i])>1e-8))) throw new Error("Invalid phone posterior");
    cholesky(p.psi);
  }
  return m;
}
/** Generic engine discovery; labels and incident keys cannot choose windows or candidates. */
export function addPredictiveEvidence(model: PredictiveModel | undefined,audio: Float32Array,chunks: PhonemeEvidence[],
  index: ReferenceIndex,phrases?: PhraseAsset,options: CheckerOptions={},mode: "free"|"memorization"="free",
  inspect?: (c: PredictivePhoneEvidence|null,candidate: SequenceEvidenceCandidate)=>void): PhonemeEvidence[] {
  if (!model||!audio.length) return chunks;
  const cells=insertionCells(chunks); if (!cells.length) return chunks;
  if (cells[0].frameMs!==model.frameMs) throw new Error("Predictive scorer clock differs from primary evidence");
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e))); checker.end(tracker.end());
  const features=predictiveFeatures(audio),certificates: PredictivePhoneEvidence[]=[];
  for (const c of candidates.values()) {
    const result=c.fromFrame>=0&&c.toFrame*4<=features.length?predictiveComparison(model,features.slice(c.fromFrame*4,c.toFrame*4),c):null;
    inspect?.(result,c); if (result&&predictiveReversal(result)) certificates.push(result);
  }
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,predictivePhoneVetoes:[...(p.predictivePhoneVetoes??[]),...certificates]})});
}
