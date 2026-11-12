import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, SpectralCurvatureEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface CurvatureEmission { mean: number[]; variance: number[] }
export interface CurvatureArm { center: number[]; scale: number[]; emissions: CurvatureEmission[] }
export interface CurvatureModel {
  schema: 1;
  inventory: string;
  tokens: string;
  frameMs: 40;
  resolution: number;
  phoneCounts: number[];
  stay: number;
  maxFrames: number;
  curvature: CurvatureArm;
  control: CurvatureArm;
}

/** Equal-frequency samples of a triangularly smoothed power envelope. The
 * primary Kaldi frontend is observed without changing its outputs. Four native
 * 10 ms windows form a 40 ms observation, anchored at the original sample zero.
 */
export class CurvatureObservations {
  readonly frames: number[][] = [];
  private pending: number[][] = [];
  private readonly frontend: Fbank;
  constructor(resolution: number) {
    if (![16,24,32].includes(resolution)) throw new Error("Unsupported envelope resolution");
    const step=(7000-125)/(resolution-1);
    const weights=Array.from({length:resolution},(_,j)=> {
      const center=125+j*step;
      return Array.from({length:256},(_,k)=>Math.max(0,1-Math.abs(k*16000/512-center)/step));
    });
    this.frontend=new Fbank({onSpectrum:power=> {
      const envelope=weights.map(w=> {
        let sum=0, total=0;
        for(let k=0;k<w.length;k++) {sum+=w[k]*power[k];total+=w[k];}
        return Math.log(Math.max(sum/total,1e-12));
      });
      this.pending.push(envelope);
      if(this.pending.length===4) {
        this.frames.push(envelope.map((_,j)=>this.pending.reduce((s,r)=>s+r[j],0)/4));
        this.pending=[];
      }
    }});
  }
  push(audio: Float32Array): void {this.frontend.push(audio);}
}
export function curvatureFeatures(audio: Float32Array,resolution: number): number[][] {
  const o=new CurvatureObservations(resolution);o.push(audio);return o.frames;
}
/** Control and curvature have exactly the same number of observations and
 * parameters. Differencing annihilates a constant and a linear spectral tilt. */
export function curvatureView(envelope: number[],curved: boolean): number[] {
  return envelope.slice(1,-1).map((v,j)=>curved ? envelope[j]-2*v+envelope[j+2] : v);
}
export function curvatureLogadd(a: number,b: number): number {
  const m=Math.max(a,b);return Number.isFinite(m)?m+Math.log(Math.exp(a-m)+Math.exp(b-m)):-Infinity;
}

/** Ordered complete phones with optional silence states, ordinary repeated-
 * label separation, and shared geometric duration priors. Subtract the same
 * identity-neutral timing partition to condition on the observed length. */
export function curvatureChain(emissions: number[][],labels: number[],stay: number): number|null {
  if(!emissions.length || !labels.length) return null;
  const states=[0];for(const p of labels) states.push(p,0);
  const n=states.length;
  const skip=(s:number)=>s>=2 && states[s]!==0 && states[s]!==states[s-2];
  const outgoing=states.map((_,s)=>1+(s+1<n?1:0)+(s+2<n&&skip(s+2)?1:0));
  const self=outgoing.map(k=>k===1?0:Math.log(stay));
  const advance=outgoing.map(k=>k>1?Math.log((1-stay)/(k-1)):-Infinity);
  let a=Array<number>(n).fill(-Infinity),prior=a.slice();
  a[0]=emissions[0][0]-Math.log(2);a[1]=emissions[0][labels[0]]-Math.log(2);
  prior[0]=prior[1]=-Math.log(2);
  for(let t=1;t<emissions.length;t++) {
    const next=Array<number>(n).fill(-Infinity),np=next.slice();
    for(let s=0;s<n;s++) {
      let v=a[s]+self[s],p=prior[s]+self[s];
      if(s>0) {v=curvatureLogadd(v,a[s-1]+advance[s-1]);p=curvatureLogadd(p,prior[s-1]+advance[s-1]);}
      if(skip(s)) {v=curvatureLogadd(v,a[s-2]+advance[s-2]);p=curvatureLogadd(p,prior[s-2]+advance[s-2]);}
      next[s]=v+emissions[t][states[s]];np[s]=p;
    }
    a=next;prior=np;
  }
  const value=curvatureLogadd(a[n-1],a[n-2])-curvatureLogadd(prior[n-1],prior[n-2]);
  return Number.isFinite(value)?value:null;
}
export function scoreCurvatureSequence(m: CurvatureModel,raw: number[][],pronunciation: string,curved: boolean): number|null {
  const labels=[...pronunciation].map(p=>m.tokens.indexOf(p)+1);
  if(!labels.length || raw.length>m.maxFrames || labels.some(p=>p<=0||m.phoneCounts[p-1]<=0) ||
    raw.some(r=>r.length!==m.resolution||r.some(v=>!Number.isFinite(v)))) return null;
  const arm=curved?m.curvature:m.control;
  const rows=raw.map(r=> {
    const x=curvatureView(r,curved).map((v,d)=>(v-arm.center[d])/arm.scale[d]);
    return arm.emissions.map(e=>x.reduce((sum,v,d)=>sum-.5*(Math.log(2*Math.PI*e.variance[d])+(v-e.mean[d])**2/e.variance[d]),0));
  });
  return curvatureChain(rows,labels,m.stay);
}
export function curvatureComparison(m: CurvatureModel,raw: number[][],c: SequenceEvidenceCandidate): SpectralCurvatureEvidence|null {
  const controlWritten=scoreCurvatureSequence(m,raw,c.written,false),controlRival=scoreCurvatureSequence(m,raw,c.rival,false);
  const curvatureWritten=scoreCurvatureSequence(m,raw,c.written,true),curvatureRival=scoreCurvatureSequence(m,raw,c.rival,true);
  if(controlWritten===null||controlRival===null||curvatureWritten===null||curvatureRival===null) return null;
  return {...c,controlWritten,controlRival,curvatureWritten,curvatureRival};
}
export const curvatureReversal=(c: SpectralCurvatureEvidence):boolean=>
  [c.controlWritten,c.controlRival,c.curvatureWritten,c.curvatureRival].every(Number.isFinite)&&
  c.controlWritten<c.controlRival&&c.curvatureWritten>c.curvatureRival;

export function validateCurvatureModel(value:unknown,inventory:string,tokens:string):CurvatureModel {
  const m=value as CurvatureModel;
  const vector=(v:number[],n:number,positive=false)=>Array.isArray(v)&&v.length===n&&v.every(x=>Number.isFinite(x)&&(!positive||x>0));
  if(!m||m.schema!==1||m.inventory!==inventory||m.tokens!==tokens||m.frameMs!==40||![16,24,32].includes(m.resolution)||
    !vector(m.phoneCounts,tokens.length)||m.phoneCounts.some(c=>c<0)||!(m.stay>0&&m.stay<1)||
    !Number.isInteger(m.maxFrames)||m.maxFrames<2||m.maxFrames>2048) throw new Error("Invalid spectral curvature model");
  for(const arm of [m.control,m.curvature]) {
    const d=m.resolution-2;
    if(!arm||!vector(arm.center,d)||!vector(arm.scale,d,true)||!Array.isArray(arm.emissions)||arm.emissions.length!==tokens.length+1||
      arm.emissions.some(e=>!e||!vector(e.mean,d)||!vector(e.variance,d,true))) throw new Error("Invalid spectral curvature emissions");
  }
  return m;
}

/** Discover actual confirming comparisons; labels never participate. Certificates
 * are tied to complete pronunciations and exact absolute engine-requested frames. */
export function addCurvatureEvidence(model:CurvatureModel|undefined,audio:Float32Array,chunks:PhonemeEvidence[],
  index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},mode:"free"|"memorization"="free",
  inspect?:(c:SpectralCurvatureEvidence|null,candidate:SequenceEvidenceCandidate)=>void):PhonemeEvidence[] {
  if(!model||!audio.length) return chunks;
  const cells=insertionCells(chunks);if(!cells.length) return chunks;
  if(cells[0].frameMs!==model.frameMs) throw new Error("Curvature scorer clock differs from primary evidence");
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for(const e of cells) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const features=curvatureFeatures(audio,model.resolution),certificates:SpectralCurvatureEvidence[]=[];
  for(const c of candidates.values()) {
    const comparison=c.fromFrame>=0&&c.toFrame<=features.length?curvatureComparison(model,features.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(comparison,c);if(comparison&&curvatureReversal(comparison)) certificates.push(comparison);
  }
  if(!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,curvatureVetoes:[...(p.curvatureVetoes??[]),...certificates]})});
}
