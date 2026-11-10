import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, PhonationEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank, fft } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

/** Additive phone and voice-quality emissions; the latter are shared by every phone. */
export interface PhonationModel {
  schema: 1;
  inventory: string;
  tokens: string;
  frameMs: 40;
  observationMs: 10;
  dimensions: 23;
  center: number[];
  scale: number[];
  means: number[][];
  variances: number[][];
  offsets: number[][];
  qualityMeans: number[][];
  qualityVariances: number[][];
  counts: number[];
  stay: number[];
  initial: number[];
  transition: number[][];
  maxFrames: number;
}

/** Exact primary analysis geometry, with separate un-preemphasized voice-quality observations. */
export class PhonationObservations {
  readonly frames: number[][] = [];
  private readonly fbank = new Fbank();
  private pending = new Float32Array(0);
  push(audio: Float32Array): void {
    const mel = this.fbank.push(audio);
    const buf = new Float32Array(this.pending.length + audio.length);
    buf.set(this.pending); buf.set(audio, this.pending.length);
    let off = 0, t = 0;
    for (; off + 400 <= buf.length; off += 160) {
      const x = Float64Array.from(buf.subarray(off, off + 400));
      const mean = x.reduce((a,b) => a+b,0)/400;
      for (let i=0;i<400;i++) x[i]-=mean;
      const re=new Float64Array(512), im=new Float64Array(512);
      for (let i=0;i<400;i++) re[i]=x[i]*Math.pow(.5-.5*Math.cos(2*Math.PI*i/399),.85);
      fft(re,im);
      let sx=0,sy=0,sxx=0,sxy=0,n=0;
      for (let k=4;k<=128;k++) {
        const f=Math.log(k*16000/512), p=Math.log(Math.max(1e-12,re[k]**2+im[k]**2));
        sx+=f;sy+=p;sxx+=f*f;sxy+=f*p;n++;
      }
      const tilt=(n*sxy-sx*sy)/(n*sxx-sx*sx);
      const periodicity=(signal:Float64Array) => {
        let best=0;
        for (let lag=40;lag<=320;lag++) {
          let ab=0,aa=0,bb=0;
          for(let i=lag;i<signal.length;i++) { ab+=signal[i]*signal[i-lag];aa+=signal[i]**2;bb+=signal[i-lag]**2; }
          best=Math.max(best,ab/Math.sqrt(Math.max(1e-20,aa*bb)));
        }
        return Math.min(.999,Math.max(.001,best));
      };
      const r=periodicity(x), hnr=Math.log(r/(1-r));
      // Regularized Levinson recursion. Residual periodicity reads excitation,
      // whereas the first autocorrelation reads the original harmonic signal.
      const ac=Array.from({length:13},(_,lag)=>x.reduce((s,v,i)=>s+(i>=lag?v*x[i-lag]:0),0));
      ac[0]+=Math.max(1e-12,ac[0]*1e-4);
      let a:number[]=[1], error=ac[0];
      for(let order=1;order<=12;order++) {
        const reflection=Math.max(-.99,Math.min(.99,-(ac[order]+a.slice(1).reduce((s,v,j)=>s+v*ac[order-j-1],0))/error));
        const next=[...a,reflection];
        for(let j=1;j<order;j++) next[j]=a[j]+reflection*a[order-j];
        a=next;error=Math.max(1e-12,error*(1-reflection**2));
      }
      const residual=Float64Array.from(x.subarray(12),(_,i)=>a.reduce((s,v,j)=>s+v*x[i+12-j],0));
      const regularity=periodicity(residual);
      const row=mel[t++];
      this.frames.push([...Array.from({length:20},(_,d)=>(row[4*d]+row[4*d+1]+row[4*d+2]+row[4*d+3])/4),tilt,hnr,regularity]);
    }
    this.pending=buf.slice(off);
  }
}
export function phonationFeatures(audio:Float32Array):number[][] {
  const observations=new PhonationObservations();observations.push(audio);return observations.frames;
}
export const phonationLogadd=(a:number,b:number):number => {
  const m=Math.max(a,b);return Number.isFinite(m)?m+Math.log(Math.exp(a-m)+Math.exp(b-m)):-Infinity;
};

/** Product of a complete phone chain and a shared switching voice-quality chain.
 * Fixed control samples the same initial prior, then retains that state for the
 * entire window. Both arms condition the phone-duration prior on completion.
 * Edge background states absorb silence; no target phone is optional.
 */
export function scorePhonationSequence(model:PhonationModel, raw:number[][], pronunciation:string, switching:boolean):number|null {
  const phones=[...pronunciation].map(p=>model.tokens.indexOf(p)+1);
  if(!phones.length||phones.some(p=>p<1||model.counts[p]<=0)||raw.length<phones.length+2||raw.length>model.maxFrames||
    raw.some(r=>r.length!==23||r.some(v=>!Number.isFinite(v))))return null;
  const ids=[0,...phones,0], n=ids.length;
  let alpha=Array<number>(n*3).fill(-Infinity), prior=Array<number>(n).fill(-Infinity);
  for(let t=0;t<raw.length;t++) {
    const x=raw[t].map((v,d)=>(v-model.center[d])/model.scale[d]);
    const emission=new Map<number,number[]>();
    for(const p of new Set(ids)) emission.set(p,Array.from({length:3},(_,z)=> {
      let score=0;
      for(let d=0;d<23;d++) {
        const mu=d<20?model.means[p][d]+model.offsets[z][d]:model.qualityMeans[z][d-20];
        const v=d<20?model.variances[p][d]:model.qualityVariances[z][d-20];
        score-=.5*(Math.log(2*Math.PI*v)+(x[d]-mu)**2/v);
      }
      return score;
    }));
    const next=alpha.slice(), nextPrior=prior.slice();
    for(let s=0;s<n;s++) {
      const stay=Math.log(model.stay[ids[s]]), step=s?Math.log1p(-model.stay[ids[s-1]]):-Infinity;
      nextPrior[s]=t===0?(s===0?0:-Infinity):phonationLogadd(prior[s]+stay,s?prior[s-1]+step:-Infinity);
      for(let z=0;z<3;z++) {
        let value=-Infinity;
        if(t===0) value=s===0?Math.log(model.initial[z]):-Infinity;
        else for(let previous=0;previous<3;previous++) {
          const transition=switching?Math.log(model.transition[previous][z]):(previous===z?0:-Infinity);
          value=phonationLogadd(value,transition+phonationLogadd(alpha[s*3+previous]+stay,s?alpha[(s-1)*3+previous]+step:-Infinity));
        }
        next[s*3+z]=value+emission.get(ids[s])![z];
      }
    }
    alpha=next;prior=nextPrior;
  }
  const score=alpha.slice(-3).reduce(phonationLogadd,-Infinity)-prior[n-1];
  return Number.isFinite(score)?score:null;
}
export function phonationComparison(model:PhonationModel, window:number[][], c:SequenceEvidenceCandidate):PhonationEvidence|null {
  const fixedWritten=scorePhonationSequence(model,window,c.written,false), fixedRival=scorePhonationSequence(model,window,c.rival,false);
  const switchingWritten=scorePhonationSequence(model,window,c.written,true), switchingRival=scorePhonationSequence(model,window,c.rival,true);
  if(fixedWritten===null||fixedRival===null||switchingWritten===null||switchingRival===null)return null;
  return {...c,fixedWritten,fixedRival,switchingWritten,switchingRival};
}
export const phonationReversal=(c:PhonationEvidence):boolean =>
  [c.fixedWritten,c.fixedRival,c.switchingWritten,c.switchingRival].every(Number.isFinite)&&
  c.fixedWritten<c.fixedRival&&c.switchingWritten>c.switchingRival;

/** Token metadata survives ordinary JSON and native-frame rechunking without changing primary values. */
export function carryPhonationEvidence(chunks:PhonemeEvidence[],certificates:PhonationEvidence[]):PhonemeEvidence[] {
  if(!certificates.length)return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,phonationVetoes:[...(p.phonationVetoes??[]),...certificates]})});
}

export function validatePhonationModel(value:unknown, inventory:string, tokens:string):PhonationModel {
  const m=value as PhonationModel, n=tokens.length+1;
  const vector=(v:number[],len:number,positive=false)=>Array.isArray(v)&&v.length===len&&v.every(x=>Number.isFinite(x)&&(!positive||x>0));
  const matrix=(v:number[][],rows:number,cols:number,positive=false)=>Array.isArray(v)&&v.length===rows&&v.every(r=>vector(r,cols,positive));
  const probability=(v:number[])=>vector(v,3,true)&&Math.abs(v.reduce((s,x)=>s+x,0)-1)<1e-8;
  if(!m||m.schema!==1||m.inventory!==inventory||m.tokens!==tokens||m.frameMs!==40||m.observationMs!==10||m.dimensions!==23||
    !vector(m.center,23)||!vector(m.scale,23,true)||!matrix(m.means,n,20)||!matrix(m.variances,n,20,true)||
    !matrix(m.offsets,3,20)||!matrix(m.qualityMeans,3,3)||!matrix(m.qualityVariances,3,3,true)||
    !vector(m.counts,n)||m.counts.some(c=>c<0)||!vector(m.stay,n,true)||m.stay.some(p=>p>=1)||
    !probability(m.initial)||!Array.isArray(m.transition)||m.transition.length!==3||!m.transition.every(probability)||
    !Number.isInteger(m.maxFrames)||m.maxFrames<3||m.maxFrames>4096)throw new Error("Invalid phonation acoustic model or inventory");
  return m;
}

/** Discover baseline confirmations; only exact reversal certificates are added to primary evidence. */
export function addPhonationEvidence(model:PhonationModel|undefined,audio:Float32Array,chunks:PhonemeEvidence[],
  index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},mode:"free"|"memorization"="free",
  inspect?:(c:PhonationEvidence|null,candidate:SequenceEvidenceCandidate)=>void):PhonemeEvidence[] {
  if(!model||!audio.length)return chunks;
  const cells=insertionCells(chunks);if(!cells.length)return chunks;
  if(cells[0].frameMs!==model.frameMs)throw new Error("Phonation scorer clock differs from primary evidence");
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for(const e of cells)checker.push(e,tracker.push(trackerInput(e)));checker.end(tracker.end());
  const features=phonationFeatures(audio),certificates:PhonationEvidence[]=[];
  for(const c of candidates.values()) {
    const comparison=c.fromFrame>=0&&c.toFrame*4<=features.length?phonationComparison(model,features.slice(c.fromFrame*4,c.toFrame*4),c):null;
    inspect?.(comparison,c);if(comparison&&phonationReversal(comparison))certificates.push(comparison);
  }
  return carryPhonationEvidence(chunks,certificates);
}
