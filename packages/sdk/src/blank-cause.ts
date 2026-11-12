import { createChecker, createTracker, PosteriorFrames, trackerInput } from "@tilawa/core";
import type { BlankCauseEvidence, CheckerOptions, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

/** Quiet/speech HMM fitted only on none-labelled training speakers. No phone-conditioned parameters. */
export interface BlankCauseModel {
  schema: 1;
  inventory: string;
  tokens: string;
  frameMs: 40;
  dimensions: 4;
  center: number[];
  scale: number[];
  means: number[][];
  variances: number[][];
  initial: number[];
  transition: number[][];
  maxFrames: number;
}

export function validateBlankCauseModel(value: unknown, inventory: string, tokens: string): BlankCauseModel {
  const m=value as BlankCauseModel;
  const vector=(v: number[], n: number, positive=false) => Array.isArray(v)&&v.length===n&&v.every(x=>Number.isFinite(x)&&(!positive||x>0));
  const distribution=(v: number[]) => vector(v,2,true)&&v.every(p=>p<1)&&Math.abs(v[0]+v[1]-1)<1e-8;
  if (!m || m.schema!==1 || m.inventory!==inventory || m.tokens!==tokens || m.frameMs!==40 || m.dimensions!==4 ||
    !vector(m.center,4)||!vector(m.scale,4,true)||
    !Array.isArray(m.means)||m.means.length!==2||!m.means.every(v=>vector(v,4))||
    !Array.isArray(m.variances)||m.variances.length!==2||!m.variances.every(v=>vector(v,4,true))||
    !distribution(m.initial)||!Array.isArray(m.transition)||m.transition.length!==2||!m.transition.every(distribution)||
    !(m.means[0][0]<m.means[1][0])||!Number.isInteger(m.maxFrames)||m.maxFrames<1||m.maxFrames>2048)
    throw new Error("Invalid blank-cause model, normalized dynamics, or inventory");
  return m;
}

/** Causal 10 ms observations pooled over the same nine-frame receptive field as CTC subsampling.
 * Column order: log energy, normalized spectral flux, periodicity; the posterior blank logit is added later. */
export class BlankObservations {
  readonly frames: number[][]=[];
  private readonly bank=new Fbank();
  private pcm=new Float32Array(0);
  private previous: number[] | undefined;
  private pending: number[][]=[];
  push(samples: Float32Array): void {
    const buf=new Float32Array(this.pcm.length+samples.length);
    buf.set(this.pcm);buf.set(samples,this.pcm.length);
    const mel=this.bank.push(samples);
    for (let f=0;f<mel.length;f++) {
      const off=f*160, wave=buf.subarray(off,off+400);
      let mean=0;for (const x of wave) mean+=x;mean/=400;
      let energy=0;for (const x of wave) energy+=(x-mean)**2;
      const power=Array.from(mel[f],x=>Math.exp(x)), total=power.reduce((a,b)=>a+b,0);
      const spectrum=power.map(x=>Math.sqrt(x/total));
      const flux=this.previous?spectrum.reduce((a,x,i)=>a+(x-this.previous![i])**2,0):0;
      this.previous=spectrum;
      let periodicity=0;
      // The pitch range and lag grid are fixed signal geometry, not selected on practice audio.
      for (let lag=40;lag<=320;lag+=4) {
        let dot=0,a=0,b=0;
        for (let i=lag;i<400;i++) {const x=wave[i]-mean,y=wave[i-lag]-mean;dot+=x*y;a+=x*x;b+=y*y;}
        periodicity=Math.max(periodicity,dot/Math.sqrt(Math.max(1e-20,a*b)));
      }
      this.pending.push([Math.log(Math.max(1e-12,energy/400)),flux,periodicity]);
      if (this.pending.length===9) {
        this.frames.push([0,1,2].map(d=>this.pending.reduce((s,r)=>s+r[d],0)/9));
        this.pending.splice(0,4);
      }
    }
    this.pcm=buf.slice(mel.length*160);
  }
}
export function blankFeatures(audio: Float32Array): number[][] {const o=new BlankObservations();o.push(audio);return o.frames;}

export function blankObservation(feature: number[], row: ArrayLike<number>): number[] {
  const p=Math.min(1-1e-6,Math.max(1e-6,Math.exp(row[0])));
  return [...feature,Math.log(p/(1-p))];
}
const add=(a: number,b: number): number => a===-Infinity?b:b===-Infinity?a:Math.max(a,b)+Math.log1p(Math.exp(-Math.abs(a-b)));

/** Joint CTC × acoustic-state forward sum. Quiet emits only blank; ongoing speech emits the unchanged
 * CTC distribution, including speech-internal blank. These emission distributions and the acoustic
 * transition rows are normalized. All pronunciations use the same parameters and must finish fully.
 * The control integrates the same emissions against the stationary prior independently each frame,
 * yielding one blank state and removing only the cause-duration dependence. */
export function scoreBlankSequence(model: BlankCauseModel, rows: readonly ArrayLike<number>[], features: readonly number[][],
  pronunciation: string, dynamic: boolean): number | null {
  if (rows.length!==features.length||!rows.length||rows.length>model.maxFrames) return null;
  const labels=[...pronunciation].map(p=>model.tokens.indexOf(p)+1);
  if (labels.some(id=>id<=0)) return null;
  const L=labels.length,S=2*L+1,label=(s:number)=>s%2?labels[(s-1)>>1]:0;
  const emissions=features.map((f,i)=> {
    const x=blankObservation(f,rows[i]);
    if (x.length!==4||x.some(v=>!Number.isFinite(v))) return null;
    return model.means.map((m,z)=>x.reduce((sum,v,d)=> {
      const y=(v-model.center[d])/model.scale[d],variance=model.variances[z][d];
      return sum-.5*(Math.log(2*Math.PI*variance)+(y-m[d])**2/variance);
    },0));
  });
  if (emissions.some(e=>e===null)) return null;
  // The stationary prior makes the collapsed control insensitive to where a crop starts.
  const a=model.transition[0][1],b=model.transition[1][0],prior=[b/(a+b),a/(a+b)];
  const Z=dynamic?2:1;
  let prev=new Float64Array(S*Z).fill(-Infinity),next=new Float64Array(S*Z);
  const emission=(f:number,s:number,z:number) => {
    const id=label(s),e=emissions[f]!;
    if (dynamic) return e[z]+(z===0?(id===0?0:-Infinity):rows[f][id]);
    return add(Math.log(prior[0])+e[0]+(id===0?0:-Infinity),Math.log(prior[1])+e[1]+rows[f][id]);
  };
  for (let s=0;s<Math.min(S,2);s++) for (let z=0;z<Z;z++)
    prev[s*Z+z]=(dynamic?Math.log(prior[z]):0)+emission(0,s,z);
  for (let f=1;f<rows.length;f++) {
    next.fill(-Infinity);
    for (let s=0;s<S;s++) for (let z=0;z<Z;z++) {
      const predecessors=[s];
      if (s>0) predecessors.push(s-1);
      if (s>1&&s%2===1&&label(s)!==label(s-2)) predecessors.push(s-2);
      let sum=-Infinity;
      for (const t of predecessors) for (let q=0;q<Z;q++)
        sum=add(sum,prev[t*Z+q]+(dynamic?Math.log(model.transition[q][z]):0));
      next[s*Z+z]=sum+emission(f,s,z);
    }
    [prev,next]=[next,prev];
  }
  let result=-Infinity;
  for (const s of L?[S-2,S-1]:[0]) for (let z=0;z<Z;z++) result=add(result,prev[s*Z+z]);
  return result;
}

export function blankComparison(model: BlankCauseModel, features: readonly number[][], frames: PosteriorFrames,
  c: SequenceEvidenceCandidate): BlankCauseEvidence | null {
  if (!Number.isInteger(c.fromFrame)||!Number.isInteger(c.toFrame)||c.fromFrame<frames.startFrame||
    c.toFrame>frames.endFrame||c.toFrame>features.length||c.toFrame<=c.fromFrame) return null;
  const rows=Array.from({length:c.toFrame-c.fromFrame},(_,i)=>frames.row(c.fromFrame+i)),x=features.slice(c.fromFrame,c.toFrame);
  const controlWritten=scoreBlankSequence(model,rows,x,c.written,false),controlRival=scoreBlankSequence(model,rows,x,c.rival,false),
    stateWritten=scoreBlankSequence(model,rows,x,c.written,true),stateRival=scoreBlankSequence(model,rows,x,c.rival,true);
  if ([controlWritten,controlRival,stateWritten,stateRival].some(v=>v===null)) return null;
  return {...c,controlWritten:controlWritten!,controlRival:controlRival!,stateWritten:stateWritten!,stateRival:stateRival!};
}
export const blankReversal=(c: BlankCauseEvidence):boolean =>
  [c.controlWritten,c.controlRival,c.stateWritten,c.stateRival].every(Number.isFinite)&&
  c.controlWritten<c.controlRival&&c.stateWritten>c.stateRival;

/** Attach exact-window certificates without modifying primary rows, tokens or ownership decisions. */
export function addBlankCauseEvidence(model: BlankCauseModel | undefined, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset, options: CheckerOptions={},mode:"free"|"memorization"="free",
  inspect?:(comparison:BlankCauseEvidence|null,candidate:SequenceEvidenceCandidate)=>void):PhonemeEvidence[] {
  if (!model||!audio.length) return chunks;
  const cells=insertionCells(chunks);if (!cells.length) return chunks;
  if (cells[0].inventory!==model.inventory||cells[0].frameMs!==40) throw new Error("Blank-cause evidence clock or inventory differs");
  const frames=new PosteriorFrames(index.tokens.size),candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000}),checker=createChecker(index,{...options,phrases,
    sequenceEvidence(c) {candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) {frames.append(e.posteriors!);checker.push(e,tracker.push(trackerInput(e)));}
  checker.end(tracker.end());
  const features=blankFeatures(audio),certificates:BlankCauseEvidence[]=[];
  for (const c of candidates.values()) {
    const comparison=blankComparison(model,features,frames,c);inspect?.(comparison,c);
    if (comparison&&blankReversal(comparison)) certificates.push(comparison);
  }
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,
    blankCauseVetoes:[...(p.blankCauseVetoes??[]),...certificates]})});
}
