import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CycleEvidence, CheckerOptions, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { insertionCells } from "./seam-insertion.js";
import { evidenceGrid } from "./evidence-grid.js";

const RATE = 16000, WINDOW = 800, HOP = 160, BINS = 8, PHASE_SAMPLES = 64;
export interface CycleModel {
  schema: 1; inventory: string; tokens: string; frameMs: 40; observationMs: 10;
  center: number[]; scale: number[]; means: number[][]; variances: number[][];
  counts: number[]; cycleCounts: number[]; stay: number; maxFrames: number;
}

/** Fixed high-band filters; their steady-state delay is shared by training and inference. */
function bandpass(frequency: number, width: number) {
  const w = 2*Math.PI*frequency/RATE, alpha = Math.sin(w)/(2*frequency/width);
  const b = alpha/(1+alpha), a1 = -2*Math.cos(w)/(1+alpha), a2 = (1-alpha)/(1+alpha);
  let x1=0, x2=0, y1=0, y2=0;
  return (x:number) => {
    const y=b*(x-x2)-a1*y1-a2*y2;
    x2=x1; x1=x; y2=y1; y1=y; return y;
  };
}

/** Closure proxies from the low-band waveform, never from text or network tokens.
 * Periods must agree across at least three complete cycles. A low correlation or
 * irregular closure sequence leaves phase unobserved, rather than inventing pitch.
 */
function closures(low: number[]): number[] | null {
  const down=low.filter((_,i)=>i%4===0), mean=down.reduce((a,b)=>a+b,0)/down.length;
  const x=down.map(v=>v-mean), correlations:number[]=[];
  for (let lag=10;lag<=67;lag++) {
    let xy=0, xx=0, yy=0;
    for(let i=lag;i<x.length;i++) { xy+=x[i]*x[i-lag];xx+=x[i]*x[i];yy+=x[i-lag]*x[i-lag]; }
    correlations[lag]=xx*yy>1e-24 ? xy/Math.sqrt(xx*yy) : 0;
  }
  const best=Math.max(...correlations.slice(10));
  if(best<.6) return null;
  let lag=10;
  for(;lag<=67;lag++) if(correlations[lag]>=best*.9 &&
    (lag===10||correlations[lag]>=correlations[lag-1]) && (lag===67||correlations[lag]>=correlations[lag+1])) break;
  if(lag>67) return null;
  const period=lag*4, derivative=low.map((v,i)=>i ? v-low[i-1] : 0);
  const minimum=(a:number,b:number) => {
    let at=Math.max(1,Math.ceil(a));
    for(let i=at+1;i<=Math.min(low.length-1,Math.floor(b));i++) if(derivative[i]<derivative[at])at=i;
    // Subsample localization prevents integer closure jitter from turning a
    // deterministic high harmonic into an artificial residual across cycles.
    if(at>0 && at+1<derivative.length) {
      const curve=derivative[at-1]-2*derivative[at]+derivative[at+1];
      if(curve>0)return at+Math.max(-.5,Math.min(.5,(derivative[at-1]-derivative[at+1])/(2*curve)));
    }
    return at;
  };
  const points=[minimum(1,period)];
  while(points.at(-1)!+period*1.25<low.length) {
    const previous=points.at(-1)!, next=minimum(previous+period*.75,previous+period*1.25);
    if(Math.abs((next-previous)/period-1)>.15) return null;
    points.push(next);
  }
  return points.length>=4 ? points : null;
}

/** Aperiodic waveform energy is measured after subtracting a phase-locked mean
 * cycle, so deterministic high harmonics do not masquerade as turbulence.
 * Helmert contrasts of log phase-bin energy retain the phase distribution while
 * removing its average. The first two observations are the matched pooled view.
 */
export function cycleWindow(low:number[], bands:number[][], power:number):number[] {
  const pooled=bands.map(x=>Math.log((x.reduce((v,y)=>v+y*y,0)/WINDOW+1e-30)/(power+1e-30)));
  const points=closures(low), phase=Array(14).fill(0);
  if(!points || power<1e-20) return [...pooled,...phase,0];
  const cyclePooled=pooled.slice();
  for(let b=0;b<2;b++) {
    const cycles:number[][]=[];
    for(let k=1;k<points.length;k++) {
      const from=points[k-1], length=points[k]-from;
      cycles.push(Array.from({length:PHASE_SAMPLES},(_,i)=>{
        const at=from+i*length/PHASE_SAMPLES, j=Math.floor(at), f=at-j;
        return bands[b][j]*(1-f)+bands[b][j+1]*f;
      }));
    }
    const template=Array.from({length:PHASE_SAMPLES},(_,i)=>cycles.reduce((v,c)=>v+c[i],0)/cycles.length);
    const energy=Array(BINS).fill(0);
    for(const c of cycles) for(let i=0;i<PHASE_SAMPLES;i++) energy[Math.floor(i*BINS/PHASE_SAMPLES)]+=(c[i]-template[i])**2;
    const total=energy.reduce((a,v)=>a+v,0);
    if(total<=power*1e-8) return [...pooled,...Array(14).fill(0),0];
    cyclePooled[b]=Math.log((total/(cycles.length*PHASE_SAMPLES)+1e-30)/(power+1e-30));
    const logs=energy.map(v=>Math.log((v+total*.001)/(total*(1+BINS*.001))));
    for(let k=1;k<BINS;k++) phase[b*7+k-1]=(logs.slice(0,k).reduce((a,v)=>a+v,0)-k*logs[k])/Math.sqrt(k*(k+1));
  }
  return [...cyclePooled,...phase,1];
}

/** 50 ms observations on the native 10 ms front-end grid. Filters and windows
 * retain state across pushes; no sample-grid shift is introduced by caller chunks. */
export class CycleObservations {
  readonly frames:number[][]=[];
  private low1=0;
  private low2=0;
  private ring=Array.from({length:4},()=>new Float64Array(WINDOW));
  private filters=[bandpass(3300,1800),bandpass(5500,2200)];
  private samples=0;
  push(audio:Float32Array):void {
    const decay=Math.exp(-2*Math.PI*800/RATE);
    for(const sample of audio) {
      this.low1=decay*this.low1+(1-decay)*sample;
      this.low2=decay*this.low2+(1-decay)*this.low1;
      const slot=this.samples%WINDOW;
      this.ring[0][slot]=sample;this.ring[1][slot]=this.low2;
      for(let b=0;b<2;b++)this.ring[b+2][slot]=this.filters[b](sample);
      this.samples++;
      if(this.samples<WINDOW || (this.samples-WINDOW)%HOP)continue;
      const ordered=this.ring.map(r=>Array.from({length:WINDOW},(_,i)=>r[(this.samples+i)%WINDOW]));
      const power=ordered[0].reduce((v,x)=>v+x*x,0)/WINDOW;
      this.frames.push(cycleWindow(ordered[1],ordered.slice(2),power));
    }
  }
}
export function cycleFeatures(audio:Float32Array):number[][] {
  const o=new CycleObservations();o.push(audio);return o.frames;
}
export const logadd=(a:number,b:number):number=> {
  const m=Math.max(a,b);return Number.isFinite(m) ? m+Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};

/** Identical pooled Gaussian emissions in both arms. Unreliable phase is
 * integrated out exactly. Subtract a shared standardized background so the
 * conditioned and pooled views use the same observation measure. */
export function cycleEmission(m:CycleModel,raw:number[],p:number,conditioned:boolean):number {
  let result=0;
  for(let d=0;d<16;d++) {
    if(d>=2 && (!conditioned || raw[16]===0))continue;
    const x=(raw[d]-m.center[d])/m.scale[d], delta=x-m.means[p][d], v=m.variances[p][d];
    result+=-.5*(Math.log(2*Math.PI*v)+delta*delta/v);
    if(d>=2)result+=.5*(Math.log(2*Math.PI)+x*x);
  }
  return result;
}

/** Normalized complete-phone chain: all phone boundaries are marginalized,
 * every phone must emit, and both arms use the same timing prior. */
export function scoreCycleSequence(m:CycleModel,raw:number[][],pronunciation:string,conditioned:boolean):number|null {
  const ids=[...pronunciation].map(p=>m.tokens.indexOf(p)+1);
  if(!ids.length || ids.some(p=>p<1 || m.counts[p]<=0 || (conditioned && m.cycleCounts[p]<=0)) ||
    raw.length<ids.length || raw.length>m.maxFrames || !raw.some(r=>r[16]===1) ||
    raw.some(r=>r.length!==17 || r.some(v=>!Number.isFinite(v)) || (r[16]!==0 && r[16]!==1)))return null;
  let scores=ids.map(()=>-Infinity),prior=scores.slice();
  for(let t=0;t<raw.length;t++) {
    const next=scores.slice(),np=prior.slice(),emissions=new Map<number,number>();
    for(const p of new Set(ids))emissions.set(p,cycleEmission(m,raw[t],p,conditioned));
    for(let s=0;s<ids.length;s++) {
      const a=t===0 ? (s===0 ? 0 : -Infinity) : logadd(scores[s]+Math.log(m.stay),s?scores[s-1]+Math.log1p(-m.stay):-Infinity);
      np[s]=t===0 ? (s===0 ? 0 : -Infinity) : logadd(prior[s]+Math.log(m.stay),s?prior[s-1]+Math.log1p(-m.stay):-Infinity);
      next[s]=a+emissions.get(ids[s])!;
    }
    scores=next;prior=np;
  }
  const value=scores.at(-1)!-prior.at(-1)!;
  return Number.isFinite(value) ? value : null;
}
export function cycleComparison(m:CycleModel,window:number[][],c:SequenceEvidenceCandidate):CycleEvidence|null {
  const controlWritten=scoreCycleSequence(m,window,c.written,false),controlRival=scoreCycleSequence(m,window,c.rival,false);
  const conditionedWritten=scoreCycleSequence(m,window,c.written,true),conditionedRival=scoreCycleSequence(m,window,c.rival,true);
  if(controlWritten===null || controlRival===null || conditionedWritten===null || conditionedRival===null)return null;
  return {...c,controlWritten,controlRival,conditionedWritten,conditionedRival};
}
export const cycleReversal=(c:CycleEvidence):boolean=>
  [c.controlWritten,c.controlRival,c.conditionedWritten,c.conditionedRival].every(Number.isFinite) &&
  c.controlWritten<c.controlRival && c.conditionedWritten>c.conditionedRival;

export function validateCycleModel(value:unknown,inventory:string,tokens:string):CycleModel {
  const m=value as CycleModel,n=tokens.length+1;
  const vector=(v:number[],len:number,pos=false)=>Array.isArray(v)&&v.length===len&&v.every(x=>Number.isFinite(x)&&(!pos||x>0));
  const rows=(v:number[][])=>Array.isArray(v)&&v.length===n&&v.every(r=>vector(r,16));
  if(!m || m.schema!==1 || m.inventory!==inventory || m.tokens!==tokens || m.frameMs!==40 || m.observationMs!==10 ||
    !vector(m.center,16) || !vector(m.scale,16,true) || !rows(m.means) || !rows(m.variances) || m.variances.some(r=>r.some(v=>v<=0)) ||
    !vector(m.counts,n) || !vector(m.cycleCounts,n) || m.counts.some(v=>v<0) || m.cycleCounts.some((v,p)=>v<0||v>m.counts[p]+1e-6) ||
    !Number.isFinite(m.stay)||m.stay<=0||m.stay>=1 || !Number.isInteger(m.maxFrames)||m.maxFrames<1||m.maxFrames>4096)
    throw new Error("Invalid cycle turbulence model or inventory");
  return m;
}

/** Baseline discovery uses the actual complete hypotheses and exact requested
 * frames. Metadata alone changes; no primary posterior or tracker token is replaced. */
export function addCycleEvidence(m:CycleModel|undefined,audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,
  phrases?:PhraseAsset,options:CheckerOptions={},mode:"free"|"memorization"="free",
  inspect?:(c:CycleEvidence|null,candidate:SequenceEvidenceCandidate)=>void,evidenceMs:number|undefined=m?.frameMs):PhonemeEvidence[] {
  if(!m || !audio.length)return chunks;
  const cells=insertionCells(chunks);if(!cells.length)return chunks;
  if(cells[0].frameMs!==m.frameMs)throw new Error("Cycle scorer clock differs from primary evidence");
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const discover=(stream:PhonemeEvidence[],acquireMs?:number)=> {
    const tracker=createTracker(index,{mode,phrases,...(acquireMs&&{acquireMs})});
    const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
    for(const e of stream)checker.push(e,tracker.push(trackerInput(e)));checker.end(tracker.end());
  };
  // Direct core consumers use complete token runs on the native frame grid.
  // SDK sessions use their own fixed cells and acquisition cadence. Completion
  // edges can yield different hypotheses; certificates match each exact request.
  discover(cells);
  const grid=evidenceGrid(evidenceMs??m.frameMs),sdkCells:PhonemeEvidence[]=[];
  for(const e of chunks) {
    grid.add(e);
    sdkCells.push(...grid.ready(e.posteriors&&e.posteriors.startFrame+e.posteriors.nFrames));
  }
  sdkCells.push(...grid.rest());discover(sdkCells,1000);
  const features=cycleFeatures(audio),certificates:CycleEvidence[]=[];
  for(const c of candidates.values()) {
    const comparison=c.fromFrame>=0&&c.toFrame*4<=features.length ? cycleComparison(m,features.slice(c.fromFrame*4,c.toFrame*4),c) : null;
    inspect?.(comparison,c);if(comparison&&cycleReversal(comparison))certificates.push(comparison);
  }
  if(!certificates.length)return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,cycleVetoes:[...(p.cycleVetoes??[]),...certificates]})});
}
