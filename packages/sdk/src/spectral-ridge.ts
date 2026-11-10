import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, SpectralRidgeEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface RidgeModel {
  schema: 1; inventory: string; tokens: string; frameMs: 40; observationMs: 10;
  dimensions: 56; controlDimensions: 32; radius: 1 | 3;
  center: number[]; scale: number[]; means: number[][]; variances: number[][];
  counts: number[]; stay: number; maxFrames: number;
}
interface Interval { lo: number; hi: number; peak: number }
const LEVELS = [-2, -1, 0, 1, 2];
const intervalCenter = (r: Interval) => (r.lo+r.hi)/2;
const links = (a: Interval, b: Interval) => a.lo <= b.hi+1 && b.lo <= a.hi+1;
function intervals(row: number[], level: number): Interval[] {
  const result: Interval[]=[];
  for (let k=0;k<row.length;k++) if (row[k]>level) {
    const lo=k; let peak=row[k];
    while (k+1<row.length && row[k+1]>level) peak=Math.max(peak,row[++k]);
    result.push({lo,hi:k,peak});
  }
  return result;
}

/** Integrate superlevel ridge components, including their split / merge edges.
 * No energy level decides whether a ridge exists. Five quadrature levels approximate
 * its persistence; the same nodes supply location and energy statistics in the control.
 */
export function ridgePatch(spectra: number[][], at: number, radius: number): number[] {
  const out=Array<number>(56).fill(0);
  const normalized=spectra.map(row=> {
    const mean=row.reduce((s,v)=>s+v,0)/64;
    const scale=Math.max(.1,Math.sqrt(row.reduce((s,v)=>s+(v-mean)**2,0)/64));
    return row.map(v=>(v-mean)/scale);
  });
  const row=normalized[at];
  for (let b=0;b<8;b++) out[b*4]=spectra[at].slice(b*8,b*8+8).reduce((s,v)=>s+v,0)/8;
  for (const level of LEVELS) {
    const graph=normalized.map(r=>intervals(r,level));
    for (const node of graph[at]) {
      const b=Math.min(7,Math.floor(intervalCenter(node)/8)), weight=1/LEVELS.length;
      out[b*4+1]+=weight; // ridge locations and energy remain available to both arms
      out[b*4+2]+=weight*(intervalCenter(node)%8)/8;
      out[b*4+3]+=weight*(node.peak-level);
      let extent=0, branches=0, interruptions=0;
      for (const direction of [-1,1]) {
        let active=[node];
        for (let distance=1;distance<=radius;distance++) {
          const t=at+direction*distance;
          if (t<0||t>=graph.length) break;
          const next=graph[t].filter(n=>active.some(a=>links(a,n)));
          for (const a of active) {
            const degree=next.filter(n=>links(a,n)).length;
            branches+=Math.max(0,degree-1);
            if (degree===0) interruptions++;
          }
          if (!next.length) break;
          extent++; active=next;
        }
      }
      out[32+b*3]+=weight*extent/(2*radius);
      out[33+b*3]+=weight*branches/(2*radius);
      out[34+b*3]+=weight*interruptions/2;
    }
  }
  return out;
}

/** Streaming independent spectral observations. The radius is lookahead, shared
 * between hypotheses; native evidence coordinates always map to four 10 ms rows.
 */
export class RidgeObservations {
  readonly frames: number[][]=[];
  private spectra: number[][]=[];
  private produced=0;
  private readonly fbank: Fbank;
  constructor(private readonly radius: 1 | 3 = 3) {
    this.fbank=new Fbank({spectrum: power=>this.spectra.push(Array.from({length:64},(_,b)=> {
      let p=0; for (let k=1+b*4;k<=4+b*4;k++) p+=power[k];
      return Math.log(Math.max(p/4,1.1920928955078125e-7));
    }))});
  }
  private drain(final: boolean) {
    const end=final?this.spectra.length:Math.max(0,this.spectra.length-this.radius);
    for (;this.produced<end;this.produced++) {
      const from=Math.max(0,this.produced-this.radius),to=Math.min(this.spectra.length,this.produced+this.radius+1);
      this.frames.push(ridgePatch(this.spectra.slice(from,to),this.produced-from,this.radius));
    }
  }
  push(audio: Float32Array): void { this.fbank.push(audio); this.drain(false); }
  end(): void { this.drain(true); }
}
export function ridgeFeatures(audio: Float32Array,radius: 1 | 3 = 3): number[][] {
  const o=new RidgeObservations(radius);o.push(audio);o.end();return o.frames;
}
export const ridgeLogadd=(a:number,b:number):number=> {
  const m=Math.max(a,b);return Number.isFinite(m)?m+Math.log(Math.exp(a-m)+Math.exp(b-m)):-Infinity;
};

/** Complete phone chain: no optional target phones, no free hypothesis-specific crop.
 * Subtract the emission-free completion partition so length-dependent path counts
 * do not become evidence. Control marginalizes the connection coordinates of the
 * same fitted diagonal distribution; it preserves all location / energy coordinates.
 */
export function scoreRidgeSequence(model:RidgeModel,raw:number[][],pronunciation:string,connected:boolean):number|null {
  const ids=[...pronunciation].map(p=>model.tokens.indexOf(p)+1);
  if (!ids.length||ids.some(p=>p<1||model.counts[p]<=0)||raw.length<ids.length||raw.length>model.maxFrames||
    raw.some(r=>r.length!==56||r.some(v=>!Number.isFinite(v)))) return null;
  let score=Array<number>(ids.length).fill(-Infinity),prior=score.slice();
  const stay=Math.log(model.stay),step=Math.log1p(-model.stay),dims=connected?56:32;
  for (let t=0;t<raw.length;t++) {
    const x=raw[t].map((v,d)=>(v-model.center[d])/model.scale[d]);
    const values=new Map<number,number>();
    for (const p of new Set(ids)) {
      let v=0;for (let d=0;d<dims;d++) {
        const variance=model.variances[p][d],delta=x[d]-model.means[p][d];
        v-=.5*(Math.log(2*Math.PI*variance)+delta*delta/variance);
      } values.set(p,v);
    }
    const next=score.slice(),nextPrior=prior.slice();
    for (let s=0;s<ids.length;s++) {
      const previous=t===0?(s===0?0:-Infinity):ridgeLogadd(score[s]+stay,s?score[s-1]+step:-Infinity);
      next[s]=previous+values.get(ids[s])!;
      nextPrior[s]=t===0?(s===0?0:-Infinity):ridgeLogadd(prior[s]+stay,s?prior[s-1]+step:-Infinity);
    } score=next;prior=nextPrior;
  }
  const result=score.at(-1)!-prior.at(-1)!;return Number.isFinite(result)?result:null;
}
export function ridgeComparison(model:RidgeModel,window:number[][],c:SequenceEvidenceCandidate):SpectralRidgeEvidence|null {
  const controlWritten=scoreRidgeSequence(model,window,c.written,false),controlRival=scoreRidgeSequence(model,window,c.rival,false);
  const connectedWritten=scoreRidgeSequence(model,window,c.written,true),connectedRival=scoreRidgeSequence(model,window,c.rival,true);
  if (controlWritten===null||controlRival===null||connectedWritten===null||connectedRival===null) return null;
  return {...c,controlWritten,controlRival,connectedWritten,connectedRival};
}
export const ridgeReversal=(c:SpectralRidgeEvidence):boolean=>
  [c.controlWritten,c.controlRival,c.connectedWritten,c.connectedRival].every(Number.isFinite)&&
  c.controlWritten<c.controlRival&&c.connectedWritten>c.connectedRival;
export function validateRidgeModel(value:unknown,inventory:string,tokens:string):RidgeModel {
  const m=value as RidgeModel,n=tokens.length+1;
  const vector=(v:number[],len:number,positive=false)=>Array.isArray(v)&&v.length===len&&v.every(x=>Number.isFinite(x)&&(!positive||x>0));
  const matrix=(v:number[][],positive=false)=>Array.isArray(v)&&v.length===n&&v.every(r=>vector(r,56,positive));
  if (!m||m.schema!==1||m.inventory!==inventory||m.tokens!==tokens||m.frameMs!==40||m.observationMs!==10||
    m.dimensions!==56||m.controlDimensions!==32||![1,3].includes(m.radius)||!vector(m.center,56)||!vector(m.scale,56,true)||
    !matrix(m.means)||!matrix(m.variances,true)||!vector(m.counts,n)||m.counts.some(c=>c<0)||
    !Number.isFinite(m.stay)||m.stay<=0||m.stay>=1||!Number.isInteger(m.maxFrames)||m.maxFrames<1||m.maxFrames>2048)
    throw new Error("Invalid spectral ridge acoustic model or inventory");
  return m;
}
export function addRidgeEvidence(model:RidgeModel|undefined,audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,
  phrases?:PhraseAsset,options:CheckerOptions={},mode:"free"|"memorization"="free",
  inspect?:(c:SpectralRidgeEvidence|null,candidate:SequenceEvidenceCandidate)=>void):PhonemeEvidence[] {
  if (!model||!audio.length) return chunks;
  const cells=insertionCells(chunks);if (!cells.length) return chunks;
  if (cells[0].frameMs!==40) throw new Error("Ridge scorer clock differs from primary evidence");
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const features=ridgeFeatures(audio,model.radius),certificates:SpectralRidgeEvidence[]=[];
  for (const c of candidates.values()) {
    const comparison=c.fromFrame>=0&&c.toFrame*4<=features.length?ridgeComparison(model,features.slice(c.fromFrame*4,c.toFrame*4),c):null;
    inspect?.(comparison,c);if (comparison&&ridgeReversal(comparison)) certificates.push(comparison);
  }
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,ridgeVetoes:[...(p.ridgeVetoes??[]),...certificates]})});
}
