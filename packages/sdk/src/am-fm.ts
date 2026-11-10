import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { AmFmEvidence, CheckerOptions, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { insertionCells } from "./seam-insertion.js";

/** Complex two-pole filter centres, bandwidths and 10 ms observation clock are fixed before validation. */
const CENTRES = [250, 450, 750, 1150, 1750, 2600, 3800, 5500];
const WIDTHS = [100, 130, 180, 240, 340, 480, 650, 850];
export interface AmFmModel {
  schema: 1; inventory: string; tokens: string; frameMs: 40; observationMs: 10;
  bands: number[]; center: number[][]; scale: number[][];
  means: number[][]; variances: number[][]; covariance: number[][];
  counts: number[]; stay: number; maxFrames: number;
}

/** Raw waveform only. The complex filter's phase increment measures instantaneous frequency;
 * jointly observed changes of log amplitude and frequency retain their signed coordination.
 * Frequency observations at weak amplitudes carry explicit reliability, rather than a fabricated zero.
 */
export class AmFmObservations {
  readonly frames: number[][] = [];
  private states = CENTRES.map((f,b) => ({
    r: Math.exp(-Math.PI*WIDTHS[b]/16000), c: Math.cos(2*Math.PI*f/16000), s: Math.sin(2*Math.PI*f/16000),
    ar:0, ai:0, br:0, bi:0, energy:0, motion:0, weight:0, previousEnergy:0, previousFrequency:0, energies:new Float64Array(400), motions:new Float64Array(400), weights:new Float64Array(400),
  }));
  private samples = 0;
  private power = 0;
  private powers = new Float64Array(400);
  push(audio: Float32Array): void {
    for (const sample of audio) {
      const slot=this.samples%400;
      this.power += sample*sample-this.powers[slot]; this.powers[slot]=sample*sample;
      for (let b=0;b<this.states.length;b++) {
        const z=this.states[b], oldr=z.br, oldi=z.bi;
        const ar=z.r*(z.c*z.ar-z.s*z.ai)+(1-z.r)*sample;
        const ai=z.r*(z.s*z.ar+z.c*z.ai);
        const br=z.r*(z.c*z.br-z.s*z.bi)+(1-z.r)*ar;
        const bi=z.r*(z.s*z.br+z.c*z.bi)+(1-z.r)*ai;
        z.ar=ar; z.ai=ai; z.br=br; z.bi=bi;
        const e=br*br+bi*bi, olde=oldr*oldr+oldi*oldi;
        const weight=Math.sqrt(e*olde);
        // Residual phase after removing the filter centre; unwrap locally, never across silence.
        const angle=Math.atan2(bi*oldr-br*oldi,br*oldr+bi*oldi)-2*Math.PI*CENTRES[b]/16000;
        const residual=Math.atan2(Math.sin(angle),Math.cos(angle))*16000/(2*Math.PI*WIDTHS[b]);
        z.energy+=e-z.energies[slot]; z.motion+=weight*residual-z.motions[slot]; z.weight+=weight-z.weights[slot];
        z.energies[slot]=e; z.motions[slot]=weight*residual; z.weights[slot]=weight;
      }
      this.samples++;
      if (this.samples<400 || (this.samples-400)%160!==0) continue;
      const row:number[]=[];
      for (const z of this.states) {
        const energy=Math.max(0,z.energy/400), frequency=z.weight>0 ? z.motion/z.weight : 0;
        const floor=Math.max(Math.max(0,this.power)/400*0.01,1e-30);
        const reliable=Math.min(energy,z.previousEnergy);
        row.push(Math.log((energy+floor)/(z.previousEnergy+floor))/2,
          frequency-z.previousFrequency, reliable/(reliable+floor));
        z.previousEnergy=energy; z.previousFrequency=frequency;
      }
      // Exact native 25 ms snipped windows, advanced by 10 ms.
      this.frames.push(row);
    }
  }
}
export function amFmFeatures(audio: Float32Array): number[][] {
  const o=new AmFmObservations(); o.push(audio); return o.frames;
}
export const logadd=(a:number,b:number):number=> {
  const m=Math.max(a,b); return Number.isFinite(m) ? m+Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};

/** Exact Gaussian marginal over unreliable IF, with the same univariate marginals in the control.
 * q=0 integrates IF out completely. q=1 uses the joint (or factorized) distribution.
 */
export function amFmEmission(m: AmFmModel, raw:number[], p:number, joint:boolean):number {
  let value=0;
  for (let j=0;j<m.bands.length;j++) {
    const b=m.bands[j], a=(raw[b*3]-m.center[j][0])/m.scale[j][0], f=(raw[b*3+1]-m.center[j][1])/m.scale[j][1];
    const da=a-m.means[p][j*2], df=f-m.means[p][j*2+1], va=m.variances[p][j*2], vf=m.variances[p][j*2+1];
    const c=joint ? m.covariance[p][j] : 0, det=va*vf-c*c;
    const amplitude=-.5*(Math.log(2*Math.PI*va)+da*da/va);
    // Divide by the shared standardized frequency background density. This makes
    // observed and marginalized branches comparable on the same observation measure.
    const full=-Math.log(2*Math.PI)-.5*Math.log(det)-.5*(vf*da*da+va*df*df-2*c*da*df)/det
      +.5*(Math.log(2*Math.PI)+f*f);
    const q=raw[b*3+2];
    value+=q===0 ? amplitude : q===1 ? full : logadd(Math.log(q)+full,Math.log1p(-q)+amplitude);
  }
  return value;
}

/** Complete phone chain; every phone emits, and normalized timing priors integrate all boundaries.
 * Both arms use identical learned durations, means, variances and observations; only covariance differs.
 */
export function scoreAmFmSequence(m:AmFmModel, raw:number[][], pronunciation:string, joint:boolean):number|null {
  const ids=[...pronunciation].map(p=>m.tokens.indexOf(p)+1);
  if (!ids.length || ids.some(p=>p<1 || m.counts[p]<=0) || raw.length<ids.length || raw.length>m.maxFrames ||
    raw.some(r=>r.length!==24 || r.some(v=>!Number.isFinite(v)) || m.bands.some(b=>r[b*3+2]<0 || r[b*3+2]>1))) return null;
  let scores=ids.map(()=>-Infinity), prior=scores.slice();
  for (let t=0;t<raw.length;t++) {
    const next=scores.slice(), np=prior.slice(), e=new Map<number,number>();
    for (const p of new Set(ids)) e.set(p,amFmEmission(m,raw[t],p,joint));
    for (let s=0;s<ids.length;s++) {
      const a=t===0 ? (s===0 ? 0 : -Infinity) : logadd(scores[s]+Math.log(m.stay),s ? scores[s-1]+Math.log1p(-m.stay) : -Infinity);
      np[s]=t===0 ? (s===0 ? 0 : -Infinity) : logadd(prior[s]+Math.log(m.stay),s ? prior[s-1]+Math.log1p(-m.stay) : -Infinity);
      next[s]=a+e.get(ids[s])!;
    }
    scores=next; prior=np;
  }
  const value=scores.at(-1)!-prior.at(-1)!;
  return Number.isFinite(value) ? value : null;
}
export function amFmComparison(m:AmFmModel, window:number[][], c:SequenceEvidenceCandidate):AmFmEvidence|null {
  const controlWritten=scoreAmFmSequence(m,window,c.written,false), controlRival=scoreAmFmSequence(m,window,c.rival,false);
  const jointWritten=scoreAmFmSequence(m,window,c.written,true), jointRival=scoreAmFmSequence(m,window,c.rival,true);
  if (controlWritten===null || controlRival===null || jointWritten===null || jointRival===null) return null;
  return {...c,controlWritten,controlRival,jointWritten,jointRival};
}
export const amFmReversal=(c:AmFmEvidence):boolean=>
  [c.controlWritten,c.controlRival,c.jointWritten,c.jointRival].every(Number.isFinite) && c.controlWritten<c.controlRival && c.jointWritten>c.jointRival;
export function validateAmFmModel(value:unknown, inventory:string,tokens:string):AmFmModel {
  const m=value as AmFmModel,n=tokens.length+1;
  const vector=(v:number[],len:number,pos=false)=>Array.isArray(v)&&v.length===len&&v.every(x=>Number.isFinite(x)&&(!pos||x>0));
  const rows=(v:number[][],len:number,width:number,pos=false)=>Array.isArray(v)&&v.length===len&&v.every(r=>vector(r,width,pos));
  if (!m || m.schema!==1 || m.inventory!==inventory || m.tokens!==tokens || m.frameMs!==40 || m.observationMs!==10 ||
    !Array.isArray(m.bands) || !m.bands.length || new Set(m.bands).size!==m.bands.length || m.bands.some(b=>!Number.isInteger(b)||b<0||b>=8) ||
    !rows(m.center,m.bands.length,2) || !rows(m.scale,m.bands.length,2,true) || !rows(m.means,n,m.bands.length*2) ||
    !rows(m.variances,n,m.bands.length*2,true) || !rows(m.covariance,n,m.bands.length) || !vector(m.counts,n) || m.counts.some(c=>c<0) ||
    !Number.isFinite(m.stay)||m.stay<=0||m.stay>=1 || !Number.isInteger(m.maxFrames)||m.maxFrames<1||m.maxFrames>4096 ||
    m.covariance.some((r,p)=>r.some((c,b)=>c*c>=m.variances[p][b*2]*m.variances[p][b*2+1]))) throw new Error("Invalid AM-FM model or inventory");
  return m;
}

/** Discover the actual confirming comparisons, with complete pronunciations and original windows. */
export function addAmFmEvidence(m:AmFmModel|undefined,audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,
  phrases?:PhraseAsset,options:CheckerOptions={},mode:"free"|"memorization"="free",
  inspect?:(c:AmFmEvidence|null,candidate:SequenceEvidenceCandidate)=>void):PhonemeEvidence[] {
  if (!m || !audio.length) return chunks;
  const cells=insertionCells(chunks); if (!cells.length) return chunks;
  if (cells[0].frameMs!==m.frameMs) throw new Error("AM-FM scorer clock differs from primary evidence");
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e))); checker.end(tracker.end());
  const features=amFmFeatures(audio),certificates:AmFmEvidence[]=[];
  for (const c of candidates.values()) {
    const comparison=c.fromFrame>=0&&c.toFrame*4<=features.length ? amFmComparison(m,features.slice(c.fromFrame*4,c.toFrame*4),c) : null;
    inspect?.(comparison,c); if (comparison && amFmReversal(comparison)) certificates.push(comparison);
  }
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,amFmVetoes:[...(p.amFmVetoes??[]),...certificates]})});
}
