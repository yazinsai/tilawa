import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, PhonemeEvidence, PhraseAsset, ReferenceIndex, ReverberantEvidence, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

/** Training-frozen normalized spectral emissions and causal finite decay kernel. */
export interface ReverberantModel {
  schema: 1;
  inventory: string;
  tokens: string;
  frameMs: 40;
  dimensions: 20;
  history: 4;
  mix: number;
  decay: number;
  spectra: number[][];
  variances: number[][];
  counts: number[];
  stay: number[];
  initial: number[];
  maxFrames: number;
}

/** Retain the primary frontend and clock. Average POWER over four 10 ms frames and
 * four adjacent mel bins, then normalize spectral mass. No waveform is reconstructed.
 */
export class ReverberantObservations {
  readonly frames: number[][] = [];
  private readonly fbank = new Fbank();
  private pending: Float32Array[] = [];
  push(audio: Float32Array): void {
    for (const row of this.fbank.push(audio)) {
      this.pending.push(row);
      if (this.pending.length !== 4) continue;
      const power = Array.from({ length: 20 }, (_, d) => this.pending.reduce((s, r) =>
        s + Math.exp(r[4*d]) + Math.exp(r[4*d+1]) + Math.exp(r[4*d+2]) + Math.exp(r[4*d+3]), 0));
      const mass = power.reduce((s, p) => s+p, 0);
      this.frames.push(power.map(p => Math.log(p/mass)));
      this.pending = [];
    }
  }
}
export function reverberantFeatures(audio: Float32Array): number[][] {
  const observations = new ReverberantObservations();
  observations.push(audio);
  return observations.frames;
}
export function logadd(a: number, b: number): number {
  const m = Math.max(a,b);
  return Number.isFinite(m) ? m+Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
}
const popcount = (bits: number): number => {
  let n=0;
  for (; bits; bits &= bits-1) n++;
  return n;
};
export function decayWeights(m: ReverberantModel): number[] {
  const tail = Array.from({length:m.history}, (_,l) => m.decay**l);
  const norm = tail.reduce((s,v) => s+v,0);
  return [1-m.mix, ...tail.map(v => m.mix*v/norm)];
}

/** Exact finite-memory forward sum. A state stores the phone position, the last
 * four stay/advance decisions, and (for the first four frames) a latent incoming
 * spectrum. At frame four that incoming state is summed out, never replaced by
 * the observed spectrum. All phone boundaries and all reachable histories are
 * retained. Every articulation, including equal adjacent labels, is mandatory.
 *
 * The instantaneous control disables only the convolution, retaining the same
 * fitted spectra, covariance, complete-phone topology and normalized timing prior.
 */
export function scoreReverberantSequence(m: ReverberantModel, raw: number[][], pronunciation: string, memory: boolean): number | null {
  const ids = [...pronunciation].map(p => m.tokens.indexOf(p)+1), n=ids.length;
  if (!n || ids.some(p => p<1 || m.counts[p]<=0) || raw.length<n || raw.length>m.maxFrames ||
    raw.some(r => r.length!==m.dimensions || r.some(v => !Number.isFinite(v)))) return null;
  const density = (x: number[], spectrum: number[], p: number) => x.reduce((sum,y,d) => {
    const v=m.variances[p][d], residual=y-Math.log(spectrum[d]);
    return sum-.5*(Math.log(2*Math.PI*v)+residual*residual/v);
  },0);
  let prior=Array<number>(n).fill(-Infinity); prior[0]=0;
  for (let t=1;t<raw.length;t++) prior=prior.map((v,i) => logadd(v+Math.log(m.stay[ids[i]]),
    i ? prior[i-1]+Math.log1p(-m.stay[ids[i-1]]) : -Infinity));
  let result: number;
  if (!memory) {
    let scores=Array<number>(n).fill(-Infinity);
    scores[0]=density(raw[0],m.spectra[ids[0]],ids[0]);
    for (let t=1;t<raw.length;t++) scores=scores.map((v,i) =>
      logadd(v+Math.log(m.stay[ids[i]]),i ? scores[i-1]+Math.log1p(-m.stay[ids[i-1]]) : -Infinity)+
      density(raw[t],m.spectra[ids[i]],ids[i]));
    result=scores[n-1]-prior[n-1];
  } else {
    const weights=decayWeights(m), masks=1<<m.history, vocab=m.tokens.length+1;
    const key=(i:number,mask:number,seed:number) => (i*masks+mask)*vocab+seed;
    const emission=(t:number,i:number,mask:number,seed:number) => {
      const spectrum=m.spectra[ids[i]].map(v => v*weights[0]);
      for (let l=1;l<=m.history;l++) {
        const p=t<l ? seed : ids[i-popcount(mask & ((1<<l)-1))];
        for (let d=0;d<m.dimensions;d++) spectrum[d]+=weights[l]*m.spectra[p][d];
      }
      return density(raw[t],spectrum,ids[i]);
    };
    let scores=new Map<number,number>();
    for (let seed=0;seed<vocab;seed++) if (m.initial[seed]>0)
      scores.set(key(0,0,seed),Math.log(m.initial[seed])+emission(0,0,0,seed));
    for (let t=1;t<raw.length;t++) {
      const next=new Map<number,number>();
      for (const [k,value] of scores) {
        const seed=t>=m.history ? 0 : k%vocab, packed=Math.floor(k/vocab), mask=packed%masks, i=Math.floor(packed/masks);
        for (const advance of [0,1]) {
          if (i+advance>=n) continue;
          const h=((mask<<1)|advance)&(masks-1), j=i+advance, dest=key(j,h,seed);
          const transition=advance ? Math.log1p(-m.stay[ids[i]]) : Math.log(m.stay[ids[i]]);
          next.set(dest,logadd(next.get(dest)??-Infinity,value+transition));
        }
      }
      for (const [k,value] of next) {
        const packed=Math.floor(k/vocab);
        next.set(k,value+emission(t,Math.floor(packed/masks),packed%masks,k%vocab));
      }
      scores=next;
    }
    let total=-Infinity;
    for (const [k,value] of scores) if (Math.floor(k/(vocab*masks))===n-1) total=logadd(total,value);
    result=total-prior[n-1];
  }
  return Number.isFinite(result) ? result : null;
}

export function reverberantComparison(m: ReverberantModel, raw: number[][], c: SequenceEvidenceCandidate): ReverberantEvidence | null {
  const controlWritten=scoreReverberantSequence(m,raw,c.written,false), controlRival=scoreReverberantSequence(m,raw,c.rival,false);
  const memoryWritten=scoreReverberantSequence(m,raw,c.written,true), memoryRival=scoreReverberantSequence(m,raw,c.rival,true);
  if (controlWritten===null || controlRival===null || memoryWritten===null || memoryRival===null) return null;
  return {...c,controlWritten,controlRival,memoryWritten,memoryRival};
}
export const reverberantReversal = (c: ReverberantEvidence): boolean =>
  [c.controlWritten,c.controlRival,c.memoryWritten,c.memoryRival].every(Number.isFinite) &&
  c.controlWritten<c.controlRival && c.memoryWritten>c.memoryRival;

export function validateReverberantModel(value: unknown, inventory: string, tokens: string): ReverberantModel {
  const m=value as ReverberantModel, n=tokens.length+1;
  const vector=(v:number[],len:number,positive=false) => Array.isArray(v) && v.length===len && v.every(x=>Number.isFinite(x)&&(!positive||x>0));
  const matrix=(v:number[][]) => Array.isArray(v) && v.length===n && v.every(r=>vector(r,20,true));
  if (!m || m.schema!==1 || m.inventory!==inventory || m.tokens!==tokens || m.frameMs!==40 || m.dimensions!==20 || m.history!==4 ||
    !Number.isFinite(m.mix) || m.mix<0 || m.mix>=1 || !Number.isFinite(m.decay) || m.decay<=0 || m.decay>=1 ||
    !matrix(m.spectra) || m.spectra.some(r=>Math.abs(r.reduce((s,v)=>s+v,0)-1)>1e-5) || !matrix(m.variances) ||
    !vector(m.counts,n) || m.counts.some(v=>v<0) || !vector(m.stay,n,true) || m.stay.some(v=>v>=1) ||
    !vector(m.initial,n) || m.initial.some(v=>v<0) || Math.abs(m.initial.reduce((s,v)=>s+v,0)-1)>1e-5 ||
    !Number.isInteger(m.maxFrames) || m.maxFrames<4 || m.maxFrames>2048) throw new Error("Invalid reverberant spectral model or inventory");
  return m;
}

/** Generate certificates from the actual confirming comparisons, after existing
 * gap and insertion enrichment. Only metadata is added; primary rows stay intact.
 */
export function addReverberantEvidence(m: ReverberantModel | undefined, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset, options: CheckerOptions = {}, mode: "free" | "memorization" = "free",
  inspect?: (comparison: ReverberantEvidence | null, candidate: SequenceEvidenceCandidate) => void): PhonemeEvidence[] {
  if (!m || !audio.length) return chunks;
  const cells=insertionCells(chunks);
  if (!cells.length) return chunks;
  if (cells[0].frameMs!==m.frameMs) throw new Error("Reverberant scorer clock differs from primary evidence");
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c) { candidates.set(JSON.stringify(c),c); return true; }});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const features=reverberantFeatures(audio), certificates:ReverberantEvidence[]=[];
  for (const c of candidates.values()) {
    const comparison=c.fromFrame>=0 && c.toFrame<=features.length
      ? reverberantComparison(m,features.slice(c.fromFrame,c.toFrame),c) : null;
    inspect?.(comparison,c);
    if (comparison && reverberantReversal(comparison)) certificates.push(comparison);
  }
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first ? e : {...e,phonemes:e.phonemes.map((p,j)=>j ? p :
    {...p,reverberantVetoes:[...(p.reverberantVetoes??[]),...certificates]})});
}
