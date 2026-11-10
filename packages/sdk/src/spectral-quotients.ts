import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, SpectralQuotientEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface QuotientEmission { mean: number[]; variance: number[] }
export interface QuotientModel {
  schema: 1;
  inventory: string;
  tokens: string;
  frameMs: 40;
  dimensions: 32;
  scale: number[];
  phoneCounts: number[];
  maxPart: number;
  maxFrames: number;
  quotient: Record<string, QuotientEmission>;
  absolute: Record<string, QuotientEmission>;
}

/** Keep exact FFT frequencies: a stationary multiplicative channel adds the same
 * log response to both intervals. Four 10 ms analyses form each 40 ms observation.
 * The extra frontend never replaces primary listener features.
 */
export class QuotientObservations {
  readonly frames: number[][] = [];
  private pending: number[][] = [];
  private readonly frontend = new Fbank({ onSpectrum: power => {
    this.pending.push(Array.from({ length: 32 }, (_, d) => Math.log(Math.max(power[4 + 7*d], 1e-12))));
    if (this.pending.length === 4) {
      this.frames.push(Array.from({ length: 32 }, (_, d) => this.pending.reduce((s,r) => s+r[d],0)/4));
      this.pending = [];
    }
  } });
  push(audio: Float32Array): void { this.frontend.push(audio); }
}
export function quotientFeatures(audio: Float32Array): number[][] {
  const o = new QuotientObservations(); o.push(audio); return o.frames;
}

const VOWELS = "َُِاۦۥ";
const CONSONANTS = "ءبتثجحخدذرزسشصضطظعغفقكلمنهويڇں۾";
/** Disjoint CV units and residual single phones cover the complete pronunciation.
 * Repetitions are retained. No token edit script or observed spike chooses a boundary.
 */
export function quotientUnits(pronunciation: string): string[] {
  const p = [...pronunciation], units: string[] = [];
  for (let i=0;i<p.length;i++) {
    if (CONSONANTS.includes(p[i]) && i+1<p.length && VOWELS.includes(p[i+1])) units.push(p[i]+p[++i]);
    else units.push(p[i]);
  }
  return units;
}
export function quotientLogadd(a: number, b: number): number {
  const m=Math.max(a,b);
  return Number.isFinite(m) ? m+Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
}
export function quotientGaussian(x: number[], e: QuotientEmission): number {
  return x.reduce((s,v,d) => s-.5*(Math.log(2*Math.PI*e.variance[d])+(v-e.mean[d])**2/e.variance[d]),0);
}

/** Sum over every bounded unit interval and, for CV units, every internal cut.
 * Both arms have identical normalized timing priors and require all phones.
 * The relative arm reads mean consonant minus mean vowel log spectra; its matched
 * absolute control reads the mean consonant spectrum. Both condition the same
 * 32-dimensional emission family on the complete CV identity.
 */
export function scoreQuotientSequence(m: QuotientModel, raw: number[][], pronunciation: string, relative: boolean): number | null {
  const units=quotientUnits(pronunciation), tmax=raw.length, d=m.dimensions;
  if (!units.length || !units.some(u => u.length===2) || tmax>m.maxFrames ||
    [...pronunciation].some(p => m.tokens.indexOf(p)<0 || m.phoneCounts[m.tokens.indexOf(p)]<=0) ||
    units.some(u => !m.absolute[u] || !m.quotient[u]) ||
    raw.some(r => r.length!==d || r.some(v => !Number.isFinite(v)))) return null;
  const prefix=[Array<number>(d).fill(0)];
  for (const r of raw) prefix.push(r.map((v,k) => prefix.at(-1)![k]+v/m.scale[k]));
  const mean=(a: number,b: number) => prefix[b].map((v,k) => (v-prefix[a][k])/(b-a));
  let score=Array<number>(tmax+1).fill(-Infinity), prior=score.slice(); score[0]=prior[0]=0;
  const cache=new Map<string,number>();
  for (const u of units) {
    const next=score.map(() => -Infinity), nextPrior=next.slice();
    const pair=u.length===2, limit=pair ? m.maxPart*2 : m.maxPart;
    for (let b=pair?2:1;b<=tmax;b++) for (let len=pair?2:1;len<=Math.min(limit,b);len++) {
      const a=b-len;
      if (!Number.isFinite(score[a])) continue;
      const key=JSON.stringify([u,a,b]), old=cache.get(key);
      let emission=old ?? -Infinity;
      if (old===undefined) {
        if (!pair) emission=quotientGaussian(mean(a,b),m.absolute[u]);
        else {
          const lo=Math.max(a+1,b-m.maxPart), hi=Math.min(b-1,a+m.maxPart);
          for (let c=lo;c<=hi;c++) {
            const left=mean(a,c), right=mean(c,b);
            const x=relative ? left.map((v,k) => v-right[k]) : left;
            emission=quotientLogadd(emission,quotientGaussian(x,relative ? m.quotient[u] : m.absolute[u]));
          }
          emission-=Math.log(hi-lo+1);
        }
        cache.set(key,emission);
      }
      const duration=-Math.log(limit-(pair?2:1)+1);
      next[b]=quotientLogadd(next[b],score[a]+duration+emission);
      nextPrior[b]=quotientLogadd(nextPrior[b],prior[a]+duration);
    }
    score=next; prior=nextPrior;
  }
  const value=score[tmax]-prior[tmax];
  return Number.isFinite(value) ? value : null;
}

export function quotientComparison(m: QuotientModel, raw: number[][], c: SequenceEvidenceCandidate): SpectralQuotientEvidence | null {
  const controlWritten=scoreQuotientSequence(m,raw,c.written,false), controlRival=scoreQuotientSequence(m,raw,c.rival,false);
  const quotientWritten=scoreQuotientSequence(m,raw,c.written,true), quotientRival=scoreQuotientSequence(m,raw,c.rival,true);
  if (controlWritten===null || controlRival===null || quotientWritten===null || quotientRival===null) return null;
  return { ...c,controlWritten,controlRival,quotientWritten,quotientRival };
}
export const quotientReversal=(c: SpectralQuotientEvidence): boolean =>
  [c.controlWritten,c.controlRival,c.quotientWritten,c.quotientRival].every(Number.isFinite) &&
  c.controlWritten<c.controlRival && c.quotientWritten>c.quotientRival;

export function validateQuotientModel(value: unknown, inventory: string, tokens: string): QuotientModel {
  const m=value as QuotientModel;
  const vector=(v: number[],n: number,positive=false) => Array.isArray(v)&&v.length===n&&v.every(x => Number.isFinite(x)&&(!positive||x>0));
  if (!m || m.schema!==1 || m.inventory!==inventory || m.tokens!==tokens || m.frameMs!==40 || m.dimensions!==32 ||
    !vector(m.scale,32,true) || !vector(m.phoneCounts,tokens.length) || m.phoneCounts.some(c => c<0) ||
    !Number.isInteger(m.maxPart) || m.maxPart<1 || m.maxPart>32 || !Number.isInteger(m.maxFrames) || m.maxFrames<2 || m.maxFrames>1024 ||
    !m.absolute || !m.quotient) throw new Error("Invalid spectral quotient model");
  for (const u of Object.keys(m.absolute)) {
    const pair=quotientUnits(u);
    if (pair.length!==1 || pair[0]!==u || [...u].some(p => !tokens.includes(p)) || !m.quotient[u] ||
      !vector(m.absolute[u].mean,32) || !vector(m.absolute[u].variance,32,true) ||
      !vector(m.quotient[u].mean,32) || !vector(m.quotient[u].variance,32,true)) throw new Error("Invalid spectral quotient emission");
  }
  if (!Object.keys(m.absolute).length || Object.keys(m.quotient).some(u => !m.absolute[u])) throw new Error("Incomplete spectral quotient model");
  return m;

}
/** Discover actual confirming comparisons, preserving complete pronunciations and original frames. */
export function addQuotientEvidence(model: QuotientModel | undefined, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset, options: CheckerOptions = {}, mode: "free" | "memorization" = "free",
  inspect?: (c: SpectralQuotientEvidence | null, candidate: SequenceEvidenceCandidate) => void): PhonemeEvidence[] {
  if (!model || !audio.length) return chunks;
  const cells=insertionCells(chunks);
  if (!cells.length) return chunks;
  if (cells[0].frameMs!==model.frameMs) throw new Error("Quotient scorer clock differs from primary evidence");
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c) { candidates.set(JSON.stringify(c),c); return true; }});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const features=quotientFeatures(audio), certificates: SpectralQuotientEvidence[]=[];
  for (const c of candidates.values()) {
    const comparison=c.fromFrame>=0 && c.toFrame<=features.length
      ? quotientComparison(model,features.slice(c.fromFrame,c.toFrame),c) : null;
    inspect?.(comparison,c);
    if (comparison && quotientReversal(comparison)) certificates.push(comparison);
  }
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,quotientVetoes:[...(p.quotientVetoes??[]),...certificates]})});
}
