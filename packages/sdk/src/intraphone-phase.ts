import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, IntraphoneEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

/** Parameters selected on speaker-disjoint training partitions, never warning windows. */
export interface PhaseModel {
  schema: 1;
  inventory: string;
  tokens: string;
  frameMs: 40;
  observationMs: 10;
  dimensions: 20;
  center: number[];
  scale: number[];
  means: number[][][];
  variances: number[][][];
  counts: number[];
  stay: number[];
  maxFrames: number;
}

/** The primary 25 ms / 10 ms frontend, averaged into twenty spectral coordinates.
 * Each native evidence frame indexes four observations; no time pooling discards the phases.
 */
export class PhaseObservations {
  readonly frames: number[][] = [];
  private readonly fbank = new Fbank();
  push(audio: Float32Array): void {
    for (const row of this.fbank.push(audio)) this.frames.push(Array.from({ length: 20 }, (_, d) =>
      (row[4*d] + row[4*d+1] + row[4*d+2] + row[4*d+3]) / 4));
  }
}
export function phaseFeatures(audio: Float32Array): number[][] {
  const observations = new PhaseObservations();
  observations.push(audio);
  return observations.frames;
}
export const phaseLogadd = (a: number, b: number): number => {
  const m = Math.max(a, b);
  return Number.isFinite(m) ? m + Math.log(Math.exp(a-m) + Math.exp(b-m)) : -Infinity;
};

/** Complete, left-to-right attack / sustain / release chain. Every phone visits every phase;
 * self loops marginalize phase duration, release-to-attack edges marginalize phone boundaries.
 * The control uses the same fitted Gaussian mixture at all three phases of a phone, with exactly
 * the same transitions and completion-conditioned timing prior as the ordered arm.
 */
export function scorePhaseSequence(model: PhaseModel, raw: number[][], pronunciation: string, ordered: boolean): number | null {
  const ids = [...pronunciation].map(p => model.tokens.indexOf(p)+1);
  if (!ids.length || ids.some(p => p < 1 || model.counts[p] <= 0) || raw.length < ids.length*3 ||
    raw.length > model.maxFrames || raw.some(r => r.length !== model.dimensions || r.some(v => !Number.isFinite(v)))) return null;
  const n = ids.length*3;
  let score = Array<number>(n).fill(-Infinity), prior = score.slice();
  for (let t=0; t<raw.length; t++) {
    const x = raw[t].map((v,d) => (v-model.center[d])/model.scale[d]);
    const emissions = new Map<number,number[]>();
    for (const p of new Set(ids)) {
      const values = model.means[p].map((mu,q) => mu.reduce((s,m,d) => {
        const v=model.variances[p][q][d];
        return s-0.5*(Math.log(2*Math.PI*v)+(x[d]-m)**2/v);
      },0));
      const pooled = values.reduce(phaseLogadd,-Infinity)-Math.log(3);
      emissions.set(p,ordered ? values : [pooled,pooled,pooled]);
    }
    const next=score.slice(), nextPrior=prior.slice();
    for (let s=0; s<n; s++) {
      const q=s%3;
      const stay=Math.log(model.stay[q]);
      const advance=Math.log1p(-model.stay[(s+2)%3]);
      const previous=t===0 ? (s===0 ? 0 : -Infinity) : phaseLogadd(score[s]+stay,s ? score[s-1]+advance : -Infinity);
      const previousPrior=t===0 ? (s===0 ? 0 : -Infinity) : phaseLogadd(prior[s]+stay,s ? prior[s-1]+advance : -Infinity);
      next[s]=previous+emissions.get(ids[Math.floor(s/3)])![q];
      nextPrior[s]=previousPrior;
    }
    score=next; prior=nextPrior;
  }
  const result=score[n-1]-prior[n-1];
  return Number.isFinite(result) ? result : null;
}

export function phaseComparison(model: PhaseModel, window: number[][], c: SequenceEvidenceCandidate): IntraphoneEvidence | null {
  const controlWritten=scorePhaseSequence(model,window,c.written,false);
  const controlRival=scorePhaseSequence(model,window,c.rival,false);
  const orderedWritten=scorePhaseSequence(model,window,c.written,true);
  const orderedRival=scorePhaseSequence(model,window,c.rival,true);
  if (controlWritten===null || controlRival===null || orderedWritten===null || orderedRival===null) return null;
  return { ...c, controlWritten, controlRival, orderedWritten, orderedRival };
}
export const phaseReversal = (c: IntraphoneEvidence): boolean =>
  [c.controlWritten,c.controlRival,c.orderedWritten,c.orderedRival].every(Number.isFinite) &&
  c.controlWritten<c.controlRival && c.orderedWritten>c.orderedRival;

export function validatePhaseModel(value: unknown, inventory: string, tokens: string): PhaseModel {
  const m=value as PhaseModel, n=tokens.length+1;
  const vector=(v: number[],len: number,positive=false) => Array.isArray(v) && v.length===len && v.every(x=>Number.isFinite(x)&&(!positive||x>0));
  const tensor=(v: number[][][],positive=false) => Array.isArray(v) && v.length===n && v.every(p=>Array.isArray(p)&&p.length===3&&p.every(r=>vector(r,20,positive)));
  if (!m || m.schema!==1 || m.inventory!==inventory || m.tokens!==tokens || m.frameMs!==40 || m.observationMs!==10 || m.dimensions!==20 ||
    !vector(m.center,20) || !vector(m.scale,20,true) || !tensor(m.means) || !tensor(m.variances,true) ||
    !vector(m.counts,n) || m.counts.some(c=>c<0) || !vector(m.stay,3,true) || m.stay.some(p=>p>=1) ||
    !Number.isInteger(m.maxFrames) || m.maxFrames<3 || m.maxFrames>2048) throw new Error("Invalid intraphone acoustic model or inventory");
  return m;
}

/** Discover actual confirming comparisons, preserving complete pronunciations and original frames. */
export function addPhaseEvidence(model: PhaseModel | undefined, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset, options: CheckerOptions = {}, mode: "free" | "memorization" = "free",
  inspect?: (c: IntraphoneEvidence | null, candidate: SequenceEvidenceCandidate) => void): PhonemeEvidence[] {
  if (!model || !audio.length) return chunks;
  const cells=insertionCells(chunks);
  if (!cells.length) return chunks;
  if (cells[0].frameMs!==model.frameMs) throw new Error("Phase scorer clock differs from primary evidence");
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c) { candidates.set(JSON.stringify(c),c); return true; }});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const features=phaseFeatures(audio), certificates: IntraphoneEvidence[]=[];
  for (const c of candidates.values()) {
    const comparison=c.fromFrame>=0 && c.toFrame*4<=features.length
      ? phaseComparison(model,features.slice(c.fromFrame*4,c.toFrame*4),c) : null;
    inspect?.(comparison,c);
    if (comparison && phaseReversal(comparison)) certificates.push(comparison);
  }
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,phaseVetoes:[...(p.phaseVetoes??[]),...certificates]})});
}
