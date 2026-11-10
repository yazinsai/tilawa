import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, MaskingEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

/** Phone emissions and masking parameters frozen on training speakers before inference. */
export interface MaskingModel {
  schema: 1;
  inventory: string;
  tokens: string;
  frameMs: 40;
  dimensions: 24;
  center: number[];
  scale: number[];
  means: number[][];
  variances: number[][];
  counts: number[];
  strength: number;
  maxFrames: number;
}
const bark = (hz: number) => 13 * Math.atan(0.00076 * hz) + 3.5 * Math.atan((hz / 7500) ** 2);
const barkStep = bark(7600) / 24;
const bands = Array.from({ length: 24 }, (_, d) => Array.from({ length: 256 }, (_, k) => k)
  .filter(k => k > 0 && k * 16000 / 512 <= 7600 && Math.min(23, Math.floor(bark(k * 16000 / 512) / barkStep)) === d));

/** Unchanged listener analysis windows; four 10 ms spectra per native evidence frame.
 * No acoustic reconstruction, recognition outputs or expected text enter this view.
 */
export function maskingFeatures(audio: Float32Array): number[][] {
  const observations = new MaskingObservations();
  observations.push(audio);
  return observations.frames;
}

/** Causal observation buffering, invariant to microphone push boundaries. */
export class MaskingObservations {
  readonly frames: number[][] = [];
  private readonly fbank = new Fbank();
  private sums = Array(24).fill(0) as number[];
  private count = 0;
  push(audio: Float32Array): void {
    this.fbank.push(audio, power => {
      for (let d=0; d<24; d++) {
        const b=bands[d];
        this.sums[d] += b.length ? b.reduce((s,k)=>s+power[k],0)/b.length : 0;
      }
      if (++this.count === 4) {
        this.frames.push(this.sums.map(s=>Math.log(Math.max(1e-10,s/4))));
        this.sums.fill(0); this.count=0;
      }
    });
  }
}

/** Simultaneous, signal-dependent masking by stronger neighbours, never a persistent missing-band mask.
 * Asymmetric spreading: higher-frequency cues are more readily masked by lower-frequency energy.
 * The uncertainty is in log-power units and is shared by both pronunciation hypotheses.
 */
export function maskingVariance(row: number[], strength: number): number[] {
  return row.map((x, d) => {
    let threshold = 0;
    for (let j = Math.max(0, d-3); j <= Math.min(row.length-1, d+3); j++) {
      if (j === d || row[j] <= x) continue;
      const slope = j < d ? 12 : 27; // dB per Bark of masking spread
      threshold += Math.exp(row[j]-x) * 10 ** (-slope * Math.abs(j-d) * barkStep / 10);
    }
    return strength * Math.log1p(threshold) ** 2;
  });
}
const logadd = (a: number, b: number) => {
  const m = Math.max(a,b);
  return Number.isFinite(m) ? m + Math.log(Math.exp(a-m) + Math.exp(b-m)) : -Infinity;
};

/** Exact forward sum over monotonic phone boundaries. Each complete pronunciation phone must occupy
 * at least one frame, including repeated articulations. A normalized uniform boundary prior is identical
 * for the control and masked arms. Gaussian convolution analytically integrates observation uncertainty.
 */
export function scoreMaskedSequence(model: MaskingModel, raw: number[][], pronunciation: string,
  masked: boolean): number | null {
  const ids = [...pronunciation].map(p => model.tokens.indexOf(p)+1);
  if (!ids.length || ids.some(p => p < 1 || model.counts[p] === 0) || raw.length < ids.length ||
    raw.length > model.maxFrames || raw.some(r => r.length !== model.dimensions || r.some(v => !Number.isFinite(v)))) return null;
  let scores = ids.map(() => -Infinity), prior = scores.slice();
  for (let t=0; t<raw.length; t++) {
    const row = raw[t], uncertainty = masked ? maskingVariance(row,model.strength) : row.map(() => 0);
    const next = scores.slice(), nextPrior = prior.slice();
    for (let s=0; s<ids.length; s++) {
      const p = ids[s];
      let emission = 0;
      for (let d=0; d<row.length; d++) {
        const x = (row[d]-model.center[d])/model.scale[d];
        const v = model.variances[p][d]+uncertainty[d]/model.scale[d]**2;
        emission -= 0.5*(Math.log(2*Math.PI*v)+(x-model.means[p][d])**2/v);
      }
      const previous = t === 0 ? (s === 0 ? 0 : -Infinity) : logadd(scores[s], s ? scores[s-1] : -Infinity);
      const previousPrior = t === 0 ? (s === 0 ? 0 : -Infinity) : logadd(prior[s], s ? prior[s-1] : -Infinity);
      next[s] = previous + emission;
      nextPrior[s] = previousPrior;
    }
    scores = next; prior = nextPrior;
  }
  const result = scores.at(-1)!-prior.at(-1)!;
  return Number.isFinite(result) ? result : null;
}

export function maskingComparison(model: MaskingModel, window: number[][], c: SequenceEvidenceCandidate): MaskingEvidence | null {
  const controlWritten = scoreMaskedSequence(model,window,c.written,false);
  const controlRival = scoreMaskedSequence(model,window,c.rival,false);
  const maskedWritten = scoreMaskedSequence(model,window,c.written,true);
  const maskedRival = scoreMaskedSequence(model,window,c.rival,true);
  if (controlWritten === null || controlRival === null || maskedWritten === null || maskedRival === null) return null;
  return { ...c, controlWritten, controlRival, maskedWritten, maskedRival };
}
export const maskingReversal = (c: MaskingEvidence) => c.controlWritten < c.controlRival && c.maskedWritten > c.maskedRival;

export function validateMaskingModel(value: unknown, inventory: string, tokens: string): MaskingModel {
  const m=value as MaskingModel, n=tokens.length+1;
  const vector=(v: number[],len: number,positive=false) => Array.isArray(v) && v.length===len && v.every(x=>Number.isFinite(x)&&(!positive||x>0));
  const matrix=(v: number[][],positive=false) => Array.isArray(v) && v.length===n && v.every(r=>vector(r,24,positive));
  if (!m || m.schema!==1 || m.inventory!==inventory || m.tokens!==tokens || m.frameMs!==40 || m.dimensions!==24 ||
    !vector(m.center,24) || !vector(m.scale,24,true) || !matrix(m.means) || !matrix(m.variances,true) ||
    !vector(m.counts,n) || m.counts.some(c=>c<0) || !Number.isFinite(m.strength) || m.strength<0 ||
    !Number.isInteger(m.maxFrames) || m.maxFrames<1 || m.maxFrames>512) throw new Error("Invalid masking acoustic model or inventory");
  return m;
}

/** Discovery runs the original checker with identical mode, passage, policy and acoustic windows.
 * Certificates identify complete comparisons, never just a disputed consonant or a changed threshold.
 */
export function addMaskingEvidence(model: MaskingModel | undefined, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset, options: CheckerOptions = {}, mode: "free" | "memorization" = "free",
  inspect?: (c: MaskingEvidence | null, candidate: SequenceEvidenceCandidate) => void): PhonemeEvidence[] {
  if (!model || !audio.length) return chunks;
  const cells=insertionCells(chunks);
  if (!cells.length) return chunks;
  if (cells[0].frameMs!==model.frameMs) throw new Error("Masking scorer clock differs from primary evidence");
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c) { candidates.set(JSON.stringify(c),c); return true; }});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const features=maskingFeatures(audio), certificates: MaskingEvidence[]=[];
  for (const c of candidates.values()) {
    const comparison=c.fromFrame >= 0 && c.toFrame <= features.length
      ? maskingComparison(model,features.slice(c.fromFrame,c.toFrame),c) : null;
    inspect?.(comparison,c);
    if (comparison && maskingReversal(comparison)) certificates.push(comparison);
  }
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,maskingVetoes:[...(p.maskingVetoes??[]),...certificates]})});
}
