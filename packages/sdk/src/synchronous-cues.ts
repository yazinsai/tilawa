import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate, SynchronousCueEvidence } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

/** Three categorical emission streams, fitted on training speakers only. */
export interface CueModel {
  schema: 1;
  inventory: string;
  tokens: string;
  frameMs: 40;
  observationMs: 10;
  dimensions: 20;
  center: number[];
  scale: number[];
  /** Blank and every original phone map to place, manner and voicing classes. */
  classes: number[][];
  means: number[][][];
  variances: number[][][];
  counts: number[];
  stay: number;
  maxFrames: number;
}

/** Same native log-mel frontend and clock in training and inference. */
export class CueObservations {
  readonly frames: number[][] = [];
  private readonly fbank = new Fbank();
  push(audio: Float32Array): void {
    for (const row of this.fbank.push(audio)) this.frames.push(Array.from({ length: 20 }, (_, d) =>
      (row[4*d] + row[4*d+1] + row[4*d+2] + row[4*d+3]) / 4));
  }
}
export function cueFeatures(audio: Float32Array): number[][] {
  const observations = new CueObservations();
  observations.push(audio);
  return observations.frames;
}
export const cueLogadd = (a: number, b: number): number => {
  const m = Math.max(a, b);
  return Number.isFinite(m) ? m + Math.log(Math.exp(a-m) + Math.exp(b-m)) : -Infinity;
};

/** Original phone identity determines topology even when two phones share feature classes.
 * Optional background states separate articulations; all pronunciation phones are mandatory.
 */
export function cueGraph(ids: number[], stay: number): { labels: number[]; incoming: [number, number][][] } {
  const labels = [0, ...ids.flatMap(p => [p, 0])];
  const incoming: [number, number][][] = labels.map(() => []);
  for (let s = 0; s < labels.length; s++) {
    const next = s+1 < labels.length ? [s+1] : [];
    if (s%2 === 1 && s+2 < labels.length && labels[s] !== labels[s+2]) next.push(s+2);
    incoming[s].push([s, Math.log(next.length ? stay : 1)]);
    for (const to of next) incoming[to].push([s, Math.log((1-stay)/next.length)]);
  }
  return { labels, incoming };
}

/** Marginalize the complete graph and condition its prior on reaching an accepting state.
 * Synchronization sums the three emissions BEFORE summing boundaries; the control sums three
 * separately marginalized graphs. Emissions, topology, duration priors and observations match.
 */
export function scoreCueSequence(model: CueModel, raw: number[][], pronunciation: string, synchronized: boolean): number | null {
  const ids = [...pronunciation].map(p => model.tokens.indexOf(p)+1);
  if (!ids.length || ids.some(p => p < 1 || model.counts[p] <= 0) || raw.length < ids.length ||
    raw.length > model.maxFrames || raw.some(r => r.length !== 20 || r.some(v => !Number.isFinite(v)))) return null;
  const {labels,incoming} = cueGraph(ids,model.stay), n = labels.length;
  // Each head reads the same observations, without consuming the primary network's logits.
  const streams = Array.from({length:3},(_,h) => raw.map(row => model.means[h].map((mu,c) =>
    mu.reduce((sum,m,d) => {
      const x=(row[d]-model.center[d])/model.scale[d], v=model.variances[h][c][d];
      return sum-.5*(Math.log(2*Math.PI*v)+(x-m)**2/v);
    },0))));
  const forward = (heads: number[]): number => {
    let score = Array(n).fill(-Infinity), prior = score.slice();
    for (let t=0;t<raw.length;t++) {
      const next = score.slice(), np = prior.slice();
      for (let s=0;s<n;s++) {
        const a = t ? incoming[s].reduce((v,[p,w]) => cueLogadd(v,score[p]+w),-Infinity)
          : s<=1 ? -Math.log(2) : -Infinity;
        np[s] = t ? incoming[s].reduce((v,[p,w]) => cueLogadd(v,prior[p]+w),-Infinity)
          : s<=1 ? -Math.log(2) : -Infinity;
        next[s]=a+heads.reduce((v,h) => v+streams[h][t][model.classes[labels[s]][h]],0);
      }
      score=next; prior=np;
    }
    return cueLogadd(score[n-1],score[n-2])-cueLogadd(prior[n-1],prior[n-2]);
  };
  const value = synchronized ? forward([0,1,2]) : [0,1,2].reduce((v,h) => v+forward([h]),0);
  return Number.isFinite(value) ? value : null;
}

export function cueComparison(model: CueModel, window: number[][], c: SequenceEvidenceCandidate): SynchronousCueEvidence | null {
  const profile = (s: string) => [...s].map(p => model.classes[model.tokens.indexOf(p)+1]);
  // Different spellings with identical feature predictions provide no synchrony witness.
  if (JSON.stringify(profile(c.written)) === JSON.stringify(profile(c.rival))) return null;
  const independentWritten=scoreCueSequence(model,window,c.written,false);
  const independentRival=scoreCueSequence(model,window,c.rival,false);
  const synchronizedWritten=scoreCueSequence(model,window,c.written,true);
  const synchronizedRival=scoreCueSequence(model,window,c.rival,true);
  if ([independentWritten,independentRival,synchronizedWritten,synchronizedRival].some(v => v===null)) return null;
  return {...c, independentWritten:independentWritten!, independentRival:independentRival!,
    synchronizedWritten:synchronizedWritten!, synchronizedRival:synchronizedRival!};
}
export const cueReversal = (c: SynchronousCueEvidence): boolean =>
  [c.independentWritten,c.independentRival,c.synchronizedWritten,c.synchronizedRival].every(Number.isFinite) &&
  c.independentWritten<c.independentRival && c.synchronizedWritten>c.synchronizedRival;

export function validateCueModel(value: unknown, inventory: string, tokens: string): CueModel {
  const m=value as CueModel, n=tokens.length+1;
  const vector=(v: number[],len: number,positive=false) => Array.isArray(v) && v.length===len && v.every(x=>Number.isFinite(x)&&(!positive||x>0));
  if (!m || m.schema!==1 || m.inventory!==inventory || m.tokens!==tokens || m.frameMs!==40 || m.observationMs!==10 || m.dimensions!==20 ||
    !vector(m.center,20) || !vector(m.scale,20,true) || !vector(m.counts,n) || m.counts.some(c=>c<0) ||
    !Number.isFinite(m.stay) || m.stay<=0 || m.stay>=1 || !Number.isInteger(m.maxFrames) || m.maxFrames<1 || m.maxFrames>4096 ||
    !Array.isArray(m.means) || m.means.length!==3 || !Array.isArray(m.variances) || m.variances.length!==3 ||
    !m.means.every((head,h)=>Array.isArray(head)&&head.length>0&&Array.isArray(m.variances[h])&&m.variances[h].length===head.length&&
      head.every((mu,c)=>vector(mu,20)&&vector(m.variances[h][c],20,true))) ||
    !Array.isArray(m.classes) || m.classes.length!==n || !m.classes.every(row=>Array.isArray(row)&&row.length===3&&
      row.every((c,h)=>Number.isInteger(c)&&c>=0&&c<m.means[h].length))) throw new Error("Invalid synchronous cue acoustic model or inventory");
  return m;
}

/** Replay the unchanged engine only to discover its actual comparisons, then attach exact-window certificates. */
export function addCueEvidence(model: CueModel | undefined, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset, options: CheckerOptions = {}, mode: "free" | "memorization" = "free",
  inspect?: (c: SynchronousCueEvidence | null, candidate: SequenceEvidenceCandidate) => void): PhonemeEvidence[] {
  if (!model || !audio.length) return chunks;
  const cells=insertionCells(chunks);
  if (!cells.length) return chunks;
  if (cells[0].frameMs!==model.frameMs) throw new Error("Cue scorer clock differs from primary evidence");
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c) {candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const features=cueFeatures(audio), certificates: SynchronousCueEvidence[]=[];
  for (const c of candidates.values()) {
    const comparison=c.fromFrame>=0 && c.toFrame*4<=features.length
      ? cueComparison(model,features.slice(c.fromFrame*4,c.toFrame*4),c) : null;
    inspect?.(comparison,c);
    if (comparison && cueReversal(comparison)) certificates.push(comparison);
  }
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,cueVetoes:[...(p.cueVetoes??[]),...certificates]})});
}
