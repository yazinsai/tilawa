import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, SparseFactorEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

/** None labels are canonical supervision only when the recording is usable and has no edits/repeats. */
export function eligibleSparseFactorTake(r: {kind?:string;split?:string;speaker?:string;exclude?:boolean|string|null;excluded?:boolean|string|null;scored?:boolean;events?:unknown[];non_errors?:unknown[]}, excluded:ReadonlySet<string>):boolean {
  return r.kind === "none" && r.split === "train" && !!r.speaker && !excluded.has(r.speaker) &&
    !r.exclude && !r.excluded && r.scored !== false && !r.events?.length && !r.non_errors?.length;
}

export interface SparseFactorArm {
  encoder: number[][]; encoderBias: number[];
  output: number[][]; outputBias: number[];
}
export interface SparseFactorMetadata {
  schema: 1; inventory: string; tokens: string; width: number;
  featureOutput: string; primarySha256: string;
}
export interface SparseFactorModel extends SparseFactorMetadata {
  center: number[]; scale: number[]; hidden: number;
  sparseFactor: SparseFactorArm; dense: SparseFactorArm; control: SparseFactorArm;
}
const logadd = (a: number, b: number): number => {
  const m = Math.max(a,b); return Number.isFinite(m) ? m + Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
/** Complete CTC pronunciation, with all boundary paths marginalized. Legal outgoing arcs have
 * equal probability; conditioning the timing prior on completion avoids a length/path-count reward. */
export function sparseFactorBoundaryScore(values: number[][], ids: number[]): number {
  if (!values.length || !ids.length || !ids.every(p => Number.isInteger(p) && p > 0) ||
      !values.every(r => r.length === values[0].length && r.every(Number.isFinite)) ||
      ids.some(p => p >= values[0].length)) return -Infinity;
  const labels = [0, ...ids.flatMap(p => [p,0])];
  const edges = labels.map((p,s) => [s, ...(s+1<labels.length?[s+1]:[]),
    ...(s+2<labels.length && labels[s+2]!==0 && labels[s+2]!==p ? [s+2]:[])]);
  let a = labels.map(()=>-Infinity), prior = a.slice();
  a[0]=values[0][0]; a[1]=values[0][ids[0]]; prior[0]=prior[1]=0;
  for (let t=1;t<values.length;t++) {
    const b=labels.map(()=>-Infinity), q=b.slice();
    for (let s=0;s<labels.length;s++) for (const next of edges[s]) {
      const cost=Math.log(edges[s].length);
      b[next]=logadd(b[next],a[s]-cost); q[next]=logadd(q[next],prior[s]-cost);
    }
    a=b.map((v,s)=>v+values[t][labels[s]]); prior=q;
  }
  const n=labels.length;
  const likelihood=logadd(a[n-1],a[n-2]), timing=logadd(prior[n-1],prior[n-2]);
  return Number.isFinite(likelihood) && Number.isFinite(timing) ? likelihood-timing : -Infinity;
}
/** All arms have identical inference capacity and full-vocabulary CTC topology. */
export function sparseFactorRows(model: SparseFactorModel, arm: SparseFactorArm, raw: number[][]): number[][] {
  return raw.map(row => {
    const x = row.map((v,d) => (v-model.center[d])/model.scale[d]);
    const hidden = arm.encoder.map((w,i) => Math.max(0,w.reduce((sum,v,d) => sum+v*x[d],arm.encoderBias[i])));
    const logits = arm.output.map((w,i) => w.reduce((sum,v,d) => sum+v*hidden[d],arm.outputBias[i]));
    const max = Math.max(...logits), z = max + Math.log(logits.reduce((sum,v) => sum+Math.exp(v-max),0));
    return logits.map(v => v-z);
  });
}
function scores(model: SparseFactorModel, arm: SparseFactorArm, raw: number[][], sequences: number[][]): number[] {
  const rows = sparseFactorRows(model,arm,raw);
  return sequences.map(s => sparseFactorBoundaryScore(rows,s));
}
export function sparseFactorReversal(c: SparseFactorEvidence): boolean {
  return [c.sparseFactorWritten,c.sparseFactorRival,c.controlWritten,c.controlRival,c.denseWritten,c.denseRival].every(Number.isFinite) &&
    c.sparseFactorWritten>c.sparseFactorRival && c.controlWritten<c.controlRival && c.denseWritten<c.denseRival;
}
/** Only retract the exact live occurrence removed by an otherwise identical evidence replay.
 * Repeated words and later graph confirmations may share a spelling or issue id. */
export function sparseFactorVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity = (e: CheckerEvent) => JSON.stringify([e.id, e.timeMs, e.location, e.expected, e.observed]);
  const kept = new Set(reviewed.map(identity));
  const vetoed = new Set(baseline.filter(e => e.errorClass === "wrong_word" && !kept.has(identity(e))).map(identity));
  return live.filter(e => e.errorClass === "wrong_word" && vetoed.has(identity(e)));
}
/** Carry auxiliary judgments without changing any primary token or posterior. */
export function carrySparseFactorEvidence(chunks: PhonemeEvidence[], certificates: SparseFactorEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e,i) => i !== first ? e : {...e, phonemes:e.phonemes.map((p,j) => j ? p : {
    ...p, sparseFactorVetoes:certificates,
  })});
}
export function sparseFactorComparison(model: SparseFactorModel, raw: number[][],
  candidate: SequenceEvidenceCandidate): SparseFactorEvidence | null {
  const sequences = [candidate.written,candidate.rival].map(s => [...s].map(p => model.tokens.indexOf(p)+1));
  if (!Number.isInteger(candidate.fromFrame) || !Number.isInteger(candidate.toFrame) || candidate.fromFrame < 0 ||
      candidate.toFrame-candidate.fromFrame !== raw.length || !raw.length ||
      !raw.every(r => r.length===model.width && r.every(Number.isFinite)) ||
      sequences.some(s => !s.length || s.some(p => p<1))) return null;
  const a = scores(model,model.sparseFactor,raw,sequences), b = scores(model,model.control,raw,sequences),
    d = scores(model,model.dense,raw,sequences);
  const result = {...candidate,sparseFactorWritten:a[0],sparseFactorRival:a[1],controlWritten:b[0],controlRival:b[1],
    denseWritten:d[0],denseRival:d[1]};
  return [a[0],a[1],b[0],b[1],d[0],d[1]].every(Number.isFinite) ? result : null;
}
export function validateSparseFactorModel(value: unknown, manifest: ModelManifest): SparseFactorModel {
  const m = value as SparseFactorModel;
  const vector = (r: unknown, n: number): r is number[] => Array.isArray(r) && r.length===n && r.every(Number.isFinite);
  const matrix = (v: unknown, rows: number, cols: number): v is number[][] => Array.isArray(v) && v.length===rows && v.every(r => vector(r,cols));
  if (!m || m.schema!==1 || m.inventory!==manifest.inventory || m.tokens!==manifest.tokens ||
      m.primarySha256!==manifest.sha256 || !Number.isSafeInteger(m.width) || m.width<1 ||
      m.hidden!==2*m.width || typeof m.featureOutput!=="string" || !m.featureOutput ||
      !vector(m.center,m.width) || !vector(m.scale,m.width) ||
      m.scale.some(v => v<=0)) throw new Error("Invalid sparse-factor model");
  for (const arm of [m.sparseFactor,m.dense,m.control]) {
    if (!arm || !matrix(arm.encoder,m.hidden,m.width) || !vector(arm.encoderBias,m.hidden) ||
        !matrix(arm.output,m.tokens.length+1,m.hidden) || !vector(arm.outputBias,m.tokens.length+1))
      throw new Error("Invalid sparse-factor readout");
  }
  return m;
}
/** Full-session separate streaming state. The final linear input includes activation quantization.
 * All arms retain the same frame clock; primary rows and caches are untouched. */
export async function sparseFactorObservations(backend: InferenceBackend, manifest: ModelManifest, meta: SparseFactorMetadata, audio: Float32Array) {
  const features = new Fbank().push(audio), logProbs: number[][] = [], featuresOut: number[][] = [];
  const total = Math.max(0,Math.floor((Math.floor((features.length-7)/2)+1)/2));
  let states: Record<string,Tensor> = Object.fromEntries(manifest.states.map(s=>[s.name,{type:s.dtype,dims:s.shape,
    data:s.dtype === "int64" ? new BigInt64Array(s.shape.reduce((a,b)=>a*b,1)) : new Float32Array(s.shape.reduce((a,b)=>a*b,1))}]));
  const pad = new Float32Array(manifest.fbank.mels).fill(manifest.fbank.padValue);
  for (let f=0;logProbs.length<total;f+=manifest.shift) {
    const input = new Float32Array(manifest.window*manifest.fbank.mels);
    for(let i=0;i<manifest.window;i++)input.set(features[f+i]??pad,i*manifest.fbank.mels);
    const out=await backend.run({x:{type:"float32",dims:[1,manifest.window,manifest.fbank.mels],data:input},...states});
    states=Object.fromEntries(manifest.states.map(s=>[s.name,out[`new_${s.name}`]]));
    const vocab=manifest.tokens.length+1, n=out.log_probs.data.length/vocab;
    const h=out[meta.featureOutput];
    if(!Number.isSafeInteger(n)||n<=0||!h||h.type!=="float32"||
       h.data.length!==n*meta.width) throw new Error("Invalid final projection geometry");
    for(let t=0;t<n && logProbs.length<total;t++) {
      featuresOut.push(Array.from(h.data.slice(t*meta.width,(t+1)*meta.width),Number));
      logProbs.push(Array.from(out.log_probs.data.slice(t*vocab,(t+1)*vocab),Number));
    }
  }
  return {features:featuresOut,logProbs};
}
export async function addSparseFactorEvidence(scorer: {model:SparseFactorModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:SparseFactorEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if (!scorer||!audio.length) return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>(),cells=insertionCells(chunks);
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e))); checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const features=await sparseFactorObservations(scorer.backend,manifest,scorer.model,audio),certificates:SparseFactorEvidence[]=[];
  for (const c of candidates.values()) {
    const result=c.fromFrame>=0&&c.toFrame<=features.features.length ? sparseFactorComparison(scorer.model,
      features.features.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(result,c); if (result&&sparseFactorReversal(result)) certificates.push(result);
  }
  return carrySparseFactorEvidence(chunks,certificates);
}
