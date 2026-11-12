import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, SharedAcousticsEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

/** None labels are canonical supervision only when the recording is usable and has no edits/repeats. */
export function eligibleSharedAcousticsTake(r: {kind?:string;split?:string;speaker?:string;exclude?:boolean|string|null;excluded?:boolean|string|null;scored?:boolean;events?:unknown[];non_errors?:unknown[]}, excluded:ReadonlySet<string>):boolean {
  return r.kind === "none" && r.split === "train" && !!r.speaker && !excluded.has(r.speaker) &&
    !r.exclude && !r.excluded && r.scored !== false && !r.events?.length && !r.non_errors?.length;
}

export interface SharedAcousticsEmission { mean: number[]; precision: number[][]; logDet: number; count: number }
export interface SharedAcousticsArm { center: number[]; scale: number[]; phones: SharedAcousticsEmission[] }
export interface SharedAcousticsMetadata {
  schema: 1; inventory: string; tokens: string; dimensions: number; width: number;
  featureOutput: string; logitOutput: string; primarySha256: string;
  featureCenter: number[]; sharedProjection: number[][]; controlProjection: number[][];
  correlations: number[]; covarianceRidge: number; trainingRecordings: number;
}
export interface SharedAcousticsModel extends SharedAcousticsMetadata { shared: SharedAcousticsArm; control: SharedAcousticsArm }
const logadd = (a: number, b: number): number => {
  const m = Math.max(a,b); return Number.isFinite(m) ? m + Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
/** Complete CTC pronunciation, with all boundary paths marginalized. Legal outgoing arcs have
 * equal probability; conditioning the timing prior on completion avoids a length/path-count reward. */
export function sharedBoundaryScore(values: number[][], ids: number[]): number {
  if (!values.length || !ids.length) return -Infinity;
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
function scores(arm: SharedAcousticsArm, raw: number[][], sequences: number[][]): number[] {
  const x=raw.map(r=>r.map((v,d)=>(v-arm.center[d])/arm.scale[d]));
  const used=[...new Set([0,...sequences.flat()])];
  const rows=x.map(r=> {
    const values=arm.phones.map(()=>-Infinity);
    for (const p of used) {
      const e=arm.phones[p]; let q=0;
      for (const row of e.precision) { let z=0; for (let d=0;d<r.length;d++) z+=row[d]*(r[d]-e.mean[d]); q+=z*z; }
      values[p]=-.5*(r.length*Math.log(2*Math.PI)+e.logDet+q);
    }
    return values;
  });
  return sequences.map(s=>sharedBoundaryScore(rows,s));
}
export function sharedReversal(c: SharedAcousticsEvidence): boolean {
  return [c.sharedWritten,c.sharedRival,c.controlWritten,c.controlRival].every(Number.isFinite) &&
    c.sharedWritten>c.sharedRival && c.controlWritten<c.controlRival;
}
/** Only retract the exact live occurrence removed by an otherwise identical evidence replay.
 * Repeated words and later graph confirmations may share a spelling or issue id. */
export function sharedVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity = (e: CheckerEvent) => JSON.stringify([e.id, e.timeMs, e.location, e.expected, e.observed]);
  const kept = new Set(reviewed.map(identity));
  const vetoed = new Set(baseline.filter(e => e.errorClass === "wrong_word" && !kept.has(identity(e))).map(identity));
  return live.filter(e => e.errorClass === "wrong_word" && vetoed.has(identity(e)));
}
/** Carry auxiliary judgments without changing any primary token or posterior. */
export function carrySharedAcousticsEvidence(chunks: PhonemeEvidence[], certificates: SharedAcousticsEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e,i) => i !== first ? e : {...e, phonemes:e.phonemes.map((p,j) => j ? p : {
    ...p, sharedAcousticsVetoes:certificates,
  })});
}
export function sharedComparison(model: SharedAcousticsModel, shared: number[][], control: number[][],
  candidate: SequenceEvidenceCandidate): SharedAcousticsEvidence | null {
  const sequences=[candidate.written,candidate.rival].map(s=>[...s].map(p=>model.tokens.indexOf(p)+1));
  const valid=(x:number[][])=>x.length>0&&x.every(r=>r.length===model.dimensions&&r.every(Number.isFinite));
  if (!Number.isInteger(candidate.fromFrame) || candidate.fromFrame < 0 ||
    candidate.toFrame-candidate.fromFrame !== shared.length || shared.length!==control.length || !valid(shared)||!valid(control) ||
    sequences.some(s=>!s.length||s.some(p=>p<1||model.shared.phones[p].count<=0||model.control.phones[p].count<=0))) return null;
  const a=scores(model.shared,shared,sequences), b=scores(model.control,control,sequences);
  const result={...candidate,sharedWritten:a[0],sharedRival:a[1],controlWritten:b[0],controlRival:b[1]};
  return [a[0],a[1],b[0],b[1]].every(Number.isFinite)?result:null;
}
export function validateSharedAcousticsModel(value: unknown, manifest: ModelManifest): SharedAcousticsModel {
  const m = value as SharedAcousticsModel;
  const vector = (r: unknown, n: number): r is number[] => Array.isArray(r) && r.length === n && r.every(Number.isFinite);
  const matrix = (v: unknown, rows: number, cols: number): v is number[][] => Array.isArray(v) && v.length === rows && v.every(r => vector(r, cols));
  if (!m || m.schema !== 1 || m.inventory !== manifest.inventory || m.tokens !== manifest.tokens ||
      m.primarySha256 !== manifest.sha256 || !Number.isSafeInteger(m.dimensions) || m.dimensions <= 0 ||
      !Number.isSafeInteger(m.width) || m.width < m.dimensions ||
      !m.featureOutput || !m.logitOutput || !vector(m.featureCenter,m.width) ||
      !matrix(m.sharedProjection,m.dimensions,m.width) || !matrix(m.controlProjection,m.dimensions,m.width) ||
      !vector(m.correlations,m.dimensions) || m.correlations.some(v=>v<=0||v>1+1e-6) ||
      !Number.isFinite(m.covarianceRidge) || m.covarianceRidge<=0 ||
      !Number.isSafeInteger(m.trainingRecordings) || m.trainingRecordings<3)
    throw new Error("Invalid shared-acoustics model");
  for (const arm of [m.shared,m.control]) {
    if (!arm || !vector(arm.center,m.dimensions) || !vector(arm.scale,m.dimensions) || arm.scale.some(v=>v<=0) ||
        !Array.isArray(arm.phones) || arm.phones.length !== m.tokens.length + 1) throw new Error("Invalid shared emission arm");
    for (const e of arm.phones) if (!e || !vector(e.mean,m.dimensions) || !matrix(e.precision,m.dimensions,m.dimensions) ||
        !Number.isFinite(e.logDet) || !Number.isFinite(e.count) || e.count < 0 ||
        e.precision.some((r,i)=>r[i]<=0||r.some((v,j)=>j>i&&v!==0))) throw new Error("Invalid shared emission");
  }
  return m;
}
function project(x: Float32Array, matrix: number[][], center: number[]): number[] {
  return matrix.map(row => row.reduce((s,v,d) => s+v*(x[d]-center[d]),0));
}
/** Full-session separate streaming state. The final linear input includes activation quantization.
 * Both arms retain the same frame clock; primary rows and caches are untouched. */
export async function encoderObservations(backend: InferenceBackend, manifest: ModelManifest, meta: Pick<SharedAcousticsMetadata,"featureOutput"|"logitOutput"|"width">, audio: Float32Array) {
  const features = new Fbank().push(audio), logProbs: number[][] = [], encoder: number[][] = [];
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
    const h=out[meta.featureOutput],z=out[meta.logitOutput];
    if(!Number.isSafeInteger(n)||n<=0||!h||!z||h.type!=="float32"||z.type!=="float32"||
       h.data.length!==n*meta.width||z.data.length!==n*vocab) throw new Error("Invalid final projection geometry");
    for(let t=0;t<n && logProbs.length<total;t++) {
      encoder.push(Array.from(h.data.slice(t*meta.width,(t+1)*meta.width),Number));
      logProbs.push(Array.from(out.log_probs.data.slice(t*vocab,(t+1)*vocab),Number));
    }
  }
  return {encoder,logProbs};
}
export async function sharedObservations(backend: InferenceBackend, manifest: ModelManifest, meta: SharedAcousticsMetadata, audio: Float32Array) {
  const {encoder,logProbs}=await encoderObservations(backend,manifest,meta,audio);
  return {shared:encoder.map(r=>project(Float32Array.from(r),meta.sharedProjection,meta.featureCenter)),
    control:encoder.map(r=>project(Float32Array.from(r),meta.controlProjection,meta.featureCenter)),logProbs};
}
export async function addSharedAcousticsEvidence(scorer: {model:SharedAcousticsModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:SharedAcousticsEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if (!scorer||!audio.length) return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>(),cells=insertionCells(chunks);
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e))); checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const features=await sharedObservations(scorer.backend,manifest,scorer.model,audio),certificates:SharedAcousticsEvidence[]=[];
  for (const c of candidates.values()) {
    const result=c.fromFrame>=0&&c.toFrame<=features.shared.length ? sharedComparison(scorer.model,
      features.shared.slice(c.fromFrame,c.toFrame),features.control.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(result,c); if (result&&sharedReversal(result)) certificates.push(result);
  }
  return carrySharedAcousticsEvidence(chunks,certificates);
}
