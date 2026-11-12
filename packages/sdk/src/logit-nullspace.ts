import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, LogitNullspaceEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

/** None labels are canonical supervision only when the recording is usable and has no edits/repeats. */
export function eligibleNullspaceTake(r: {kind?:string;split?:string;speaker?:string;exclude?:boolean|string|null;excluded?:boolean|string|null;scored?:boolean;events?:unknown[];non_errors?:unknown[]}, excluded:ReadonlySet<string>):boolean {
  return r.kind === "none" && r.split === "train" && !!r.speaker && !excluded.has(r.speaker) &&
    !r.exclude && !r.excluded && r.scored !== false && !r.events?.length && !r.non_errors?.length;
}

export interface NullspaceEmission { mean: number[]; precision: number[][]; logDet: number; count: number }
export interface NullspaceArm { center: number[]; scale: number[]; phones: NullspaceEmission[] }
export interface NullspaceMetadata {
  schema: 1; inventory: string; tokens: string; dimensions: number; width: number;
  featureOutput: string; logitOutput: string; primarySha256: string;
  rank: number; nullity: number; nullProjection: number[][]; effectiveWeight: number[][];
}
export interface NullspaceModel extends NullspaceMetadata { nullspace: NullspaceArm; control: NullspaceArm }
const logadd = (a: number, b: number): number => {
  const m = Math.max(a,b); return Number.isFinite(m) ? m + Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
/** Complete CTC pronunciation, with all boundary paths marginalized. Legal outgoing arcs have
 * equal probability; conditioning the timing prior on completion avoids a length/path-count reward. */
export function nullspaceBoundaryScore(values: number[][], ids: number[]): number {
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
function scores(arm: NullspaceArm, raw: number[][], sequences: number[][]): number[] {
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
  return sequences.map(s=>nullspaceBoundaryScore(rows,s));
}
export function nullspaceReversal(c: LogitNullspaceEvidence): boolean {
  return [c.nullspaceWritten,c.nullspaceRival,c.controlWritten,c.controlRival].every(Number.isFinite) &&
    c.nullspaceWritten>c.nullspaceRival && c.controlWritten<c.controlRival;
}
/** Only retract the exact live occurrence removed by an otherwise identical evidence replay.
 * Repeated words and later graph confirmations may share a spelling or issue id. */
export function nullspaceVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity = (e: CheckerEvent) => JSON.stringify([e.id, e.timeMs, e.location, e.expected, e.observed]);
  const kept = new Set(reviewed.map(identity));
  const vetoed = new Set(baseline.filter(e => e.errorClass === "wrong_word" && !kept.has(identity(e))).map(identity));
  return live.filter(e => e.errorClass === "wrong_word" && vetoed.has(identity(e)));
}
/** Carry auxiliary judgments without changing any primary token or posterior. */
export function carryNullspaceEvidence(chunks: PhonemeEvidence[], certificates: LogitNullspaceEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e,i) => i !== first ? e : {...e, phonemes:e.phonemes.map((p,j) => j ? p : {
    ...p, logitNullspaceVetoes:certificates,
  })});
}
export function nullspaceComparison(model: NullspaceModel, nullspace: number[][], control: number[][],
  candidate: SequenceEvidenceCandidate): LogitNullspaceEvidence | null {
  const sequences=[candidate.written,candidate.rival].map(s=>[...s].map(p=>model.tokens.indexOf(p)+1));
  const valid=(x:number[][])=>x.length>0&&x.every(r=>r.length===model.dimensions&&r.every(Number.isFinite));
  if (!Number.isInteger(candidate.fromFrame) || candidate.fromFrame < 0 ||
    candidate.toFrame-candidate.fromFrame !== nullspace.length || nullspace.length!==control.length || !valid(nullspace)||!valid(control) ||
    sequences.some(s=>!s.length||s.some(p=>p<1||model.nullspace.phones[p].count<=0||model.control.phones[p].count<=0))) return null;
  const a=scores(model.nullspace,nullspace,sequences), b=scores(model.control,control,sequences);
  const result={...candidate,nullspaceWritten:a[0],nullspaceRival:a[1],controlWritten:b[0],controlRival:b[1]};
  return [a[0],a[1],b[0],b[1]].every(Number.isFinite)?result:null;
}
export function validateNullspaceModel(value: unknown, manifest: ModelManifest): NullspaceModel {
  const m = value as NullspaceModel;
  const vector = (r: unknown, n: number): r is number[] => Array.isArray(r) && r.length === n && r.every(Number.isFinite);
  const matrix = (v: unknown, rows: number, cols: number): v is number[][] => Array.isArray(v) && v.length === rows && v.every(r => vector(r, cols));
  if (!m || m.schema !== 1 || m.inventory !== manifest.inventory || m.tokens !== manifest.tokens ||
      m.primarySha256 !== manifest.sha256 || m.dimensions !== m.tokens.length + 1 ||
      !Number.isSafeInteger(m.width) || m.width <= m.dimensions || !Number.isSafeInteger(m.rank) ||
      m.rank <= 0 || m.rank > m.dimensions || m.nullity !== m.width-m.rank || m.nullity < m.dimensions ||
      !m.featureOutput || !m.logitOutput || !matrix(m.nullProjection,m.dimensions,m.width) ||
      !matrix(m.effectiveWeight,m.width,m.dimensions)) throw new Error("Invalid logit-nullspace model");
  // The retained coordinates are invisible to every deployed logit, including blank.
  for (const row of m.nullProjection) for (let p=0;p<m.dimensions;p++) {
    const leak=row.reduce((sum,v,d)=>sum+v*m.effectiveWeight[d][p],0);
    if (Math.abs(leak)>1e-10) throw new Error("Feature projection leaks primary logits");
  }
  for (let i=0;i<m.dimensions;i++) for (let j=0;j<=i;j++) {
    const dot=m.nullProjection[i].reduce((sum,v,d)=>sum+v*m.nullProjection[j][d],0);
    if (Math.abs(dot-(i===j?1:0))>1e-10) throw new Error("Invalid orthonormal null-space coordinates");
  }
  for (const arm of [m.nullspace,m.control]) {
    if (!arm || !vector(arm.center,m.dimensions) || !vector(arm.scale,m.dimensions) || arm.scale.some(v=>v<=0) ||
        !Array.isArray(arm.phones) || arm.phones.length !== m.dimensions) throw new Error("Invalid nullspace emission arm");
    for (const e of arm.phones) if (!e || !vector(e.mean,m.dimensions) || !matrix(e.precision,m.dimensions,m.dimensions) ||
        !Number.isFinite(e.logDet) || !Number.isFinite(e.count) || e.count < 0 ||
        e.precision.some((r,i)=>r[i]<=0||r.some((v,j)=>j>i&&v!==0))) throw new Error("Invalid nullspace emission");
  }
  return m;
}
function project(x: Float32Array, matrix: number[][]): number[] {
  return matrix.map(row => row.reduce((s,v,d) => s+v*x[d],0));
}
/** Full-session separate streaming state. The final linear input includes activation quantization.
 * Both arms retain the same frame clock; primary rows and caches are untouched. */
export async function nullspaceObservations(backend: InferenceBackend, manifest: ModelManifest, meta: NullspaceMetadata, audio: Float32Array) {
  const features = new Fbank().push(audio), logProbs: number[][] = [], nullspace: number[][] = [], control: number[][] = [];
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
      nullspace.push(project(h.data.slice(t*meta.width,(t+1)*meta.width) as Float32Array,meta.nullProjection));
      control.push(Array.from(z.data.slice(t*vocab,(t+1)*vocab),Number));
      logProbs.push(Array.from(out.log_probs.data.slice(t*vocab,(t+1)*vocab),Number));
    }
  }
  return {nullspace,control,logProbs};
}
export async function addNullspaceEvidence(scorer: {model:NullspaceModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:LogitNullspaceEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if (!scorer||!audio.length) return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>(),cells=insertionCells(chunks);
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e))); checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const features=await nullspaceObservations(scorer.backend,manifest,scorer.model,audio),certificates:LogitNullspaceEvidence[]=[];
  for (const c of candidates.values()) {
    const result=c.fromFrame>=0&&c.toFrame<=features.nullspace.length ? nullspaceComparison(scorer.model,
      features.nullspace.slice(c.fromFrame,c.toFrame),features.control.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(result,c); if (result&&nullspaceReversal(result)) certificates.push(result);
  }
  return carryNullspaceEvidence(chunks,certificates);
}
