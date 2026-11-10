import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, DepthTrajectoryEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface DepthEmission { mean: number[]; precision: number[][]; logDet: number; count: number }
export interface DepthArm { center: number[]; scale: number[]; phones: DepthEmission[] }
export interface DepthModel {
  schema: 1; inventory: string; tokens: string; dimensions: number;
  outputs: string[]; projections: number[][][]; controlProjection: number[][];
  trajectory: DepthArm; control: DepthArm; primarySha256: string;
}
const logadd = (a: number, b: number): number => {
  const m = Math.max(a,b); return Number.isFinite(m) ? m + Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
/** Complete CTC pronunciation, with all boundary paths marginalized. Legal outgoing arcs have
 * equal probability; conditioning the timing prior on completion avoids a length/path-count reward. */
export function depthBoundaryScore(values: number[][], ids: number[]): number {
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
function scores(arm: DepthArm, raw: number[][], sequences: number[][]): number[] {
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
  return sequences.map(s=>depthBoundaryScore(rows,s));
}
export function depthReversal(c: DepthTrajectoryEvidence): boolean {
  return [c.trajectoryWritten,c.trajectoryRival,c.controlWritten,c.controlRival].every(Number.isFinite) &&
    c.trajectoryWritten>c.trajectoryRival && c.controlWritten<c.controlRival;
}
/** Only retract the exact live occurrence removed by an otherwise identical evidence replay.
 * Repeated words and later graph confirmations may share a spelling or issue id. */
export function depthVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity = (e: CheckerEvent) => JSON.stringify([e.id, e.timeMs, e.location, e.expected, e.observed]);
  const kept = new Set(reviewed.map(identity));
  const vetoed = new Set(baseline.filter(e => e.errorClass === "wrong_word" && !kept.has(identity(e))).map(identity));
  return live.filter(e => e.errorClass === "wrong_word" && vetoed.has(identity(e)));
}
/** Carry auxiliary judgments without changing any primary token or posterior. */
export function carryDepthEvidence(chunks: PhonemeEvidence[], certificates: DepthTrajectoryEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e,i) => i !== first ? e : {...e, phonemes:e.phonemes.map((p,j) => j ? p : {
    ...p, depthTrajectoryVetoes:certificates,
  })});
}
export function depthComparison(model: DepthModel, trajectory: number[][], control: number[][],
  candidate: SequenceEvidenceCandidate): DepthTrajectoryEvidence | null {
  const sequences=[candidate.written,candidate.rival].map(s=>[...s].map(p=>model.tokens.indexOf(p)+1));
  const valid=(x:number[][])=>x.length>0&&x.every(r=>r.length===model.dimensions&&r.every(Number.isFinite));
  if (trajectory.length!==control.length || !valid(trajectory)||!valid(control) ||
    sequences.some(s=>!s.length||s.some(p=>p<1||model.trajectory.phones[p].count<=0||model.control.phones[p].count<=0))) return null;
  const a=scores(model.trajectory,trajectory,sequences), b=scores(model.control,control,sequences);
  const result={...candidate,trajectoryWritten:a[0],trajectoryRival:a[1],controlWritten:b[0],controlRival:b[1]};
  return [a[0],a[1],b[0],b[1]].every(Number.isFinite)?result:null;
}
export function validateDepthModel(value: unknown,manifest: ModelManifest): DepthModel {
  const m=value as DepthModel;
  const vector=(r:unknown,n:number):r is number[]=>Array.isArray(r)&&r.length===n&&r.every(Number.isFinite);
  if (!m||m.schema!==1||m.inventory!==manifest.inventory||m.tokens!==manifest.tokens||m.primarySha256!==manifest.sha256||
    m.dimensions!==16||m.outputs.length!==4||new Set(m.outputs).size!==4||m.outputs.some(s=>typeof s!=="string"||!s.length)||
    m.projections.length!==4||m.projections.some(p=>p.length!==4||!p[0]?.length||!p.every(r=>vector(r,p[0].length)))||
    m.controlProjection.length!==16||!m.controlProjection.every(r=>vector(r,m.projections[3][0].length))) throw new Error("Invalid depth trajectory model");
  for (const arm of [m.trajectory,m.control]) {
    if (!vector(arm.center,16)||!vector(arm.scale,16)||arm.scale.some(v=>v<=0)||arm.phones.length!==m.tokens.length+1) throw new Error("Invalid depth emission arm");
    for (const e of arm.phones) {
      if (!vector(e.mean,16)||e.precision.length!==16||!e.precision.every(r=>vector(r,16))||!Number.isFinite(e.logDet)||
        !Number.isFinite(e.count)||e.count<0||e.precision.some((r,i)=>r[i]<=0||r.some((v,j)=>j>i&&v!==0))) throw new Error("Invalid depth emission");
    }
  }
  return m;
}
function project(x: Float32Array, matrix: number[][]): number[] { return matrix.map(row=>row.reduce((s,v,d)=>s+v*x[d],0)); }
/** Frozen depth output on the native streaming grid, from a separate graph/state instance.
 * This also checks the extractor's primary output against the original graph during bundle export. */
export async function depthObservations(backend: InferenceBackend, manifest: ModelManifest, model: DepthModel, audio: Float32Array) {
  const features=new Fbank().push(audio), trajectory:number[][]=[],control:number[][]=[],logProbs:number[][]=[];
  const total=Math.floor((Math.floor((features.length-7)/2)+1)/2);
  let states:Record<string,Tensor>=Object.fromEntries(manifest.states.map(s=>[s.name,{type:s.dtype,dims:s.shape,
    data:s.dtype==="int64"?new BigInt64Array(s.shape.reduce((a,b)=>a*b,1)):new Float32Array(s.shape.reduce((a,b)=>a*b,1))}]));
  const pad=new Float32Array(manifest.fbank.mels).fill(manifest.fbank.padValue);
  for (let f=0;trajectory.length<total;f+=manifest.shift) {
    const input=new Float32Array(manifest.window*manifest.fbank.mels);
    for (let i=0;i<manifest.window;i++) input.set(features[f+i]??pad,i*manifest.fbank.mels);
    const out=await backend.run({x:{type:"float32",dims:[1,manifest.window,manifest.fbank.mels],data:input},...states});
    states=Object.fromEntries(manifest.states.map(s=>[s.name,out[`new_${s.name}`]]));
    const n=out.log_probs.data.length/(manifest.tokens.length+1);
    const levels=model.outputs.map((name,l)=> {
      const tensor=out[name];
      if (!tensor||tensor.type!=="float32") throw new Error("Missing frozen encoder depth output");
      const width=model.projections[l][0].length, rows=tensor.data.length/width;
      if (!Number.isInteger(rows)||rows<n||rows%n!==0) throw new Error("Invalid encoder depth geometry");
      return {data:tensor.data as Float32Array,width,rows};
    });
    for (let t=0;t<n && trajectory.length<total;t++) {
      // Average the architecture's integer subsampling ratio onto the final native clock.
      const native=(v: typeof levels[number]) => {
        const row=new Float32Array(v.width), stride=v.rows/n;
        for (let j=0;j<stride;j++) for (let d=0;d<v.width;d++) row[d]+=v.data[(t*stride+j)*v.width+d]/stride;
        return row;
      };
      const z=levels.map((v,l)=>project(native(v),model.projections[l]));
      trajectory.push(z.flatMap((r,l)=>l?r.map((v,d)=>v-z[l-1][d]):r));
      control.push(project(native(levels[3]),model.controlProjection));
      const vocab=manifest.tokens.length+1;
      logProbs.push(Array.from(out.log_probs.data.slice(t*vocab,(t+1)*vocab),Number));
    }
  }
  return {trajectory,control,logProbs};
}
export async function addDepthEvidence(scorer: {model:DepthModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:DepthTrajectoryEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if (!scorer||!audio.length) return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>(),cells=insertionCells(chunks);
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e))); checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const features=await depthObservations(scorer.backend,manifest,scorer.model,audio),certificates:DepthTrajectoryEvidence[]=[];
  for (const c of candidates.values()) {
    const result=c.fromFrame>=0&&c.toFrame<=features.trajectory.length ? depthComparison(scorer.model,
      features.trajectory.slice(c.fromFrame,c.toFrame),features.control.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(result,c); if (result&&depthReversal(result)) certificates.push(result);
  }
  return carryDepthEvidence(chunks,certificates);
}
