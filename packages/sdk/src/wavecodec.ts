import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, WavecodecEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface CodecEmission { mean: number[]; precision: number[][]; logDet: number; count: number }
export interface CodecArm { center: number[]; scale: number[]; phones: CodecEmission[] }
export interface WavecodecModel {
  schema: 1; inventory: string; tokens: string; dimensions: number;
  output: string; controlProjection: number[][];
  hop: number; patch: number; weights: Record<string, unknown>;
  codec: CodecArm; control: CodecArm; primarySha256: string;
}
const logadd = (a: number, b: number): number => {
  const m = Math.max(a,b); return Number.isFinite(m) ? m + Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
/** Complete CTC pronunciation, with all boundary paths marginalized. Legal outgoing arcs have
 * equal probability; conditioning the timing prior on completion avoids a length/path-count reward. */
export function wavecodecBoundaryScore(values: number[][], ids: number[]): number {
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
function scores(arm: CodecArm, raw: number[][], sequences: number[][]): number[] {
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
  return sequences.map(s=>wavecodecBoundaryScore(rows,s));
}
export function wavecodecReversal(c: WavecodecEvidence): boolean {
  return [c.codecWritten,c.codecRival,c.controlWritten,c.controlRival].every(Number.isFinite) &&
    c.codecWritten>c.codecRival && c.controlWritten<c.controlRival;
}
/** Only retract the exact live occurrence removed by an otherwise identical evidence replay.
 * Repeated words and later graph confirmations may share a spelling or issue id. */
export function wavecodecVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity = (e: CheckerEvent) => JSON.stringify([e.id, e.timeMs, e.location, e.expected, e.observed]);
  const kept = new Set(reviewed.map(identity));
  const vetoed = new Set(baseline.filter(e => e.errorClass === "wrong_word" && !kept.has(identity(e))).map(identity));
  return live.filter(e => e.errorClass === "wrong_word" && vetoed.has(identity(e)));
}
/** Carry auxiliary judgments without changing any primary token or posterior. */
export function carryWavecodecEvidence(chunks: PhonemeEvidence[], certificates: WavecodecEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e,i) => i !== first ? e : {...e, phonemes:e.phonemes.map((p,j) => j ? p : {
    ...p, wavecodecVetoes:certificates,
  })});
}
export function wavecodecComparison(model: WavecodecModel, codec: number[][], control: number[][],
  candidate: SequenceEvidenceCandidate): WavecodecEvidence | null {
  const sequences=[candidate.written,candidate.rival].map(s=>[...s].map(p=>model.tokens.indexOf(p)+1));
  const valid=(x:number[][])=>x.length>0&&x.every(r=>r.length===model.dimensions&&r.every(Number.isFinite));
  if (codec.length!==control.length || !valid(codec)||!valid(control) ||
    sequences.some(s=>!s.length||s.some(p=>p<1||model.codec.phones[p].count<=0||model.control.phones[p].count<=0))) return null;
  const a=scores(model.codec,codec,sequences), b=scores(model.control,control,sequences);
  const result={...candidate,codecWritten:a[0],codecRival:a[1],controlWritten:b[0],controlRival:b[1]};
  return [a[0],a[1],b[0],b[1]].every(Number.isFinite)?result:null;
}
export function validateWavecodecModel(value: unknown, manifest: ModelManifest): WavecodecModel {
  const m=value as WavecodecModel;
  if (!m || m.schema!==1 || m.inventory!==manifest.inventory || m.tokens!==manifest.tokens ||
    m.primarySha256!==manifest.sha256 || m.dimensions!==16 || m.patch!==400 || ![160,320].includes(m.hop) ||
    typeof m.output!=="string" || !m.output.length || m.controlProjection?.length!==16 || !m.controlProjection[0]?.length ||
    !m.controlProjection.every(r=>r.length===m.controlProjection[0].length&&r.every(Number.isFinite)))
    throw new Error("Invalid waveform codec model");
  for (const arm of [m.codec,m.control]) {
    if (arm?.center?.length!==16 || arm.scale?.length!==16 || !arm.scale.every(x=>Number.isFinite(x)&&x>0) ||
      !arm.center.every(Number.isFinite) || arm.phones?.length!==m.tokens.length+1) throw new Error("Invalid codec emission arm");
    for (const e of arm.phones) {
      if (e.mean?.length!==16 || !e.mean.every(Number.isFinite) || e.precision?.length!==16 ||
        !e.precision.every((r,i)=>r.length===16&&r.every(Number.isFinite)&&r[i]>0&&r.every((v,j)=>j<=i||v===0)) ||
        !Number.isFinite(e.logDet) || !Number.isFinite(e.count) || e.count<0) throw new Error("Invalid codec emission");
    }
  }
  validateCodecWeights(m.weights);
  return m;
}
function project(x: Float32Array, matrix: number[][]): number[] { return matrix.map(row=>row.reduce((s,v,d)=>s+v*x[d],0)); }

interface CodecWeights {
  "encoder.0.weight": number[][][]; "encoder.0.bias": number[];
  "encoder.2.weight": number[][][]; "encoder.2.bias": number[];
  codebook: number[][];
}
function validateCodecWeights(raw: Record<string,unknown>): asserts raw is Record<string,unknown> & CodecWeights {
  if (!raw) throw new Error("Missing codec weights");
  const w=raw as unknown as CodecWeights;
  const conv=(kernel:number[][][],bias:number[],input:number,output:number)=>
    kernel?.length===output&&bias?.length===output&&bias.every(Number.isFinite)&&kernel.every(c=>
      c.length===input&&c.every(r=>r.length===8&&r.every(Number.isFinite)));
  if (!conv(w["encoder.0.weight"],w["encoder.0.bias"],1,8)||
    !conv(w["encoder.2.weight"],w["encoder.2.bias"],8,16)||
    ![32,64].includes(w.codebook?.length)||!w.codebook.every(r=>r.length===16&&r.every(Number.isFinite)))
    throw new Error("Invalid codec weights");
}
/** Identical to the frozen CPU reconstruction encoder: conv/tanh, conv/tanh,
 * temporal mean, nearest discrete code. No phone/text input enters encoding. */
export function codecFeatures(model: Pick<WavecodecModel,"weights"|"hop"|"patch">, audio: Float32Array): number[][] {
  validateCodecWeights(model.weights);
  const w=model.weights;
  const conv=(input:number[][],kernel:number[][][],bias:number[])=> {
    const n=Math.floor((input[0].length-8)/4)+1;
    return kernel.map((c,o)=>Array.from({length:n},(_,t)=> {
      let z=bias[o];
      for (let i=0;i<c.length;i++) for (let k=0;k<8;k++) z+=c[i][k]*input[i][t*4+k];
      return Math.tanh(z);
    }));
  const result:number[][]=[];
  for (let start=0;start+model.patch<=audio.length;start+=model.hop) {
    const input=[Array.from(audio.subarray(start,start+model.patch),v=>v*10)];
    const first=conv(input,w["encoder.0.weight"],w["encoder.0.bias"]);
    const second=conv(first,w["encoder.2.weight"],w["encoder.2.bias"]);
    const z=second.map(r=>r.reduce((s,v)=>s+v,0)/r.length);
    let best=0,distance=Infinity;
    for (let c=0;c<w.codebook.length;c++) {
      const d=z.reduce((s,v,i)=>s+(v-w.codebook[c][i])**2,0);
      if (d<distance) {best=c;distance=d;}
    }
    result.push(w.codebook[best].slice());
  }
  return result;
}
/** Frozen shipped encoder output on the native streaming grid, from a separate graph/state instance.
 * This also checks the extractor's primary output against the original graph during bundle export. */
export async function controlObservations(backend: InferenceBackend, manifest: ModelManifest,
  model: Pick<WavecodecModel,"output"|"controlProjection">, audio: Float32Array) {
  const features=new Fbank().push(audio), native:number[][]=[];
  const total=Math.floor((Math.floor((features.length-7)/2)+1)/2);
  let states:Record<string,Tensor>=Object.fromEntries(manifest.states.map(s=>[s.name,{type:s.dtype,dims:s.shape,
    data:s.dtype==="int64"?new BigInt64Array(s.shape.reduce((a,b)=>a*b,1)):new Float32Array(s.shape.reduce((a,b)=>a*b,1))}]));
  const pad=new Float32Array(manifest.fbank.mels).fill(manifest.fbank.padValue);
  for (let f=0;native.length<total;f+=manifest.shift) {
    const input=new Float32Array(manifest.window*manifest.fbank.mels);
    for (let i=0;i<manifest.window;i++) input.set(features[f+i]??pad,i*manifest.fbank.mels);
    const out=await backend.run({x:{type:"float32",dims:[1,manifest.window,manifest.fbank.mels],data:input},...states});
    states=Object.fromEntries(manifest.states.map(s=>[s.name,out[`new_${s.name}`]]));
    const n=out.log_probs.data.length/(manifest.tokens.length+1), tensor=out[model.output];
    const width=model.controlProjection[0].length;
    if (!tensor||tensor.type!=="float32"||tensor.data.length!==n*width) throw new Error("Invalid frozen encoder geometry");
    for (let t=0;t<n && native.length<total;t++) native.push(project((tensor.data as Float32Array).subarray(t*width,(t+1)*width),model.controlProjection));
  }
  return native;
}
export async function wavecodecObservations(backend: InferenceBackend, manifest: ModelManifest, model: WavecodecModel, audio: Float32Array) {
  const native=await controlObservations(backend,manifest,model,audio);
  const codes=codecFeatures(model,audio), n=Math.min(codes.length,Math.floor(native.length*640/model.hop));
  return {codec:codes.slice(0,n),control:Array.from({length:n},(_,t)=>native[Math.floor(t*model.hop/640)])};
}
export async function addWavecodecEvidence(scorer: {model:WavecodecModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:WavecodecEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if (!scorer||!audio.length) return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>(),cells=insertionCells(chunks);
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e))); checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const features=await wavecodecObservations(scorer.backend,manifest,scorer.model,audio),certificates:WavecodecEvidence[]=[];
  for (const c of candidates.values()) {
    const result=c.fromFrame>=0&&c.toFrame*640/scorer.model.hop<=features.codec.length ? wavecodecComparison(scorer.model,
      features.codec.slice(c.fromFrame*640/scorer.model.hop,c.toFrame*640/scorer.model.hop),features.control.slice(c.fromFrame*640/scorer.model.hop,c.toFrame*640/scorer.model.hop),c):null;
    inspect?.(result,c); if (result&&wavecodecReversal(result)) certificates.push(result);
  }
  return carryWavecodecEvidence(chunks,certificates);
}
