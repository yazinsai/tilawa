import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, FixedEffectPhoneEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface FeatureHead { hidden: number[][]; hiddenBias: number[]; output: number[][]; outputBias: number[] }
export interface FixedEffectModel {
  schema: 1; inventory: string; tokens: string; dimensions: number; primarySha256: string;
  featureOutput: string; projection: number[][]; center: number[]; scale: number[];
  fixedEffect: FeatureHead; control: FeatureHead;
}
const logadd = (a: number, b: number): number => {
  const m = Math.max(a,b); return Number.isFinite(m) ? m + Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
/** Exact complete CTC forward sum, with the same blank/repeat semantics as training. */
export function featureBoundaryScore(values: number[][], ids: number[]): number {
  if (!values.length || !ids.length) return -Infinity;
  const labels = [0, ...ids.flatMap(p => [p,0])];
  const edges = labels.map((p,s) => [s, ...(s+1<labels.length?[s+1]:[]),
    ...(s+2<labels.length && labels[s+2]!==0 && labels[s+2]!==p ? [s+2]:[])]);
  let a = labels.map(()=>-Infinity);
  a[0]=values[0][0]; a[1]=values[0][ids[0]];
  for (let t=1;t<values.length;t++) {
    const b=labels.map(()=>-Infinity);
    for (let s=0;s<labels.length;s++) for (const next of edges[s]) {
      b[next]=logadd(b[next],a[s]);
    }
    a=b.map((v,s)=>v+values[t][labels[s]]);
  }
  const n=labels.length;
  return logadd(a[n-1],a[n-2]);
}
/** Candidate order carries no reference privilege: each complete pronunciation uses the same emissions. */
export function featureScores(head: FeatureHead, raw: number[][], m: FixedEffectModel, sequences: number[][]): number[] {
  const dot=(a:number[],b:number[])=>a.reduce((s,v,i)=>s+v*b[i],0);
  const rows=raw.map(r=> {
    const x=r.map((v,d)=>(v-m.center[d])/m.scale[d]);
    const hidden=head.hidden.map((w,i)=>Math.tanh(dot(w,x)+head.hiddenBias[i]));
    const logits=head.output.map((w,i)=>dot(w,hidden)+head.outputBias[i]);
    const max=Math.max(...logits), z=max+Math.log(logits.reduce((s,v)=>s+Math.exp(v-max),0));
    return logits.map(v=>v-z);
  });
  return sequences.map(s=>featureBoundaryScore(rows,s));
}
export function featureReversal(c: FixedEffectPhoneEvidence): boolean {
  return [c.fixedEffectWritten,c.fixedEffectRival,c.controlWritten,c.controlRival].every(Number.isFinite) &&
    c.fixedEffectWritten>c.fixedEffectRival && c.controlWritten<c.controlRival;
}
export function fixedEffectVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity=(e:CheckerEvent)=>JSON.stringify([e.id,e.timeMs,e.location,e.expected,e.observed]);
  const kept=new Set(reviewed.map(identity));
  const vetoed=new Set(baseline.filter(e=>e.errorClass==="wrong_word"&&!kept.has(identity(e))).map(identity));
  return live.filter(e=>e.errorClass==="wrong_word"&&vetoed.has(identity(e)));
}
export function carryFixedEffectEvidence(chunks: PhonemeEvidence[], certificates: FixedEffectPhoneEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,fixedEffectPhoneVetoes:certificates})});
}
export function featureComparison(m: FixedEffectModel, x: number[][], candidate: SequenceEvidenceCandidate): FixedEffectPhoneEvidence|null {
  const sequences=[candidate.written,candidate.rival].map(s=>[...s].map(p=>m.tokens.indexOf(p)+1));
  if (!x.length||!x.every(r=>r.length===m.dimensions&&r.every(Number.isFinite))||
    sequences.some(s=>!s.length||s.some(p=>p<1))) return null;
  const a=featureScores(m.fixedEffect,x,m,sequences), b=featureScores(m.control,x,m,sequences);
  const c={...candidate,fixedEffectWritten:a[0],fixedEffectRival:a[1],controlWritten:b[0],controlRival:b[1]};
  return [a[0],a[1],b[0],b[1]].every(Number.isFinite)?c:null;
}
export function validateFixedEffectModel(value: unknown, manifest: ModelManifest): FixedEffectModel {
  const m=value as FixedEffectModel;
  const vector=(r:unknown,n:number):r is number[]=>Array.isArray(r)&&r.length===n&&r.every(Number.isFinite);
  if (!m||m.schema!==1||m.inventory!==manifest.inventory||m.tokens!==manifest.tokens||m.primarySha256!==manifest.sha256||
    m.dimensions!==32||typeof m.featureOutput!=="string"||!m.featureOutput||m.projection?.length!==32||!m.projection[0]?.length||
    !m.projection.every(r=>vector(r,m.projection[0].length))||!vector(m.center,32)||!vector(m.scale,32)||m.scale.some(v=>v<=0))
    throw new Error("Invalid fixedEffect feature model");
  for(const h of [m.fixedEffect,m.control]) {
    if (!h||h.hidden?.length!==48||!h.hidden.every(r=>vector(r,32))||!vector(h.hiddenBias,48)||
      h.output?.length!==m.tokens.length+1||!h.output.every(r=>vector(r,48))||!vector(h.outputBias,m.tokens.length+1))
      throw new Error("Invalid pronunciation comparison head");
  }
  return m;
}
const initialStates = (manifest: ModelManifest): Record<string,Tensor> => Object.fromEntries(manifest.states.map(s=>[s.name,{
  type:s.dtype,dims:s.shape,data:s.dtype==="int64"?new BigInt64Array(s.shape.reduce((a,b)=>a*b,1)):
    new Float32Array(s.shape.reduce((a,b)=>a*b,1)),
}]));
function projectOutput(out: Record<string,Tensor>, model: FixedEffectModel, manifest: ModelManifest): number[][] {
  const n=out.log_probs.data.length/(manifest.tokens.length+1);
  const tensor=out[model.featureOutput],width=model.projection[0].length;
  if(!Number.isInteger(n)||n<=0||!tensor||tensor.type!=="float32"||tensor.dims.length!==3||
    tensor.dims[2]!==width||tensor.data.length!==n*width)throw new Error("Invalid pre-head feature geometry");
  return Array.from({length:n},(_,t)=>{
    const row=tensor.data.slice(t*width,(t+1)*width) as Float32Array;
    return model.projection.map(w=>w.reduce((s,v,d)=>s+v*row[d],0));
  });
}
/** Collect features before the primary evidence is delivered. The side graph has
 * independent session caches and receives the listener's exact fbank windows. */
export function createFixedEffectStream(primary: InferenceBackend,
  scorer: {model:FixedEffectModel;backend:InferenceBackend}, manifest:ModelManifest) {
  let states=initialStates(manifest);
  const observations:number[][]=[],certificates:FixedEffectPhoneEvidence[]=[];
  const backend:InferenceBackend={name:primary.name,release:()=>primary.release(),async run(feeds){
    const [original,side]=await Promise.all([primary.run(feeds),scorer.backend.run({x:feeds.x,...states})]);
    if(original.log_probs.data.length!==side.log_probs.data.length)throw new Error("Side encoder frame clock differs from primary");
    states=Object.fromEntries(manifest.states.map(s=>[s.name,side[`new_${s.name}`]]));
    observations.push(...projectOutput(side,scorer.model,manifest));
    // Return the same primary tensors, without substituting side outputs.
    return original;
  }};
  return {backend,observations,certificates,compare(candidate:SequenceEvidenceCandidate):boolean {
    if(candidate.fromFrame<0||candidate.toFrame>observations.length)return true;
    const c=featureComparison(scorer.model,observations.slice(candidate.fromFrame,candidate.toFrame),candidate);
    if(!c||!featureReversal(c))return true;
    certificates.push(c);return false;
  }};
}
/** Separate state instance; exposes pre-head encoder features on the primary stream's native frame clock. */
export async function featureObservations(backend: InferenceBackend, manifest: ModelManifest, model: FixedEffectModel, audio: Float32Array) {
  const features=new Fbank().push(audio), observations:number[][]=[],logProbs:number[][]=[];
  const total=Math.floor((Math.floor((features.length-7)/2)+1)/2);
  let states=initialStates(manifest);
  const pad=new Float32Array(manifest.fbank.mels).fill(manifest.fbank.padValue);
  for(let f=0;observations.length<total;f+=manifest.shift) {
    const input=new Float32Array(manifest.window*manifest.fbank.mels);
    for(let i=0;i<manifest.window;i++)input.set(features[f+i]??pad,i*manifest.fbank.mels);
    const out=await backend.run({x:{type:"float32",dims:[1,manifest.window,manifest.fbank.mels],data:input},...states});
    states=Object.fromEntries(manifest.states.map(s=>[s.name,out[`new_${s.name}`]]));
    const vocab=manifest.tokens.length+1,projected=projectOutput(out,model,manifest);
    for(let t=0;t<projected.length&&observations.length<total;t++) {
      observations.push(projected[t]);
      logProbs.push(Array.from(out.log_probs.data.slice(t*vocab,(t+1)*vocab),Number));
    }
  }
  return {observations,logProbs};
}
export async function addFixedEffectEvidence(scorer: {model:FixedEffectModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:FixedEffectPhoneEvidence|null,request:SequenceEvidenceCandidate)=>void,
  cachedObservations?:number[][]):Promise<PhonemeEvidence[]> {
  if(!scorer||!audio.length)return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for(const e of insertionCells(chunks))checker.push(e,tracker.push(trackerInput(e)));checker.end(tracker.end());
  if(!candidates.size)return chunks;
  const observations=cachedObservations??(await featureObservations(scorer.backend,manifest,scorer.model,audio)).observations;
  const certificates:FixedEffectPhoneEvidence[]=[];
  for(const c of candidates.values()) {
    const result=c.fromFrame>=0&&c.toFrame<=observations.length?
      featureComparison(scorer.model,observations.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(result,c);if(result&&featureReversal(result))certificates.push(result);
  }
  return carryFixedEffectEvidence(chunks,certificates);
}
