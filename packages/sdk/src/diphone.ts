import { collapseHeld, createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, DiphoneEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface FeatureHead { hidden: number[][]; hiddenBias: number[]; output: number[][]; outputBias: number[] }
export interface DiphoneModel {
  schema: 1; inventory: string; tokens: string; dimensions: number; primarySha256: string;
  featureOutput: string; projection: number[][]; center: number[]; scale: number[];
  diphone: FeatureHead; control: FeatureHead;
}
const logadd = (a: number, b: number): number => {
  const m = Math.max(a,b); return Number.isFinite(m) ? m + Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
/** Pair zero is blank; all other labels identify an ordered articulation pair. */
export function diphoneLabels(ids: number[], phones: number): number[] {
  return ids.slice(1).map((b,i)=>1+(ids[i]-1)*phones+b-1);
}
export function consistentDiphones(labels: number[], phones: number): boolean {
  return labels.length>0 && labels.every(p=>Number.isInteger(p)&&p>0&&p<=phones*phones) &&
    labels.slice(1).every((p,i)=>(labels[i]-1)%phones===Math.floor((p-1)/phones));
}
/** A shared middle articulation is part of both ordered transitions. */
export function diphoneBoundaryScore(values: number[][], labels: number[], phones: number): number {
  return consistentDiphones(labels,phones)?featureBoundaryScore(values,labels):-Infinity;
}
/** Complete CTC pronunciation, with all boundary paths marginalized. Legal outgoing arcs have
 * equal probability; conditioning the timing prior on completion avoids a length/path-count reward. */
export function featureBoundaryScore(values: number[][], ids: number[]): number {
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
/** Candidate order carries no reference privilege: each complete pronunciation uses the same emissions. */
export function headRows(head: FeatureHead, raw: number[][], m: DiphoneModel): number[][] {
  const dot=(a:number[],b:number[])=>a.reduce((s,v,i)=>s+v*b[i],0);
  return raw.map((_,t)=> {
    const x=[-1,0,1].flatMap(offset=>raw[Math.max(0,Math.min(raw.length-1,t+offset))]
      .map((v,d)=>(v-m.center[d])/m.scale[d]));
    const hidden=head.hidden.map((w,i)=>Math.tanh(dot(w,x)+head.hiddenBias[i]));
    const logits=head.output.map((w,i)=>dot(w,hidden)+head.outputBias[i]);
    const max=Math.max(...logits), z=max+Math.log(logits.reduce((s,v)=>s+Math.exp(v-max),0));
    return logits.map(v=>v-z);
  });
}
/** The equally sized control sums its redundant outputs by phone identity.
 * It never learns the right partner. Both heads evaluate the full output matrix. */
export function singlePhoneRows(rows: number[][], phones: number): number[][] {
  return rows.map(r=>[r[0],...Array.from({length:phones},(_,a)=>
    r.slice(1+a*phones,1+(a+1)*phones).reduce(logadd,-Infinity))]);
}
export function featureScores(head: FeatureHead, raw: number[][], m: DiphoneModel, sequences: number[][], pairs=false): number[] {
  const rows=headRows(head,raw,m), phones=m.tokens.length;
  if(pairs)return sequences.map(s=>diphoneBoundaryScore(rows,diphoneLabels(s,phones),phones));
  const control=singlePhoneRows(rows,phones);
  return sequences.map(s=>featureBoundaryScore(control,s));
}
export function featureReversal(c: DiphoneEvidence): boolean {
  return [c.diphoneWritten,c.diphoneRival,c.controlWritten,c.controlRival].every(Number.isFinite) &&
    c.diphoneWritten>c.diphoneRival && c.controlWritten<c.controlRival;
}
export function diphoneVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity=(e:CheckerEvent)=>JSON.stringify([e.id,e.timeMs,e.location,e.expected,e.observed]);
  const kept=new Set(reviewed.map(identity));
  const vetoed=new Set(baseline.filter(e=>e.errorClass==="wrong_word"&&!kept.has(identity(e))).map(identity));
  return live.filter(e=>e.errorClass==="wrong_word"&&vetoed.has(identity(e)));
}
export function carryDiphoneEvidence(chunks: PhonemeEvidence[], certificates: DiphoneEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,diphoneVetoes:certificates})});
}
export function featureComparison(m: DiphoneModel, x: number[][], candidate: SequenceEvidenceCandidate): DiphoneEvidence|null {
  const sequences=[candidate.written,candidate.rival].map(s=>collapseHeld([...s]).map(p=>m.tokens.indexOf(p)+1));
  if (!x.length||!x.every(r=>r.length===m.dimensions&&r.every(Number.isFinite))||
    sequences.some(s=>s.length<2||s.some(p=>p<1))) return null;
  const a=featureScores(m.diphone,x,m,sequences,true), b=featureScores(m.control,x,m,sequences);
  const c={...candidate,diphoneWritten:a[0],diphoneRival:a[1],controlWritten:b[0],controlRival:b[1]};
  return [a[0],a[1],b[0],b[1]].every(Number.isFinite)?c:null;
}
export function validateDiphoneModel(value: unknown, manifest: ModelManifest): DiphoneModel {
  const m=value as DiphoneModel;
  const vector=(r:unknown,n:number):r is number[]=>Array.isArray(r)&&r.length===n&&r.every(Number.isFinite);
  if (!m||m.schema!==1||m.inventory!==manifest.inventory||m.tokens!==manifest.tokens||m.primarySha256!==manifest.sha256||
    m.dimensions!==32||typeof m.featureOutput!=="string"||!m.featureOutput||m.projection?.length!==32||!m.projection[0]?.length||
    !m.projection.every(r=>vector(r,m.projection[0].length))||!vector(m.center,32)||!vector(m.scale,32)||m.scale.some(v=>v<=0))
    throw new Error("Invalid diphone feature model");
  for(const h of [m.diphone,m.control]) {
    if (!h||h.hidden?.length!==48||!h.hidden.every(r=>vector(r,96))||!vector(h.hiddenBias,48)||
      h.output?.length!==m.tokens.length*m.tokens.length+1||!h.output.every(r=>vector(r,48))||!vector(h.outputBias,m.tokens.length*m.tokens.length+1))
      throw new Error("Invalid pronunciation comparison head");
  }
  return m;
}
/** Separate state instance; exposes pre-head encoder features on the primary stream's native frame clock. */
export async function featureObservations(backend: InferenceBackend, manifest: ModelManifest, model: DiphoneModel, audio: Float32Array) {
  const features=new Fbank().push(audio), observations:number[][]=[],logProbs:number[][]=[];
  const total=Math.floor((Math.floor((features.length-7)/2)+1)/2);
  let states:Record<string,Tensor>=Object.fromEntries(manifest.states.map(s=>[s.name,{type:s.dtype,dims:s.shape,
    data:s.dtype==="int64"?new BigInt64Array(s.shape.reduce((a,b)=>a*b,1)):new Float32Array(s.shape.reduce((a,b)=>a*b,1))}]));
  const pad=new Float32Array(manifest.fbank.mels).fill(manifest.fbank.padValue);
  for(let f=0;observations.length<total;f+=manifest.shift) {
    const input=new Float32Array(manifest.window*manifest.fbank.mels);
    for(let i=0;i<manifest.window;i++)input.set(features[f+i]??pad,i*manifest.fbank.mels);
    const out=await backend.run({x:{type:"float32",dims:[1,manifest.window,manifest.fbank.mels],data:input},...states});
    states=Object.fromEntries(manifest.states.map(s=>[s.name,out[`new_${s.name}`]]));
    const vocab=manifest.tokens.length+1,n=out.log_probs.data.length/vocab;
    const tensor=out[model.featureOutput],width=model.projection[0].length;
    if(!tensor||tensor.type!=="float32"||tensor.data.length!==n*width)throw new Error("Invalid pre-head feature geometry");
    for(let t=0;t<n&&observations.length<total;t++) {
      const row=tensor.data.slice(t*width,(t+1)*width) as Float32Array;
      observations.push(model.projection.map(w=>w.reduce((s,v,d)=>s+v*row[d],0)));
      logProbs.push(Array.from(out.log_probs.data.slice(t*vocab,(t+1)*vocab),Number));
    }
  }
  return {observations,logProbs};
}
export async function addDiphoneEvidence(scorer: {model:DiphoneModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:DiphoneEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if(!scorer||!audio.length)return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for(const e of insertionCells(chunks))checker.push(e,tracker.push(trackerInput(e)));checker.end(tracker.end());
  if(!candidates.size)return chunks;
  const features=await featureObservations(scorer.backend,manifest,scorer.model,audio),certificates:DiphoneEvidence[]=[];
  for(const c of candidates.values()) {
    const result=c.fromFrame>=0&&c.toFrame<=features.observations.length?
      featureComparison(scorer.model,features.observations.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(result,c);if(result&&featureReversal(result))certificates.push(result);
  }
  return carryDiphoneEvidence(chunks,certificates);
}
