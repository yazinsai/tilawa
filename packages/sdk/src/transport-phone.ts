import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, ConditionalTransportEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { Resampler } from "./audio.js";
import { insertionCells } from "./seam-insertion.js";

export interface FeatureHead { hidden: number[][]; hiddenBias: number[]; output: number[][]; outputBias: number[] }
export interface TransportModel {
  schema: 1; inventory: string; tokens: string; dimensions: number; primarySha256: string;
  featureOutput: string; projection: number[][]; center: number[]; scale: number[];
  conditional: FeatureHead; control: FeatureHead; readout: FeatureHead; temperature: number;
}
const logadd = (a: number, b: number): number => {
  const m = Math.max(a,b); return Number.isFinite(m) ? m + Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
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
export function featureScores(adapter: FeatureHead, raw: number[][], m: TransportModel, sequences: number[][]): number[] {
  const dot=(a:number[],b:number[])=>a.reduce((s,v,i)=>s+v*b[i],0);
  const apply=(head:FeatureHead,x:number[])=> {
    const h=head.hidden.map((w,i)=>Math.tanh(dot(w,x)+head.hiddenBias[i]));
    return head.output.map((w,i)=>dot(w,h)+head.outputBias[i]);
  };
  const rows=raw.map(r=> {
    const x=r.map((v,d)=>(v-m.center[d])/m.scale[d]);
    const delta=apply(adapter,x);
    // One shared audio-only adapter, followed by the identical frozen studio scorer.
    const logits=apply(m.readout,x.map((v,i)=>v+delta[i])).map(v=>v/m.temperature);
    const max=Math.max(...logits), z=max+Math.log(logits.reduce((s,v)=>s+Math.exp(v-max),0));
    return logits.map(v=>v-z);
  });
  return sequences.map(s=>featureBoundaryScore(rows,s));
}
export function featureReversal(c: ConditionalTransportEvidence): boolean {
  return [c.transportWritten,c.transportRival,c.controlWritten,c.controlRival].every(Number.isFinite) &&
    c.transportWritten>c.transportRival && c.controlWritten<c.controlRival;
}
export function carryTransportEvidence(chunks: PhonemeEvidence[], certificates: ConditionalTransportEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,conditionalTransportVetoes:certificates})});
}
export function featureComparison(m: TransportModel, x: number[][], candidate: SequenceEvidenceCandidate): ConditionalTransportEvidence|null {
  const sequences=[candidate.written,candidate.rival].map(s=>[...s].map(p=>m.tokens.indexOf(p)+1));
  if (!x.length||!x.every(r=>r.length===m.dimensions&&r.every(Number.isFinite))||
    sequences.some(s=>!s.length||s.some(p=>p<1))) return null;
  const a=featureScores(m.conditional,x,m,sequences), b=featureScores(m.control,x,m,sequences);
  const c={...candidate,transportWritten:a[0],transportRival:a[1],controlWritten:b[0],controlRival:b[1]};
  return [a[0],a[1],b[0],b[1]].every(Number.isFinite)?c:null;
}
export function validateTransportModel(value: unknown, manifest: ModelManifest): TransportModel {
  const m=value as TransportModel;
  const vector=(r:unknown,n:number):r is number[]=>Array.isArray(r)&&r.length===n&&r.every(Number.isFinite);
  if (!m||m.schema!==1||m.inventory!==manifest.inventory||m.tokens!==manifest.tokens||m.primarySha256!==manifest.sha256||
    m.dimensions!==32||typeof m.featureOutput!=="string"||!m.featureOutput||m.projection?.length!==32||!m.projection[0]?.length||
    !m.projection.every(r=>vector(r,m.projection[0].length))||!vector(m.center,32)||!vector(m.scale,32)||m.scale.some(v=>v<=0)||
    !Number.isFinite(m.temperature)||m.temperature<=0)
    throw new Error("Invalid transport feature model");
  for(const [h,width,outputs] of [[m.conditional,16,32],[m.control,16,32],[m.readout,48,m.tokens.length+1]] as const) {
    if (!h||h.hidden?.length!==width||!h.hidden.every(r=>vector(r,32))||!vector(h.hiddenBias,width)||
      h.output?.length!==outputs||!h.output.every(r=>vector(r,width))||!vector(h.outputBias,outputs))
      throw new Error("Invalid pronunciation comparison head");
  }
  return m;
}
/** Separate state instance; exposes pre-head encoder features before the primary checker consumes that chunk. */
export function createFeatureStream(backend: InferenceBackend, manifest: ModelManifest, model: TransportModel) {
  const fbank=new Fbank(), observations:number[][]=[],logProbs:number[][]=[];
  let features:Float32Array[]=[],consumed=0,totalFrames=0,resampler:Resampler|null=null;
  let states:Record<string,Tensor>=Object.fromEntries(manifest.states.map(s=>[s.name,{type:s.dtype,dims:s.shape,
    data:s.dtype==="int64"?new BigInt64Array(s.shape.reduce((a,b)=>a*b,1)):new Float32Array(s.shape.reduce((a,b)=>a*b,1))}]));
  const pad=new Float32Array(manifest.fbank.mels).fill(manifest.fbank.padValue);
  const step=async(keep:number)=> {
    const input=new Float32Array(manifest.window*manifest.fbank.mels);
    for(let i=0;i<manifest.window;i++)input.set(features[consumed+i]??pad,i*manifest.fbank.mels);
    const out=await backend.run({x:{type:"float32",dims:[1,manifest.window,manifest.fbank.mels],data:input},...states});
    states=Object.fromEntries(manifest.states.map(s=>[s.name,out[`new_${s.name}`]]));
    const vocab=manifest.tokens.length+1,n=out.log_probs.data.length/vocab;
    const tensor=out[model.featureOutput],width=model.projection[0].length;
    if(!tensor||tensor.type!=="float32"||tensor.data.length!==n*width)throw new Error("Invalid pre-head feature geometry");
    for(let t=0;t<Math.min(n,keep);t++) {
      const row=tensor.data.slice(t*width,(t+1)*width) as Float32Array;
      observations.push(model.projection.map(w=>w.reduce((s,v,d)=>s+v*row[d],0)));
      logProbs.push(Array.from(out.log_probs.data.slice(t*vocab,(t+1)*vocab),Number));
    }
    consumed+=manifest.shift;
    if(consumed>4*manifest.window){features=features.slice(consumed);consumed=0;}
  };
  const append=(samples:Float32Array)=>{const rows=fbank.push(samples);features.push(...rows);totalFrames+=rows.length;};
  return {observations,logProbs,
    async push(samples:Float32Array,sampleRate:number) {
      if(sampleRate!==manifest.sampleRate){
        if(!resampler||resampler.from!==sampleRate)resampler=new Resampler(sampleRate,manifest.sampleRate);
        samples=resampler.push(samples);
      }
      append(samples);
      while(features.length-consumed>=manifest.window)await step(Infinity);
    },
    async end() {
      if(resampler)append(resampler.flush());
      const total=Math.floor((Math.floor((totalFrames-7)/2)+1)/2);
      while(observations.length<total)await step(total-observations.length);
    },
  };
}
export async function featureObservations(backend: InferenceBackend, manifest: ModelManifest, model: TransportModel, audio: Float32Array) {
  const stream=createFeatureStream(backend,manifest,model);
  await stream.push(audio,manifest.sampleRate);await stream.end();
  return {observations:stream.observations,logProbs:stream.logProbs};
}
export async function addTransportEvidence(scorer: {model:TransportModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:ConditionalTransportEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if(!scorer||!audio.length)return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for(const e of insertionCells(chunks))checker.push(e,tracker.push(trackerInput(e)));checker.end(tracker.end());
  if(!candidates.size)return chunks;
  const features=await featureObservations(scorer.backend,manifest,scorer.model,audio),certificates:ConditionalTransportEvidence[]=[];
  for(const c of candidates.values()) {
    const result=c.fromFrame>=0&&c.toFrame<=features.observations.length?
      featureComparison(scorer.model,features.observations.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(result,c);if(result&&featureReversal(result))certificates.push(result);
  }
  return carryTransportEvidence(chunks,certificates);
}
