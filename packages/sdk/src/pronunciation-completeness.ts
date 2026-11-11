import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, CompletenessEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface CompletenessModel {
  schema: 1; inventory: string; tokens: string; dimensions: number; primarySha256: string;
  featureOutput: string; projection: number[][]; center: number[]; scale: number[]; cutoff: number;
}
export function validateCompletenessModel(value: unknown, manifest: ModelManifest): CompletenessModel {
  const m=value as CompletenessModel;
  const vector=(r:unknown,n:number):r is number[]=>Array.isArray(r)&&r.length===n&&r.every(Number.isFinite);
  if(!m||m.schema!==1||m.inventory!==manifest.inventory||m.tokens!==manifest.tokens||m.primarySha256!==manifest.sha256||
    m.dimensions!==32||typeof m.featureOutput!=="string"||!m.featureOutput||m.projection?.length!==32||
    !m.projection[0]?.length||!m.projection.every(r=>vector(r,m.projection[0].length))||
    !vector(m.center,32)||!vector(m.scale,32)||m.scale.some(v=>v<=0)||
    !Number.isFinite(m.cutoff)||m.cutoff<=0||m.cutoff>0.5)throw new Error("Invalid pronunciation completeness model");
  return m;
}
/** Frozen pre-head observations, with separate state and the primary native frame clock. */
export async function completenessFeatures(backend:InferenceBackend,manifest:ModelManifest,model:CompletenessModel,audio:Float32Array) {
  const features=new Fbank().push(audio),observations:number[][]=[],logProbs:number[][]=[];
  const total=Math.max(0,Math.floor((Math.floor((features.length-7)/2)+1)/2));
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
    if(!tensor||tensor.type!=="float32"||tensor.data.length!==n*width)throw new Error("Invalid frozen feature geometry");
    for(let t=0;t<n&&observations.length<total;t++) {
      const row=tensor.data.slice(t*width,(t+1)*width) as Float32Array;
      observations.push(model.projection.map(w=>w.reduce((s,v,d)=>s+v*row[d],0)));
      logProbs.push(Array.from(out.log_probs.data.slice(t*vocab,(t+1)*vocab),Number));
    }
  }
  return {observations,logProbs};
}
export function completenessRejects(probability:number,cutoff:number):boolean {
  return Number.isFinite(probability)&&probability>=0&&Number.isFinite(cutoff)&&cutoff>0&&cutoff<=0.5&&probability<cutoff;
}
export function carryCompletenessEvidence(chunks:PhonemeEvidence[],certificates:CompletenessEvidence[]):PhonemeEvidence[] {
  if(!certificates.length)return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,completenessVetoes:certificates})});
}
/** The frozen head judges the entire already accused sequence, with no expected-word prompt.
 * Boundary uncertainty is integrated over equal-weight contiguous edge crops inside the exact requested window. */
export async function completenessProbability(head:InferenceBackend,m:CompletenessModel,x:number[][],pronunciation:string):Promise<number|null> {
  const ids=[...pronunciation].map(p=>m.tokens.indexOf(p)+1);
  if(!x.length||!ids.length||ids.some(p=>p<1)||!x.every(r=>r.length===32&&r.every(Number.isFinite)))return null;
  const boundaries=[[0,x.length],[Math.min(1,x.length-1),x.length],[0,Math.max(1,x.length-1)]];
  const probabilities=[];
  for(const [a,b] of boundaries) {
    const crop=x.slice(a,b), data=Float32Array.from(crop.flatMap(r=>r.map((v,d)=>(v-m.center[d])/m.scale[d])));
    const out=await head.run({features:{type:"float32",dims:[1,crop.length,32],data},
      pronunciation:{type:"int64",dims:[1,ids.length],data:BigInt64Array.from(ids,BigInt)}});
    const p=Number(out.complete_probability?.data[0]);
    if(!Number.isFinite(p)||p<0||p>1)throw new Error("Invalid completeness head output");
    probabilities.push(p);
  }
  return probabilities.reduce((a,b)=>a+b,0)/probabilities.length;
}
export async function addCompletenessEvidence(scorer:{model:CompletenessModel;backend:InferenceBackend;head:InferenceBackend}|undefined,
 manifest:ModelManifest,audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,
 options:CheckerOptions={},mode:"free"|"memorization"="free",inspect?:(c:CompletenessEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if(!scorer||!audio.length)return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for(const e of insertionCells(chunks))checker.push(e,tracker.push(trackerInput(e)));checker.end(tracker.end());
  if(!candidates.size)return chunks;
  const features=await completenessFeatures(scorer.backend,manifest,scorer.model,audio),certificates:CompletenessEvidence[]=[];
  for(const c of candidates.values()) {
    const p=c.fromFrame>=0&&c.toFrame<=features.observations.length?
      await completenessProbability(scorer.head,scorer.model,features.observations.slice(c.fromFrame,c.toFrame),c.rival):null;
    const result=p===null?null:{...c,completeProbability:p,cutoff:scorer.model.cutoff};
    inspect?.(result,c);if(result&&completenessRejects(result.completeProbability,result.cutoff))certificates.push(result);
  }
  return carryCompletenessEvidence(chunks,certificates);
}
