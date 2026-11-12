import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, LogitUncertaintyEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface FeatureHead { hidden: number[][]; hiddenBias: number[]; output: number[][]; outputBias: number[] }
export interface LogitUncertaintyModel {
  schema: 1; inventory: string; tokens: string; dimensions: number; primarySha256: string;
  featureOutput: string; projection: number[][]; center: number[]; scale: number[];
  rank: 1 | 2; extent: number; quadrature: 3 | 5;
  /** Output channels are mean[V], followed by rank loading vectors[V]. */
  uncertainty: FeatureHead; control: FeatureHead;
}
const logadd = (a: number, b: number): number => {
  const m = Math.max(a,b); return Number.isFinite(m) ? m + Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
const normalize = (logits: number[]): number[] => {
  const max = Math.max(...logits);
  const z = max + Math.log(logits.reduce((s,v) => s + Math.exp(v-max), 0));
  return logits.map(v => v-z);
};
/** Standard complete CTC marginal: final phone or trailing blank must be reached. */
export function featureBoundaryScore(values: number[][], ids: number[]): number {
  if (!values.length || !ids.length) return -Infinity;
  const labels = [0, ...ids.flatMap(p => [p,0])];
  let a = labels.map(()=>-Infinity);
  a[0]=values[0][0]; a[1]=values[0][ids[0]];
  for (let t=1;t<values.length;t++) {
    a = advance(a, values[t], labels);
  }
  const n=labels.length;
  return logadd(a[n-1],a[n-2]);
}
function advance(a: number[], emissions: number[], labels: number[]): number[] {
  return labels.map((p,s) => {
    let v = s ? logadd(a[s],a[s-1]) : a[s];
    if (s>1 && p!==0 && p!==labels[s-2]) v=logadd(v,a[s-2]);
    return v+emissions[p];
  });
}
/** Gauss-Hermite product rule for standard normal latent variables. */
export function normalQuadrature(rank: number, order: 3 | 5): {z:number[];weight:number}[] {
  const root=Math.sqrt(10);
  const nodes=order===3 ? [-Math.sqrt(3),0,Math.sqrt(3)] :
    [-Math.sqrt(5+root),-Math.sqrt(5-root),0,Math.sqrt(5-root),Math.sqrt(5+root)];
  const weights=order===3 ? [1/6,2/3,1/6] : [(7-2*root)/60,(7+2*root)/60,8/15,(7+2*root)/60,(7-2*root)/60];
  let points=[{z:[] as number[],weight:1}];
  for(let d=0;d<rank;d++)points=points.flatMap(p=>nodes.map((v,i)=>({z:[...p.z,v],weight:p.weight*weights[i]})));
  return points;
}
/** Marginalize a shared latent only after each entire block has traversed the CTC states.
 * Blocks are independent, so carrying the integrated boundary alpha is exact for the product rule.
 * Averaging frame emissions first would incorrectly make nearby perturbations independent. */
export function integratedBoundaryScore(mean: number[][], loadings: number[][][], ids: number[],
  extent: number, fromFrame: number, order: 3 | 5): number {
  if(!mean.length || !ids.length) return -Infinity;
  const labels=[0,...ids.flatMap(p=>[p,0])], points=normalQuadrature(loadings[0].length,order);
  let a=labels.map(()=>-Infinity); a[0]=0;
  for(let start=0;start<mean.length;) {
    const stop=Math.min(mean.length,start+extent-(fromFrame+start)%extent);
    const paths=points.map(({z,weight})=> {
      let local=a.slice();
      for(let t=start;t<stop;t++) {
        const logits=mean[t].map((v,p)=>v+z.reduce((s,n,d)=>s+n*loadings[t][d][p],0));
        local=advance(local,normalize(logits),labels);
      }
      return local.map(v=>v+Math.log(weight));
    });
    a=labels.map((_,s)=>paths.reduce((v,row)=>logadd(v,row[s]),-Infinity));
    start=stop;
  }
  return logadd(a[a.length-1],a[a.length-2]);
}
/** Both hypotheses share emissions and absolute block boundaries. */
export function featureScores(head: FeatureHead, raw: number[][], m: LogitUncertaintyModel, sequences: number[][],
  integrated: boolean, fromFrame: number): number[] {
  const dot=(a:number[],b:number[])=>a.reduce((s,v,i)=>s+v*b[i],0);
  const rows=raw.map(r=> {
    const x=r.map((v,d)=>(v-m.center[d])/m.scale[d]);
    const hidden=head.hidden.map((w,i)=>Math.tanh(dot(w,x)+head.hiddenBias[i]));
    const logits=head.output.map((w,i)=>dot(w,hidden)+head.outputBias[i]);
    return logits;
  });
  const vocab=m.tokens.length+1,mean=rows.map(r=>r.slice(0,vocab));
  if(!integrated)return sequences.map(s=>featureBoundaryScore(mean.map(normalize),s));
  const loadings=rows.map(r=>Array.from({length:m.rank},(_,d)=>r.slice((d+1)*vocab,(d+2)*vocab)));
  return sequences.map(s=>integratedBoundaryScore(mean,loadings,s,m.extent,fromFrame,m.quadrature));
}
export function featureReversal(c: LogitUncertaintyEvidence): boolean {
  return [c.integratedWritten,c.integratedRival,c.meanWritten,c.meanRival].every(Number.isFinite) &&
    c.integratedWritten>c.integratedRival && c.meanWritten<c.meanRival;
}
export function uncertaintyVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity=(e:CheckerEvent)=>JSON.stringify([e.id,e.timeMs,e.location,e.expected,e.observed]);
  const kept=new Set(reviewed.map(identity));
  const vetoed=new Set(baseline.filter(e=>e.errorClass==="wrong_word"&&!kept.has(identity(e))).map(identity));
  return live.filter(e=>e.errorClass==="wrong_word"&&vetoed.has(identity(e)));
}
export function carryLogitUncertaintyEvidence(chunks: PhonemeEvidence[], certificates: LogitUncertaintyEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,logitUncertaintyVetoes:certificates})});
}
export function featureComparison(m: LogitUncertaintyModel, x: number[][], candidate: SequenceEvidenceCandidate): LogitUncertaintyEvidence|null {
  const sequences=[candidate.written,candidate.rival].map(s=>[...s].map(p=>m.tokens.indexOf(p)+1));
  if (!Number.isSafeInteger(candidate.fromFrame)||candidate.fromFrame<0||
    candidate.toFrame-candidate.fromFrame!==x.length||!x.length||!x.every(r=>r.length===m.dimensions&&r.every(Number.isFinite))||
    sequences.some(s=>!s.length||s.some(p=>p<1))) return null;
  const a=featureScores(m.uncertainty,x,m,sequences,true,candidate.fromFrame),
    b=featureScores(m.uncertainty,x,m,sequences,false,candidate.fromFrame);
  const c={...candidate,integratedWritten:a[0],integratedRival:a[1],meanWritten:b[0],meanRival:b[1]};
  return [a[0],a[1],b[0],b[1]].every(Number.isFinite)?c:null;
}
export function validateLogitUncertaintyModel(value: unknown, manifest: ModelManifest): LogitUncertaintyModel {
  const m=value as LogitUncertaintyModel;
  const vector=(r:unknown,n:number):r is number[]=>Array.isArray(r)&&r.length===n&&r.every(Number.isFinite);
  if (!m||m.schema!==1||m.inventory!==manifest.inventory||m.tokens!==manifest.tokens||m.primarySha256!==manifest.sha256||
    m.dimensions!==32||![1,2].includes(m.rank)||![4,8].includes(m.extent)||![3,5].includes(m.quadrature)||
    typeof m.featureOutput!=="string"||!m.featureOutput||m.projection?.length!==32||!m.projection[0]?.length||
    !m.projection.every(r=>vector(r,m.projection[0].length))||!vector(m.center,32)||!vector(m.scale,32)||m.scale.some(v=>v<=0))
    throw new Error("Invalid uncertainty feature model");
  for(const h of [m.uncertainty,m.control]) {
    if (!h||h.hidden?.length!==48||!h.hidden.every(r=>vector(r,32))||!vector(h.hiddenBias,48)||
      h.output?.length!==(m.rank+1)*(m.tokens.length+1)||!h.output.every(r=>vector(r,48))||!vector(h.outputBias,(m.rank+1)*(m.tokens.length+1)))
      throw new Error("Invalid pronunciation comparison head");
  }
  return m;
}
/** Separate state instance; exposes pre-head encoder features on the primary stream's native frame clock. */
export async function featureObservations(backend: InferenceBackend, manifest: ModelManifest, model: LogitUncertaintyModel, audio: Float32Array) {
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
export async function addLogitUncertaintyEvidence(scorer: {model:LogitUncertaintyModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:LogitUncertaintyEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if(!scorer||!audio.length)return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for(const e of insertionCells(chunks))checker.push(e,tracker.push(trackerInput(e)));checker.end(tracker.end());
  if(!candidates.size)return chunks;
  const features=await featureObservations(scorer.backend,manifest,scorer.model,audio),certificates:LogitUncertaintyEvidence[]=[];
  for(const c of candidates.values()) {
    const result=c.fromFrame>=0&&c.toFrame<=features.observations.length?
      featureComparison(scorer.model,features.observations.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(result,c);if(result&&featureReversal(result))certificates.push(result);
  }
  return carryLogitUncertaintyEvidence(chunks,certificates);
}
