import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, PairwisePhoneEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface FeatureHead { hidden: number[][]; hiddenBias: number[]; output: number[][]; outputBias: number[]; outputHasBias?: boolean }
export interface PairwiseModel {
  schema: 1; inventory: string; tokens: string; dimensions: number; primarySha256: string;
  featureOutput: string; projection: number[][]; center: number[]; scale: number[];
  pairs: { a: number; b: number; head: FeatureHead }[]; control: FeatureHead; parameters: number;
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
const dot=(a:number[],b:number[])=>a.reduce((s,v,i)=>s+v*b[i],0);
const normalize=(logits:number[])=> {
  const max=Math.max(...logits), z=max+Math.log(logits.reduce((s,v)=>s+Math.exp(v-max),0));
  return logits.map(v=>v-z);
};
function headLogits(head:FeatureHead,x:number[]):number[] {
  const hidden=head.hidden.map((w,i)=>Math.tanh(dot(w,x)+head.hiddenBias[i]));
  return head.output.map((w,i)=>dot(w,hidden)+head.outputBias[i]);
}
interface Tournament { ids:number[]; edges:number[]; coupling:number[][] }
/** Solve graph log odds with a zero-sum gauge. Contradictory cycles are reconciled by
 * least squares, with no expected-word input and no preferred candidate order. */
function tournaments(pairs:{a:number;b:number}[]):Tournament[] {
  const unseen=new Set(pairs.flatMap(p=>[p.a,p.b])), result:Tournament[]=[];
  while(unseen.size) {
    const group=new Set([Math.min(...unseen)]);
    let old=0;
    while(old!==group.size) { old=group.size;for(const p of pairs)if(group.has(p.a)||group.has(p.b)){group.add(p.a);group.add(p.b);} }
    const ids=[...group].sort((a,b)=>a-b);ids.forEach(i=>unseen.delete(i));
    const edges=pairs.flatMap((p,i)=>group.has(p.a)?[i]:[]), n=ids.length;
    const incidence=edges.map(k=>ids.map(i=>i===pairs[k].a?1:i===pairs[k].b?-1:0));
    const laplacian=ids.map((_,i)=>ids.map((_,j)=>incidence.reduce((s,r)=>s+r[i]*r[j],0)));
    // Replace one redundant Laplacian equation with sum(scores)=0.
    laplacian[0]=ids.map(()=>1);
    const rhs=ids.map((_,i)=>edges.map((_,e)=>i===0?0:incidence[e][i]));
    const matrix=laplacian.map((r,i)=>[...r,...rhs[i]]);
    for(let j=0;j<n;j++) {
      let pivot=j;for(let i=j+1;i<n;i++)if(Math.abs(matrix[i][j])>Math.abs(matrix[pivot][j]))pivot=i;
      [matrix[pivot],matrix[j]]=[matrix[j],matrix[pivot]];
      const d=matrix[j][j];if(Math.abs(d)<1e-12)throw new Error("Disconnected phone tournament");
      matrix[j]=matrix[j].map(v=>v/d);
      for(let i=0;i<n;i++)if(i!==j){const a=matrix[i][j];matrix[i]=matrix[i].map((v,k)=>v-a*matrix[j][k]);}
    }
    result.push({ids,edges,coupling:matrix.map(r=>r.slice(n))});
  }
  return result;
}
function coupled(control:number[],logOdds:number[],plan:Tournament[]):number[] {
  const result=control.slice();
  for(const t of plan) {
    const scores=t.coupling.map(r=>dot(r,t.edges.map(k=>logOdds[k]))), distribution=normalize(scores);
    const mass=t.ids.reduce((s,i)=>logadd(s,control[i]),-Infinity);
    t.ids.forEach((id,i)=>{result[id]=distribution[i]+mass;});
  }
  return result;
}
export function reconcilePhones(control:number[],logOdds:number[],pairs:{a:number;b:number}[]):number[] {
  return coupled(control,logOdds,tournaments(pairs));
}
/** Same normalized distribution and complete CTC topology for both pronunciation candidates.
 * The pair head replaces identity odds only within training-selected consonant components;
 * unpaired phones, blank and each component's total mass remain the multiclass control's. */
export function featureScores(arm:"pairwise"|"control",raw:number[][],m:PairwiseModel,sequences:number[][]):number[] {
  const plan=tournaments(m.pairs);
  const rows=raw.map(r=> {
    const x=r.map((v,d)=>(v-m.center[d])/m.scale[d]), control=normalize(headLogits(m.control,x));
    return arm==="control"?control:coupled(control,m.pairs.map(p=>headLogits(p.head,x)[0]),plan);
  });
  return sequences.map(s=>featureBoundaryScore(rows,s));
}
export function featureReversal(c: PairwisePhoneEvidence): boolean {
  return [c.pairwiseWritten,c.pairwiseRival,c.controlWritten,c.controlRival].every(Number.isFinite) &&
    c.pairwiseWritten>c.pairwiseRival && c.controlWritten<c.controlRival;
}
export function pairwiseVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity=(e:CheckerEvent)=>JSON.stringify([e.id,e.timeMs,e.location,e.expected,e.observed]);
  const kept=new Set(reviewed.map(identity));
  const vetoed=new Set(baseline.filter(e=>e.errorClass==="wrong_word"&&!kept.has(identity(e))).map(identity));
  return live.filter(e=>e.errorClass==="wrong_word"&&vetoed.has(identity(e)));
}
export function carryPairwiseEvidence(chunks: PhonemeEvidence[], certificates: PairwisePhoneEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,pairwisePhoneVetoes:certificates})});
}
export function featureComparison(m: PairwiseModel, x: number[][], candidate: SequenceEvidenceCandidate): PairwisePhoneEvidence|null {
  const sequences=[candidate.written,candidate.rival].map(s=>[...s].map(p=>m.tokens.indexOf(p)+1));
  if (!x.length||!x.every(r=>r.length===m.dimensions&&r.every(Number.isFinite))||
    sequences.some(s=>!s.length||s.some(p=>p<1))) return null;
  const a=featureScores("pairwise",x,m,sequences), b=featureScores("control",x,m,sequences);
  const c={...candidate,pairwiseWritten:a[0],pairwiseRival:a[1],controlWritten:b[0],controlRival:b[1]};
  return [a[0],a[1],b[0],b[1]].every(Number.isFinite)?c:null;
}
export function validatePairwiseModel(value: unknown, manifest: ModelManifest): PairwiseModel {
  const m=value as PairwiseModel;
  const vector=(r:unknown,n:number):r is number[]=>Array.isArray(r)&&r.length===n&&r.every(Number.isFinite);
  if (!m||m.schema!==1||m.inventory!==manifest.inventory||m.tokens!==manifest.tokens||m.primarySha256!==manifest.sha256||
    m.dimensions!==32||typeof m.featureOutput!=="string"||!m.featureOutput||m.projection?.length!==32||!m.projection[0]?.length||
    !m.projection.every(r=>vector(r,m.projection[0].length))||!vector(m.center,32)||!vector(m.scale,32)||m.scale.some(v=>v<=0))
    throw new Error("Invalid pairwise feature model");
  const head=(h:FeatureHead,outputs:number)=> {
    const width=h?.hidden?.length;
    if(!Number.isSafeInteger(width)||width<1||!h.hidden.every(r=>vector(r,32))||!vector(h.hiddenBias,width)||
      h.output?.length!==outputs||!h.output.every(r=>vector(r,width))||!vector(h.outputBias,outputs)||
      (h.outputHasBias===false&&h.outputBias.some(v=>v!==0)))throw new Error("Invalid pronunciation comparison head");
    return width*(32+1+outputs)+(h.outputHasBias===false?0:outputs);
  };
  if(!Array.isArray(m.pairs)||!m.pairs.length)throw new Error("No trained phone tournaments");
  const seen=new Set<string>();
  for(const p of m.pairs) {
    const key=JSON.stringify([p.a,p.b]);
    if(!Number.isInteger(p.a)||!Number.isInteger(p.b)||p.a<1||p.b<=p.a||p.b>m.tokens.length||seen.has(key))
      throw new Error("Invalid phone contrast");
    seen.add(key);
  }
  const control=head(m.control,m.tokens.length+1), binary=m.pairs.reduce((s,p)=>s+head(p.head,1),0);
  if(control!==binary||m.parameters!==control)throw new Error("Pairwise and multiclass parameter counts differ");
  return m;
}
/** Separate state instance; exposes pre-head encoder features on the primary stream's native frame clock. */
export async function featureObservations(backend: InferenceBackend, manifest: ModelManifest, model: PairwiseModel, audio: Float32Array) {
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
export async function addPairwiseEvidence(scorer: {model:PairwiseModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:PairwisePhoneEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if(!scorer||!audio.length)return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for(const e of insertionCells(chunks))checker.push(e,tracker.push(trackerInput(e)));checker.end(tracker.end());
  if(!candidates.size)return chunks;
  const features=await featureObservations(scorer.backend,manifest,scorer.model,audio),certificates:PairwisePhoneEvidence[]=[];
  for(const c of candidates.values()) {
    const result=c.fromFrame>=0&&c.toFrame<=features.observations.length?
      featureComparison(scorer.model,features.observations.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(result,c);if(result&&featureReversal(result))certificates.push(result);
  }
  return carryPairwiseEvidence(chunks,certificates);
}
