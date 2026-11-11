import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { AcceptedSpeech, CheckerEvent, CheckerOptions, SessionReadoutPhoneEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

/** Rows are feature coordinates (including intercept), columns are phones. */
export interface SessionReadoutModel {
  schema: 1; inventory: string; tokens: string; dimensions: number; primarySha256: string;
  featureOutput: string; projection: number[][]; center: number[]; scale: number[];
  initialization: number[][]; regularizer: number[];
}
export interface SupportObservation extends AcceptedSpeech { features: number[][] }
export interface SessionReadoutTrainingTake {
  split: string; speaker: string; kind: string; excluded?: unknown; scored?: boolean;
  events?: { errorClass: string }[]; non_errors?: unknown[];
}
/** Recording exclusions concern audio/label validity independently of the speaker split. */
export function sessionReadoutTrainingRole(take: SessionReadoutTrainingTake, heldSpeakers: ReadonlySet<string>): "none" | null {
  if (take.split !== "train" || !take.speaker || heldSpeakers.has(take.speaker) || take.excluded || take.scored === false) return null;
  if (take.kind === "none") {
    if (take.events?.length || take.non_errors?.length) throw new Error("None-labelled anchor contains scripted deviations");
    return "none";
  }
  return null;
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
/** Exact transcript-conditioned soft CTC occupancies, including repeated articulations. */
export function softOccupancies(rows: number[][], ids: number[]): number[][] | null {
  if (!rows.length || !ids.length || ids.some(p => p<1 || p>=rows[0].length)) return null;
  const labels=[0,...ids.flatMap(p=>[p,0])], n=labels.length;
  const incoming=labels.map((p,s)=>[s,...(s?[s-1]:[]),...(s>1 && p && p!==labels[s-2]?[s-2]:[])]);
  const outgoing=labels.map((_,s)=>incoming.flatMap((edges,j)=>edges.includes(s)?[j]:[]));
  const alpha=rows.map(()=>labels.map(()=>-Infinity)), beta=rows.map(()=>labels.map(()=>-Infinity));
  alpha[0][0]=rows[0][0]; alpha[0][1]=rows[0][ids[0]];
  for(let t=1;t<rows.length;t++)for(let j=0;j<n;j++)alpha[t][j]=incoming[j].reduce((v,k)=>logadd(v,alpha[t-1][k]),-Infinity)+rows[t][labels[j]];
  const z=logadd(alpha.at(-1)![n-1],alpha.at(-1)![n-2]);
  if(!Number.isFinite(z))return null;
  beta.at(-1)![n-1]=beta.at(-1)![n-2]=0;
  for(let t=rows.length-2;t>=0;t--)for(let j=0;j<n;j++)beta[t][j]=outgoing[j].reduce((v,k)=>logadd(v,beta[t+1][k]+rows[t+1][labels[k]]),-Infinity);
  return rows.map((r,t)=> {
    const result=r.map(()=>0);
    for(let j=0;j<n;j++)result[labels[j]]+=Math.exp(alpha[t][j]+beta[t][j]-z);
    const total=result.reduce((a,b)=>a+b,0);return result.map(v=>v/total);
  });
}
export const readoutInput=(row:number[],m:SessionReadoutModel)=>[...row.map((v,d)=>(v-m.center[d])/m.scale[d]),1];
/** Query-free differentiable-ridge counterpart. Equal mass per accepted word, then per frame.
 * No passage, word identity, query features or query label is accepted by this function. */
export function adaptReadout(m:SessionReadoutModel,support:SupportObservation[]):number[][] {
  const d=m.dimensions+1,vocab=m.tokens.length+1;
  if(!support.length)return m.initialization.map(r=>r.slice());
  const gram=Array.from({length:d},()=>Array(d).fill(0)), rhs=m.initialization.map((r,i)=>r.map(v=>v*m.regularizer[i]));
  for(let i=0;i<d;i++)gram[i][i]=m.regularizer[i];
  for(const word of support) {
    const ids=[...word.pronunciation].map(p=>m.tokens.indexOf(p)+1), y=softOccupancies(word.logProbs,ids);
    if(!y || y.length!==word.features.length)throw new Error("Invalid accepted-speech occupancy");
    const weight=1/support.length/y.length;
    word.features.forEach((row,t)=> {
      const x=readoutInput(row,m), target=y[t].map(v=>Math.log(Math.max(v,1e-6)));
      for(let i=0;i<d;i++) {
        for(let j=0;j<d;j++)gram[i][j]+=weight*x[i]*x[j];
        for(let p=0;p<vocab;p++)rhs[i][p]+=weight*x[i]*target[p];
      }
    });
  }
  // Cholesky of positive-definite normal equations, with a learned strictly positive diagonal.
  const l=Array.from({length:d},()=>Array(d).fill(0));
  for(let i=0;i<d;i++)for(let j=0;j<=i;j++) {
    let v=gram[i][j];for(let k=0;k<j;k++)v-=l[i][k]*l[j][k];
    l[i][j]=i===j?Math.sqrt(v):v/l[j][j];
    if(!Number.isFinite(l[i][j]) || (i===j && l[i][j]<=0))throw new Error("Non-finite ridge system");
  }
  const w=Array.from({length:d},()=>Array(vocab).fill(0));
  for(let p=0;p<vocab;p++) {
    const y=Array(d).fill(0);
    for(let i=0;i<d;i++){let v=rhs[i][p];for(let j=0;j<i;j++)v-=l[i][j]*y[j];y[i]=v/l[i][i];}
    for(let i=d-1;i>=0;i--){let v=y[i];for(let j=i+1;j<d;j++)v-=l[j][i]*w[j][p];w[i][p]=v/l[i][i];}
  }
  return w;
}
/** Candidate order has no reference privilege; both use identical adapted emissions and topology. */
export function readoutScores(w:number[][],raw:number[][],m:SessionReadoutModel,sequences:number[][]):number[] {
  const rows=raw.map(row=> {
    const x=readoutInput(row,m), logits=Array.from({length:m.tokens.length+1},(_,p)=>x.reduce((v,a,d)=>v+a*w[d][p],0));
    const max=Math.max(...logits), z=max+Math.log(logits.reduce((v,a)=>v+Math.exp(a-max),0));
    return logits.map(v=>v-z);
  });
  return sequences.map(s=>featureBoundaryScore(rows,s));
}
export function featureReversal(c: SessionReadoutPhoneEvidence): boolean {
  return [c.adaptedWritten,c.adaptedRival,c.unadaptedWritten,c.unadaptedRival].every(Number.isFinite) &&
    c.adaptedWritten>c.adaptedRival && c.unadaptedWritten<c.unadaptedRival;
}
export function sessionReadoutVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity=(e:CheckerEvent)=>JSON.stringify([e.id,e.timeMs,e.location,e.expected,e.observed]);
  const kept=new Set(reviewed.map(identity));
  const vetoed=new Set(baseline.filter(e=>e.errorClass==="wrong_word"&&!kept.has(identity(e))).map(identity));
  return live.filter(e=>e.errorClass==="wrong_word"&&vetoed.has(identity(e)));
}
export function carrySessionReadoutEvidence(chunks: PhonemeEvidence[], certificates: SessionReadoutPhoneEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,sessionReadoutVetoes:certificates})});
}
export function featureComparison(m:SessionReadoutModel,x:number[][],candidate:SequenceEvidenceCandidate,support:SupportObservation[]):SessionReadoutPhoneEvidence|null {
  const sequences=[candidate.written,candidate.rival].map(s=>[...s].map(p=>m.tokens.indexOf(p)+1));
  if(!support.length || !x.length || x.length!==candidate.toFrame-candidate.fromFrame ||
    !x.every(r=>r.length===m.dimensions&&r.every(Number.isFinite)) || sequences.some(s=>!s.length||s.some(p=>p<1)))return null;
  if(support.some(s=>s.toFrame>candidate.fromFrame-Math.ceil(300/40)||s.availableFrame>candidate.fromFrame))throw new Error("Non-causal readout support");
  const adapted=adaptReadout(m,support);
  const a=readoutScores(adapted,x,m,sequences),b=readoutScores(m.initialization,x,m,sequences);
  const c={...candidate,adaptedWritten:a[0],adaptedRival:a[1],unadaptedWritten:b[0],unadaptedRival:b[1],
    supportFrames:support.map(({fromFrame,toFrame,availableFrame})=>({fromFrame,toFrame,availableFrame}))};
  return [a[0],a[1],b[0],b[1]].every(Number.isFinite)?c:null;
}
export function validateSessionReadoutModel(value:unknown,manifest:ModelManifest):SessionReadoutModel {
  const m=value as SessionReadoutModel;
  const vector=(r:unknown,n:number):r is number[]=>Array.isArray(r)&&r.length===n&&r.every(Number.isFinite);
  if(!m||m.schema!==1||m.inventory!==manifest.inventory||m.tokens!==manifest.tokens||m.primarySha256!==manifest.sha256||
    m.dimensions!==32||typeof m.featureOutput!=="string"||!m.featureOutput||m.projection?.length!==32||!m.projection[0]?.length||
    !m.projection.every(r=>vector(r,m.projection[0].length))||!vector(m.center,32)||!vector(m.scale,32)||m.scale.some(v=>v<=0)||
    !vector(m.regularizer,33)||m.regularizer.some(v=>v<=0)||m.initialization?.length!==33||
    !m.initialization.every(r=>vector(r,m.tokens.length+1)))throw new Error("Invalid session readout model");
  return m;
}
/** Separate state instance; exposes pre-head encoder features on the primary stream's native frame clock. */
export async function featureObservations(backend: InferenceBackend, manifest: ModelManifest, model: SessionReadoutModel, audio: Float32Array) {
  const features=new Fbank().push(audio), observations:number[][]=[],logProbs:number[][]=[],availableFrames:number[]=[];
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
      // Fbank uses 25 ms windows at a 10 ms shift. Even an early output in
      // this encoder call can depend on its complete input and cache update.
      availableFrames.push(Math.ceil(((f+manifest.window-1)*10+25)/manifest.frameMs));
    }
  }
  return {observations,logProbs,availableFrames};
}
export async function addSessionReadoutEvidence(scorer: {model:SessionReadoutModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:SessionReadoutPhoneEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if(!scorer||!audio.length)return chunks;
  const candidates=new Map<string,{candidate:SequenceEvidenceCandidate;support:AcceptedSpeech[]}>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c,support){candidates.set(JSON.stringify(c),{candidate:c,support});return true;}});
  for(const e of insertionCells(chunks))checker.push(e,tracker.push(trackerInput(e)));checker.end(tracker.end());
  if(!candidates.size)return chunks;
  const features=await featureObservations(scorer.backend,manifest,scorer.model,audio),certificates:SessionReadoutPhoneEvidence[]=[];
  for(const {candidate:c,support} of candidates.values()) {
    const causalSupport=support.map(s=>({...s,
      availableFrame:Math.max(s.availableFrame,features.availableFrames[s.toFrame-1]),
      features:features.observations.slice(s.fromFrame,s.toFrame)})).filter(s=>s.availableFrame<=c.fromFrame);
    const result=c.fromFrame>=0&&c.toFrame<=features.observations.length?
      featureComparison(scorer.model,features.observations.slice(c.fromFrame,c.toFrame),c,causalSupport):null;
    inspect?.(result,c);if(result&&featureReversal(result))certificates.push(result);
  }
  return carrySessionReadoutEvidence(chunks,certificates);
}
