import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, ChannelTransitionEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface FeatureHead { /** Weights on [previous, current, next] normalized encoder frames. */ hidden: number[][]; hiddenBias: number[]; output: number[][]; outputBias: number[] }
export interface ChannelTransitionModel {
  schema: 1; inventory: string; tokens: string; dimensions: number; primarySha256: string;
  featureOutput: string; projection: number[][]; center: number[]; scale: number[];
  transition: FeatureHead; unpaired: FeatureHead; fixed: FeatureHead;
}
export interface ChannelTransitionTrainingTake {
  split: string; speaker: string; kind: string; excluded?: unknown; scored?: boolean;
  events?: { errorClass: string }[]; non_errors?: unknown[];
}
/** Recording exclusions concern audio/label validity independently of the speaker split. */
export function channelTransitionTrainingRole(take: ChannelTransitionTrainingTake, heldSpeakers: ReadonlySet<string>): "none" | null {
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
/** Candidate order carries no reference privilege: each complete pronunciation uses the same emissions. */
export function featureScores(head: FeatureHead, raw: number[][], m: ChannelTransitionModel, sequences: number[][], context: { before?: number[]; after?: number[] } = {}): number[] {
  const dot=(a:number[],b:number[])=>a.reduce((s,v,i)=>s+v*b[i],0);
  const normalize=(r:number[])=>r.map((v,d)=>(v-m.center[d])/m.scale[d]);
  const normalized=raw.map(normalize);
  const rows=normalized.map((r,t)=> {
    const x=[...(normalized[t-1] ?? normalize(context.before ?? raw[0])), ...r,
      ...(normalized[t+1] ?? normalize(context.after ?? raw[raw.length-1]))];
    const hidden=head.hidden.map((w,i)=>Math.tanh(dot(w,x)+head.hiddenBias[i]));
    const logits=head.output.map((w,i)=>dot(w,hidden)+head.outputBias[i]);
    const max=Math.max(...logits), z=max+Math.log(logits.reduce((s,v)=>s+Math.exp(v-max),0));
    return logits.map(v=>v-z);
  });
  return sequences.map(s=>featureBoundaryScore(rows,s));
}
export function featureReversal(c: ChannelTransitionEvidence): boolean {
  return [c.transitionWritten,c.transitionRival,c.unpairedWritten,c.unpairedRival,c.fixedWritten,c.fixedRival].every(Number.isFinite) &&
    c.transitionWritten>c.transitionRival && c.unpairedWritten<c.unpairedRival && c.fixedWritten<c.fixedRival;
}
export function channelTransitionVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity=(e:CheckerEvent)=>JSON.stringify([e.id,e.timeMs,e.location,e.expected,e.observed]);
  const kept=new Set(reviewed.map(identity));
  const vetoed=new Set(baseline.filter(e=>e.errorClass==="wrong_word"&&!kept.has(identity(e))).map(identity));
  return live.filter(e=>e.errorClass==="wrong_word"&&vetoed.has(identity(e)));
}
export function carryChannelTransitionEvidence(chunks: PhonemeEvidence[], certificates: ChannelTransitionEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,channelTransitionVetoes:certificates})});
}
export function featureComparison(m: ChannelTransitionModel, x: number[][], candidate: SequenceEvidenceCandidate, context: {before?:number[];after?:number[]} = {}): ChannelTransitionEvidence|null {
  const sequences=[candidate.written,candidate.rival].map(s=>[...s].map(p=>m.tokens.indexOf(p)+1));
  if (!Number.isSafeInteger(candidate.fromFrame)||candidate.fromFrame<0||!Number.isSafeInteger(candidate.toFrame)||
    x.length!==candidate.toFrame-candidate.fromFrame||!x.length||!x.every(r=>r.length===m.dimensions&&r.every(Number.isFinite))||
    sequences.some(s=>!s.length||s.some(p=>p<1)) ||
    [context.before,context.after].some(r=>r && (r.length!==m.dimensions||!r.every(Number.isFinite)))) return null;
  const a=featureScores(m.transition,x,m,sequences,context), b=featureScores(m.unpaired,x,m,sequences,context),
    f=featureScores(m.fixed,x,m,sequences,context);
  const c={...candidate,transitionWritten:a[0],transitionRival:a[1],unpairedWritten:b[0],unpairedRival:b[1],fixedWritten:f[0],fixedRival:f[1]};
  return [a[0],a[1],b[0],b[1],f[0],f[1]].every(Number.isFinite)?c:null;
}
export function validateChannelTransitionModel(value: unknown, manifest: ModelManifest): ChannelTransitionModel {
  const m=value as ChannelTransitionModel;
  const vector=(r:unknown,n:number):r is number[]=>Array.isArray(r)&&r.length===n&&r.every(Number.isFinite);
  if (!m||m.schema!==1||m.inventory!==manifest.inventory||m.tokens!==manifest.tokens||m.primarySha256!==manifest.sha256||
    m.dimensions!==32||typeof m.featureOutput!=="string"||!m.featureOutput||m.projection?.length!==32||!m.projection[0]?.length||
    !m.projection.every(r=>vector(r,m.projection[0].length))||!vector(m.center,32)||!vector(m.scale,32)||m.scale.some(v=>v<=0))
    throw new Error("Invalid channel-transition feature model");
  for(const h of [m.transition,m.unpaired,m.fixed]) {
    if (!h||h.hidden?.length!==48||!h.hidden.every(r=>vector(r,96))||!vector(h.hiddenBias,48)||
      h.output?.length!==m.tokens.length+1||!h.output.every(r=>vector(r,48))||!vector(h.outputBias,m.tokens.length+1))
      throw new Error("Invalid pronunciation comparison head");
  }
  return m;
}
/** Separate state instance; exposes pre-head encoder features on the primary stream's native frame clock. */
export async function featureObservations(backend: InferenceBackend, manifest: ModelManifest, model: ChannelTransitionModel, audio: Float32Array) {
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
    if(!Number.isSafeInteger(n)||n<=0)throw new Error("Frozen encoder returned no complete frames");
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
export async function addChannelTransitionEvidence(scorer: {model:ChannelTransitionModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:ChannelTransitionEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if(!scorer||!audio.length)return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for(const e of insertionCells(chunks))checker.push(e,tracker.push(trackerInput(e)));checker.end(tracker.end());
  if(!candidates.size)return chunks;
  const features=await featureObservations(scorer.backend,manifest,scorer.model,audio),certificates:ChannelTransitionEvidence[]=[];
  for(const c of candidates.values()) {
    const result=c.fromFrame>=0&&c.toFrame<=features.observations.length?
      featureComparison(scorer.model,features.observations.slice(c.fromFrame,c.toFrame),c,
        {before:features.observations[c.fromFrame-1],after:features.observations[c.toFrame]}):null;
    inspect?.(result,c);if(result&&featureReversal(result))certificates.push(result);
  }
  return carryChannelTransitionEvidence(chunks,certificates);
}
