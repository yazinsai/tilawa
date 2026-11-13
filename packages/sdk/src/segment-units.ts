import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate, SegmentUnitEvidence } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface SegmentMetadata {
  schema: 1; inventory: string; tokens: string; width: number;
  featureOutput: string; primarySha256: string;
}
export interface SegmentArm { centers: number[][]; variance: number; phones: number[][]; counts:number[] }
export interface SegmentModel extends SegmentMetadata {
  center: number[]; scale: number[]; projection: number[][];
  maxDuration: number; penalty: number; independent: SegmentArm; forced: SegmentArm;
}
export interface SegmentComparison extends SequenceEvidenceCandidate {
  independentWritten: number; independentRival: number; forcedWritten: number; forcedRival: number;
}
/** Both arms must share capacity, geometry and primary provenance. A declared incomplete asset is an error. */
export function validateSegmentModel(value:unknown,manifest:ModelManifest):SegmentModel {
  const m=value as SegmentModel;
  const vector=(v:unknown,n:number):v is number[]=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite);
  if(!m||m.schema!==1||m.inventory!==manifest.inventory||m.tokens!==manifest.tokens||m.primarySha256!==manifest.sha256||
     !Number.isSafeInteger(m.width)||m.width<=0||!m.featureOutput||!vector(m.center,m.width)||!vector(m.scale,m.width)||m.scale.some(v=>v<=0)||
     !Array.isArray(m.projection)||m.projection.length<1||m.projection.length>16||m.projection.some(r=>!vector(r,m.width))||
     !Number.isSafeInteger(m.maxDuration)||m.maxDuration<1||m.maxDuration>16||!Number.isFinite(m.penalty)||m.penalty<=0)
    throw new Error("Invalid segment-unit model");
  const dimensions=m.projection.length*3,k=m.independent?.centers?.length;
  if(!Number.isSafeInteger(k)||k<2||k>64)throw new Error("Invalid segment codebook capacity");
  for(const arm of [m.independent,m.forced]) {
    if(!arm||!Array.isArray(arm.centers)||arm.centers.length!==k||arm.centers.some(r=>!vector(r,dimensions))||
       !Number.isFinite(arm.variance)||arm.variance<=0||!vector(arm.counts,m.tokens.length+1)||arm.counts.some(v=>v<0)||
       !Array.isArray(arm.phones)||arm.phones.length!==m.tokens.length+1||
       arm.phones.some(r=>!vector(r,k)||r.some(v=>v<=0)||Math.abs(r.reduce((a,b)=>a+b,0)-1)>1e-6))
      throw new Error("Invalid matched segment-unit arm");
  }
  return m;
}
export function eligibleSegmentTake(r: {split?:string;speaker?:string;exclude?:unknown;excluded?:unknown}, excluded:ReadonlySet<string>):boolean {
  return r.split === "train" && !!r.speaker && !excluded.has(r.speaker) && !r.exclude && !r.excluded;
}
export function canonicalSegmentTake(r: {kind?:string;scored?:boolean;events?:unknown[];non_errors?:unknown[]}):boolean {
  return r.kind === "none" && r.scored !== false && !r.events?.length && !r.non_errors?.length;
}
const logadd=(a:number,b:number)=> {
  const m=Math.max(a,b);return Number.isFinite(m)?m+Math.log(Math.exp(a-m)+Math.exp(b-m)):-Infinity;
};
/** Three equally spaced samples in normalized segment time, including interpolation for one-frame sounds. */
export function segmentDescriptor(x:number[][],start:number,end:number):number[] {
  if (start<0||end> x.length||end<=start) throw new Error("Invalid segment interval");
  return [1/6,1/2,5/6].flatMap(u=> {
    const position=start+u*(end-start-1),lo=Math.floor(position),hi=Math.ceil(position),fraction=position-lo;
    return x[lo].map((v,d)=>v*(1-fraction)+x[hi][d]*fraction);
  });
}
function unitPosteriors(arm:SegmentArm,descriptor:number[]):{q:number[];background:number} {
  const logits=arm.centers.map(c=>-c.reduce((s,v,d)=>s+(v-descriptor[d])**2,0)/(2*arm.variance));
  const z=logits.reduce(logadd,-Infinity);return {q:logits.map(v=>Math.exp(v-z)),background:z-Math.log(logits.length)};
}
/** Complete pronunciation, marginalized over segment lengths, units and monotone phone correspondence.
 * A phone may emit several units. Blank separates repeated phones. Subtract the same completion-conditioned
 * acoustic-background timing prior so a longer spelling cannot win just by having more legal segmentations. */
export function segmentPronunciationScore(model:SegmentModel,arm:SegmentArm,x:number[][],ids:number[]):number {
  if (!x.length||!ids.length||ids.some(p=>p<1||p>=arm.phones.length)) return -Infinity;
  const labels=[0,...ids.flatMap(p=>[p,0])],n=labels.length;
  const edges=labels.map((p,s)=>[s,...(s+1<n?[s+1]:[]),...(s+2<n&&labels[s+2]&&labels[s+2]!==p?[s+2]:[])]);
  const a=Array.from({length:x.length+1},()=>Array(n).fill(-Infinity) as number[]),prior=a.map(r=>r.slice());
  for(let t=0;t<x.length;t++) for(let length=1;length<=model.maxDuration&&t+length<=x.length;length++) {
    const {q,background}=unitPosteriors(arm,segmentDescriptor(x,t,t+length));
    const emission=labels.map(p=>length*Math.log(q.reduce((s,v,k)=>s+v*arm.phones[p][k],0)*q.length));
    const cost=-model.penalty-Math.log(model.maxDuration)+length*background;
    if(t===0) for(const s of [0,1]) {
      a[length][s]=logadd(a[length][s],cost+emission[s]);prior[length][s]=logadd(prior[length][s],cost);
    }
    else for(let s=0;s<n;s++) for(const next of edges[s]) {
      const timing=cost-Math.log(edges[s].length);
      a[t+length][next]=logadd(a[t+length][next],a[t][s]+timing+emission[next]);
      prior[t+length][next]=logadd(prior[t+length][next],prior[t][s]+timing);
    }
  }
  const likelihood=logadd(a[x.length][n-1],a[x.length][n-2]),timing=logadd(prior[x.length][n-1],prior[x.length][n-2]);
  return Number.isFinite(likelihood)&&Number.isFinite(timing)?likelihood-timing:-Infinity;
}
export function segmentReversal(c:SegmentComparison):boolean {
  return [c.independentWritten,c.independentRival,c.forcedWritten,c.forcedRival].every(Number.isFinite)&&
    c.independentWritten>c.independentRival&&c.forcedWritten<c.forcedRival;
}
export function segmentComparison(model:SegmentModel,features:number[][],candidate:SequenceEvidenceCandidate):SegmentComparison|null {
  const ids=[candidate.written,candidate.rival].map(s=>[...s].map(p=>model.tokens.indexOf(p)+1));
  if(!Number.isInteger(candidate.fromFrame)||candidate.fromFrame<0||!Number.isInteger(candidate.toFrame)||
     candidate.toFrame-candidate.fromFrame!==features.length||ids.some(s=>!s.length||s.includes(0))||
     [0,...ids.flat()].some(p=>model.independent.counts[p]<=0||model.forced.counts[p]<=0)||
     !features.length||features.some(r=>r.length!==model.width||r.some(v=>!Number.isFinite(v)))) return null;
  const x=features.map(r=>model.projection.map(row=>row.reduce((sum,v,d)=>sum+v*(r[d]-model.center[d])/model.scale[d],0)));
  const independent=ids.map(s=>segmentPronunciationScore(model,model.independent,x,s));
  const forced=ids.map(s=>segmentPronunciationScore(model,model.forced,x,s));
  const result={...candidate,independentWritten:independent[0],independentRival:independent[1],forcedWritten:forced[0],forcedRival:forced[1]};
  return [result.independentWritten,result.independentRival,result.forcedWritten,result.forcedRival].every(Number.isFinite)?result:null;
}
/** Native deployed clock and fresh recurrent state; no primary outputs or caches are replaced. */
export async function segmentObservations(backend:InferenceBackend,manifest:ModelManifest,meta:SegmentMetadata,audio:Float32Array) {
  const features=new Fbank().push(audio),raw:number[][]=[],logProbs:number[][]=[];
  const total=Math.max(0,Math.floor((Math.floor((features.length-7)/2)+1)/2));
  let states:Record<string,Tensor>=Object.fromEntries(manifest.states.map(s=>[s.name,{type:s.dtype,dims:s.shape,
    data:s.dtype==="int64"?new BigInt64Array(s.shape.reduce((a,b)=>a*b,1)):new Float32Array(s.shape.reduce((a,b)=>a*b,1))}]));
  const pad=new Float32Array(manifest.fbank.mels).fill(manifest.fbank.padValue);
  for(let f=0;raw.length<total;f+=manifest.shift) {
    const input=new Float32Array(manifest.window*manifest.fbank.mels);
    for(let i=0;i<manifest.window;i++)input.set(features[f+i]??pad,i*manifest.fbank.mels);
    const out=await backend.run({x:{type:"float32",dims:[1,manifest.window,manifest.fbank.mels],data:input},...states});
    states=Object.fromEntries(manifest.states.map(s=>[s.name,out[`new_${s.name}`]]));
    const vocab=manifest.tokens.length+1,n=out.log_probs.data.length/vocab,h=out[meta.featureOutput];
    if(!Number.isSafeInteger(n)||n<=0||!h||h.type!=="float32"||h.data.length!==n*meta.width)throw new Error("Invalid encoder feature clock");
    for(let t=0;t<n&&raw.length<total;t++) {
      raw.push(Array.from(h.data.slice(t*meta.width,(t+1)*meta.width),Number));
      logProbs.push(Array.from(out.log_probs.data.slice(t*vocab,(t+1)*vocab),Number));
    }
  }
  return {features:raw,log_probs:logProbs};
}

/** Only retract the exact live occurrence removed by an otherwise identical evidence replay.
 * Repeated words and later graph confirmations may share a spelling or issue id. */
export function segmentVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity = (e: CheckerEvent) => JSON.stringify([e.id, e.timeMs, e.location, e.expected, e.observed]);
  const kept = new Set(reviewed.map(identity));
  const vetoed = new Set(baseline.filter(e => e.errorClass === "wrong_word" && !kept.has(identity(e))).map(identity));
  return live.filter(e => e.errorClass === "wrong_word" && vetoed.has(identity(e)));
}
/** Carry auxiliary judgments without changing any primary token or posterior. */
export function carrySegmentEvidence(chunks: PhonemeEvidence[], certificates: SegmentUnitEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e,i) => i !== first ? e : {...e, phonemes:e.phonemes.map((p,j) => j ? p : {
    ...p, segmentUnitVetoes:certificates,
  })});
}

export async function addSegmentEvidence(scorer: {model:SegmentModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:SegmentUnitEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if (!scorer||!audio.length) return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>(),cells=insertionCells(chunks);
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e))); checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const features=await segmentObservations(scorer.backend,manifest,scorer.model,audio),certificates:SegmentUnitEvidence[]=[];
  for (const c of candidates.values()) {
    const result=c.fromFrame>=0&&c.toFrame<=features.features.length ? segmentComparison(scorer.model,
      features.features.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(result,c); if (result&&segmentReversal(result)) certificates.push(result);
  }
  return carrySegmentEvidence(chunks,certificates);
}
