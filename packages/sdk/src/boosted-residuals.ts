import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, BoostedResidualEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface TrainingTake {kind?:string;split?:string;speaker?:string;exclude?:boolean|string|null;excluded?:boolean|string|null;scored?:boolean;events?:unknown[];non_errors?:unknown[]}
export function eligibleResidualTake(r: TrainingTake, excluded:ReadonlySet<string>):boolean {
  return r.kind === "none" && eligiblePreservationTake(r, excluded) && !r.events?.length && !r.non_errors?.length;
}
export function eligiblePreservationTake(r: TrainingTake, excluded:ReadonlySet<string>):boolean {
  return r.split === "train" && !!r.speaker && !excluded.has(r.speaker) && !r.exclude && !r.excluded && r.scored !== false;
}
/** Every split partitions feature space once for all phone gradients: shared vector-valued leaves. */
export interface ResidualTree { feature:number[]; threshold:number[]; left:number[]; right:number[]; value:number[][] }
export interface ResidualMetadata {
  schema:1; inventory:string; tokens:string; dimensions:number; width:number;
  featureOutput:string; logitOutput:string; primarySha256:string;
}
export interface ResidualModel extends ResidualMetadata { rate:number; trees:ResidualTree[] }
const logadd=(a:number,b:number):number=> {
  const m=Math.max(a,b); return Number.isFinite(m) ? m+Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
/** Ordinary transcript CTC forward sum: no duration or completion-normalization prior. */
export function residualCtcScore(logProbs:number[][], ids:number[]):number {
  if (!logProbs.length || !ids.length || ids.some(p=>!Number.isInteger(p)||p<1||p>=logProbs[0].length)) return -Infinity;
  const labels=[0,...ids.flatMap(p=>[p,0])];
  let a=labels.map(()=>-Infinity); a[0]=logProbs[0][0]; a[1]=logProbs[0][ids[0]];
  for(let t=1;t<logProbs.length;t++) a=labels.map((p,s)=> {
    let sum=a[s]; if(s>0)sum=logadd(sum,a[s-1]);
    if(s>1&&p!==0&&p!==labels[s-2])sum=logadd(sum,a[s-2]);
    return sum+logProbs[t][p];
  });
  return logadd(a[a.length-1],a[a.length-2]);
}
export function treeValue(tree:ResidualTree,x:number[]):number[] {
  let node=0;
  while(tree.left[node]!==-1)node=x[tree.feature[node]]<=tree.threshold[node] ? tree.left[node] : tree.right[node];
  return tree.value[node];
}
function normalize(row:number[]):number[] {
  const maximum=Math.max(...row), sum=maximum+Math.log(row.reduce((s,v)=>s+Math.exp(v-maximum),0));
  return row.map(v=>v-sum);
}
export function correctedResidualRows(model:ResidualModel, features:number[][], logits:number[][]):number[][] {
  return logits.map((row,t)=> {
    const z=row.slice();
    for(const tree of model.trees) { const delta=treeValue(tree,features[t]); for(let p=0;p<z.length;p++)z[p]+=model.rate*delta[p]; }
    return normalize(z);
  });
}
export function residualPreference(c:BoostedResidualEvidence):boolean {
  return [c.shippedWritten,c.shippedRival,c.correctedWritten,c.correctedRival].every(Number.isFinite) && c.shippedWritten<c.shippedRival && c.correctedWritten>c.correctedRival;
}
export function residualComparison(model:ResidualModel, features:number[][], logits:number[][],
  candidate:SequenceEvidenceCandidate):BoostedResidualEvidence|null {
  const sequences=[candidate.written,candidate.rival].map(s=>[...s].map(p=>model.tokens.indexOf(p)+1));
  const valid=(x:number[][],width:number)=>x.length>0&&x.every(r=>r.length===width&&r.every(Number.isFinite));
  if (!Number.isInteger(candidate.fromFrame)||candidate.fromFrame<0||!Number.isInteger(candidate.toFrame)||
      candidate.toFrame-candidate.fromFrame!==features.length||features.length!==logits.length||
      !valid(features,model.width)||!valid(logits,model.dimensions)||sequences.some(s=>!s.length||s.some(p=>p<1)))return null;
  const shipped=logits.map(normalize), corrected=correctedResidualRows(model,features,logits);
  const result={...candidate,shippedWritten:residualCtcScore(shipped,sequences[0]),shippedRival:residualCtcScore(shipped,sequences[1]),
    correctedWritten:residualCtcScore(corrected,sequences[0]),correctedRival:residualCtcScore(corrected,sequences[1])};
  return Object.values(result).filter(v=>typeof v==="number").every(Number.isFinite) ? result : null;
}
export function validateResidualModel(value:unknown, manifest:ModelManifest):ResidualModel {
  const m=value as ResidualModel;
  const vector=(r:unknown,n:number):r is number[]=>Array.isArray(r)&&r.length===n&&r.every(Number.isFinite);
  if(!m||m.schema!==1||m.inventory!==manifest.inventory||m.tokens!==manifest.tokens||m.primarySha256!==manifest.sha256||
     m.dimensions!==m.tokens.length+1||!Number.isSafeInteger(m.width)||m.width<1||!m.featureOutput||!m.logitOutput||
     !Number.isFinite(m.rate)||m.rate<=0||m.rate>1||!Array.isArray(m.trees)||m.trees.length>128)throw new Error("Invalid boosted residual model");
  for(const tree of m.trees) {
    const n=tree.feature?.length;
    if(!n||n>127||!vector(tree.threshold,n)||!vector(tree.left,n)||!vector(tree.right,n)||
       !Array.isArray(tree.value)||tree.value.length!==n||!tree.value.every(r=>vector(r,m.dimensions)))throw new Error("Invalid residual tree");
    const visited=new Set<number>();
    const walk=(i:number,depth:number)=> {
      if(!Number.isInteger(i)||i<0||i>=n||visited.has(i)||depth>6)throw new Error("Invalid residual tree topology");
      visited.add(i);
      if(tree.left[i]===-1) { if(tree.right[i]!==-1||tree.feature[i]!==-2)throw new Error("Invalid residual leaf"); }
      else {if(!Number.isInteger(tree.feature[i])||tree.feature[i]<0||tree.feature[i]>=m.width)throw new Error("Invalid residual feature");
        walk(tree.left[i],depth+1);walk(tree.right[i],depth+1);}
    };walk(0,0);if(visited.size!==n)throw new Error("Unreachable residual tree nodes");
  }
  return m;
}
/** Only retract the exact live occurrence removed by an otherwise identical evidence replay.
 * Repeated words and later graph confirmations may share a spelling or issue id. */
export function residualVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity = (e: CheckerEvent) => JSON.stringify([e.id, e.timeMs, e.location, e.expected, e.observed]);
  const kept = new Set(reviewed.map(identity));
  const vetoed = new Set(baseline.filter(e => e.errorClass === "wrong_word" && !kept.has(identity(e))).map(identity));
  return live.filter(e => e.errorClass === "wrong_word" && vetoed.has(identity(e)));
}
/** Carry auxiliary judgments without changing any primary token or posterior. */
export function carryResidualEvidence(chunks: PhonemeEvidence[], certificates: BoostedResidualEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e,i) => i !== first ? e : {...e, phonemes:e.phonemes.map((p,j) => j ? p : {
    ...p, boostedResidualVetoes:certificates,
  })});
}
/** Full-session separate streaming state. The final linear input includes activation quantization.
 * Frozen features and logits retain the primary clock; primary rows and caches are untouched. */
export async function residualObservations(backend: InferenceBackend, manifest: ModelManifest, meta: ResidualMetadata, audio: Float32Array) {
  if (manifest.fbank.mels !== 80 || manifest.window <= 0 || manifest.shift <= 0) throw new Error("Unsupported residual extractor geometry");
  const features = new Fbank().push(audio), logProbs: number[][] = [], featuresOut: number[][] = [], logits: number[][] = [];
  const total = Math.max(0,Math.floor((Math.floor((features.length-7)/2)+1)/2));
  let states: Record<string,Tensor> = Object.fromEntries(manifest.states.map(s=>[s.name,{type:s.dtype,dims:s.shape,
    data:s.dtype === "int64" ? new BigInt64Array(s.shape.reduce((a,b)=>a*b,1)) : new Float32Array(s.shape.reduce((a,b)=>a*b,1))}]));
  const pad = new Float32Array(manifest.fbank.mels).fill(manifest.fbank.padValue);
  for (let f=0;logProbs.length<total;f+=manifest.shift) {
    const input = new Float32Array(manifest.window*manifest.fbank.mels);
    for(let i=0;i<manifest.window;i++)input.set(features[f+i]??pad,i*manifest.fbank.mels);
    const out=await backend.run({x:{type:"float32",dims:[1,manifest.window,manifest.fbank.mels],data:input},...states});
    states=Object.fromEntries(manifest.states.map(s=>[s.name,out[`new_${s.name}`]]));
    const vocab=manifest.tokens.length+1, n=out.log_probs.data.length/vocab;
    const h=out[meta.featureOutput],z=out[meta.logitOutput];
    if(!Number.isSafeInteger(n)||n<=0||!h||!z||h.type!=="float32"||z.type!=="float32"||
       h.data.length!==n*meta.width||z.data.length!==n*vocab) throw new Error("Invalid final projection geometry");
    for(let t=0;t<n && logProbs.length<total;t++) {
      featuresOut.push(Array.from(h.data.slice(t*meta.width,(t+1)*meta.width),Number));
      logits.push(Array.from(z.data.slice(t*vocab,(t+1)*vocab),Number));
      logProbs.push(Array.from(out.log_probs.data.slice(t*vocab,(t+1)*vocab),Number));
    }
  }
  return {features:featuresOut,logits,logProbs};
}
export async function addResidualEvidence(scorer: {model:ResidualModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:BoostedResidualEvidence|null,request:SequenceEvidenceCandidate)=>void,
  compare?:(request:SequenceEvidenceCandidate)=>BoostedResidualEvidence|null):Promise<PhonemeEvidence[]> {
  if (!scorer||!audio.length) return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>(),cells=insertionCells(chunks);
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e))); checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const features=compare ? undefined : await residualObservations(scorer.backend,manifest,scorer.model,audio),certificates:BoostedResidualEvidence[]=[];
  for (const c of candidates.values()) {
    const result=compare ? compare(c) : features && c.fromFrame>=0&&c.toFrame<=features.features.length ? residualComparison(scorer.model,
      features.features.slice(c.fromFrame,c.toFrame),features.logits.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(result,c); if (result&&residualPreference(result)) certificates.push(result);
  }
  return carryResidualEvidence(chunks,certificates);
}
