import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, BoundedSequenceEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

/** None labels are canonical supervision only when the recording is usable and has no edits/repeats. */
export function eligibleBoundedTake(r: {kind?:string;split?:string;speaker?:string;exclude?:boolean|string|null;excluded?:boolean|string|null;scored?:boolean;events?:unknown[];non_errors?:unknown[]}, excluded:ReadonlySet<string>):boolean {
  return r.kind === "none" && r.split === "train" && !!r.speaker && !excluded.has(r.speaker) &&
    !r.exclude && !r.excluded && r.scored !== false && !r.events?.length && !r.non_errors?.length;
}


export interface BoundedMetadata {
  schema: 1; inventory: string; tokens: string; width: number;
  featureOutput: string; logitOutput: string; primarySha256: string;
}
export interface CtcHead { weight: number[][]; bias: number[] }
export interface BoundedModel extends BoundedMetadata {
  center: number[]; scale: number[]; bounded: CtcHead; control: CtcHead;
  enlargedBounded: CtcHead; enlargedControl: CtcHead; candidateLimits: [number, number];
}
export function validateBoundedModel(value: unknown, manifest: ModelManifest): BoundedModel {
  const m=value as BoundedModel;
  const vector=(r:unknown,n:number):r is number[]=>Array.isArray(r)&&r.length===n&&r.every(Number.isFinite);
  const matrix=(r:unknown,n:number,d:number)=>Array.isArray(r)&&r.length===n&&r.every(v=>vector(v,d));
  if(!m||m.schema!==1||m.inventory!==manifest.inventory||m.tokens!==manifest.tokens||
    m.primarySha256!==manifest.sha256||!Number.isSafeInteger(m.width)||m.width<=0||
    !m.featureOutput||!m.logitOutput||!vector(m.center,m.width)||!vector(m.scale,m.width)||m.scale.some(v=>v<=0))
    throw new Error("Invalid bounded sequence readout metadata");
  if(!Array.isArray(m.candidateLimits)||m.candidateLimits.length!==2||
    !m.candidateLimits.every(n=>Number.isSafeInteger(n)&&n>0)||m.candidateLimits[1]<=m.candidateLimits[0])
    throw new Error("Invalid competing-sequence limits");
  for(const h of [m.bounded,m.control,m.enlargedBounded,m.enlargedControl])
    if(!h||!matrix(h.weight,m.tokens.length+1,m.width)||!vector(h.bias,m.tokens.length+1))
      throw new Error("Invalid or unequal-capacity CTC heads");
  return m;
}
export function readout(head: CtcHead, raw: number[][], center: number[], scale: number[]): number[][] {
  return raw.map(row => {
    const x=row.map((v,d)=>(v-center[d])/scale[d]);
    const logits=head.weight.map((w,p)=>w.reduce((s,v,d)=>s+v*x[d],head.bias[p]));
    const max=Math.max(...logits), normal=max+Math.log(logits.reduce((s,v)=>s+Math.exp(v-max),0));
    return logits.map(v=>v-normal);
  });
}
const add=(a:number,b:number):number=> {
  const m=Math.max(a,b); return Number.isFinite(m)?m+Math.log(Math.exp(a-m)+Math.exp(b-m)):-Infinity;
};
/** Ordinary CTC forward sum: no imported word timings and no independently selected segments. */
export function sequenceCtcScore(rows: number[][], labels: number[]): number {
  if (labels.some(p=>!Number.isInteger(p)||p<=0||p>=rows[0]?.length)) return -Infinity;
  if (!labels.length) return rows.reduce((sum,r)=>sum+r[0],0);
  if (!rows.length) return -Infinity;
  const states=[0,...labels.flatMap(p=>[p,0])];
  let a=states.map(()=>-Infinity); a[0]=rows[0][0]; a[1]=rows[0][labels[0]];
  for(let t=1;t<rows.length;t++) a=states.map((p,s)=> {
    let v=a[s]; if(s>0)v=add(v,a[s-1]);
    if(s>1&&p!==0&&p!==states[s-2])v=add(v,a[s-2]);
    return v+rows[t][p];
  });
  return add(a.at(-1)!,a.at(-2)!);
}
export function boundedComparison(model: BoundedModel, features: number[][], c: SequenceEvidenceCandidate) {
  if(c.writtenParts.join("")!==c.written||c.rivalParts.join("")!==c.rival) return null;
  const {written,rival}=c;
  if(!written||!rival||written===rival||!Number.isInteger(c.fromFrame)||c.fromFrame<0||!Number.isInteger(c.toFrame)||
    !features.length||features.length!==c.toFrame-c.fromFrame||
    features.some(r=>r.length!==model.width||r.some(v=>!Number.isFinite(v)))) return null;
  const ids=(s:string)=>[...s].map(p=>model.tokens.indexOf(p)+1);
  const pair=(h:CtcHead)=> {
    const rows=readout(h,features,model.center,model.scale);
    return [sequenceCtcScore(rows,ids(written)),sequenceCtcScore(rows,ids(rival))];
  };
  const [boundedWritten,boundedRival]=pair(model.bounded),[controlWritten,controlRival]=pair(model.control);
  const [enlargedBoundedWritten,enlargedBoundedRival]=pair(model.enlargedBounded);
  const [enlargedControlWritten,enlargedControlRival]=pair(model.enlargedControl);
  const scores={boundedWritten,boundedRival,controlWritten,controlRival,
    enlargedBoundedWritten,enlargedBoundedRival,enlargedControlWritten,enlargedControlRival};
  return {...c,...scores,veto:Object.values(scores).every(Number.isFinite)&&
    scores.boundedWritten>scores.boundedRival&&scores.controlWritten<scores.controlRival&&
    scores.enlargedBoundedWritten>scores.enlargedBoundedRival&&scores.enlargedControlWritten<scores.enlargedControlRival};
}
/** Proper categorical Brier loss over distinct named sequences and the remaining mass.
 * The target is the first named transcript; probabilities are never conditioned on the named set. */
export function sequenceBrier(probabilities:number[]):number {
  if(!probabilities.length||probabilities.some(p=>!Number.isFinite(p)||p<0||p>1))
    throw new Error("Invalid sequence probabilities");
  const remaining=1-probabilities.reduce((a,b)=>a+b,0);
  if(remaining < -1e-10)throw new Error("Named sequences exceed total probability");
  return (1-probabilities[0])**2+probabilities.slice(1).reduce((sum,p)=>sum+p*p,0)+remaining*remaining;
}
/** Full-session separate streaming state. The final linear input includes activation quantization.
 * Both arms retain the same frame clock; primary rows and caches are untouched. */
export async function boundedObservations(backend: InferenceBackend, manifest: ModelManifest, meta: BoundedMetadata, audio: Float32Array) {
  const features = new Fbank().push(audio), logProbs: number[][] = [], encoder: number[][] = [];
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
      encoder.push(Array.from(h.data.slice(t*meta.width,(t+1)*meta.width),Number));
      logProbs.push(Array.from(out.log_probs.data.slice(t*vocab,(t+1)*vocab),Number));
    }
  }
  return {encoder,logProbs};
}

export function carryBoundedEvidence(chunks:PhonemeEvidence[],certificates:BoundedSequenceEvidence[]):PhonemeEvidence[] {
  if(!certificates.length)return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,boundedSequenceVetoes:certificates})});
}
export function boundedVerdictRemovals(live:CheckerEvent[],baseline:CheckerEvent[],reviewed:CheckerEvent[]):CheckerEvent[] {
  const key=(e:CheckerEvent)=>JSON.stringify([e.id,e.timeMs,e.location,e.expected,e.observed]);
  const kept=new Set(reviewed.map(key));
  const removed=new Set(baseline.filter(e=>e.errorClass==="wrong_word"&&!kept.has(key(e))).map(key));
  return live.filter(e=>e.errorClass==="wrong_word"&&removed.has(key(e)));
}
export async function addBoundedEvidence(scorer:{model:BoundedModel;backend:InferenceBackend}|undefined,manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free"):Promise<PhonemeEvidence[]> {
  if(!scorer||!audio.length)return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence:c=>{candidates.set(JSON.stringify(c),c);}});
  for(const e of insertionCells(chunks))checker.push(e,tracker.push(trackerInput(e)));checker.end(tracker.end());
  if(!candidates.size)return chunks;
  const features=await boundedObservations(scorer.backend,manifest,scorer.model,audio),certificates:BoundedSequenceEvidence[]=[];
  for(const c of candidates.values()) {
    if(c.fromFrame<0||c.toFrame>features.encoder.length)continue;
    const result=boundedComparison(scorer.model,features.encoder.slice(c.fromFrame,c.toFrame),c);
    if(result?.veto)certificates.push(result);
  }
  return carryBoundedEvidence(chunks,certificates);
}
