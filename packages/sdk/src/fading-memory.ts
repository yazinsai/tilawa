import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, FadingMemoryEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export function eligibleFadingMemoryTake(r: {kind?:string;split?:string;speaker?:string;exclude?:boolean|string|null;excluded?:boolean|string|null;scored?:boolean;events?:unknown[];non_errors?:unknown[]}, excluded:ReadonlySet<string>):boolean {
  return r.kind === "none" && r.split === "train" && !!r.speaker && !excluded.has(r.speaker) &&
    !r.exclude && !r.excluded && r.scored !== false && !r.events?.length && !r.non_errors?.length;
}
export interface FadingMemoryMetadata {
  schema: 1; inventory: string; tokens: string; width: number;
  featureOutput: string; logitOutput: string; primarySha256: string;
}
export interface PhoneReadout { weight: number[][]; bias: number[] }
export interface FadingMemoryModel extends FadingMemoryMetadata {
  center: number[]; scale: number[]; input: number[][]; rho: number;
  memory: PhoneReadout; instant: PhoneReadout; shuffled: PhoneReadout;
}
export function validateFadingMemoryModel(value: unknown, manifest: ModelManifest): FadingMemoryModel {
  const m=value as FadingMemoryModel;
  const vector=(r:unknown,n:number):r is number[]=>Array.isArray(r)&&r.length===n&&r.every(Number.isFinite);
  const matrix=(r:unknown,n:number,d:number)=>Array.isArray(r)&&r.length===n&&r.every(v=>vector(v,d));
  if(!m||m.schema!==1||m.inventory!==manifest.inventory||m.tokens!==manifest.tokens||
    m.primarySha256!==manifest.sha256||!Number.isSafeInteger(m.width)||m.width<=0||
    !m.featureOutput||!m.logitOutput||!vector(m.center,m.width)||!vector(m.scale,m.width)||m.scale.some(v=>v<=0)||
    !Number.isFinite(m.rho)||m.rho<0||m.rho>=1||!Array.isArray(m.input)||!m.input.length||
    !matrix(m.input,m.input.length,m.width)) throw new Error("Invalid fading-memory metadata or noncontractive reservoir");
  for(const h of [m.memory,m.instant,m.shuffled])if(!h||!matrix(h.weight,m.tokens.length+1,m.width+m.input.length)||
    !vector(h.bias,m.tokens.length+1)) throw new Error("Invalid or unequal-capacity fading-memory phone readouts");
  return m;
}
/** tanh is 1-Lipschitz and the cyclic shift is orthogonal: state contraction is at most rho. */
export class Reservoir {
  private state: number[];
  constructor(private input: number[][], private rho: number) {
    if(!Number.isFinite(rho)||rho<0||rho>=1||!input.length)throw new Error("Noncontractive or empty reservoir");
    this.state=input.map(()=>0);
  }
  push(x:number[]):number[] {
    this.state=this.input.map((w,h)=>Math.tanh(w.reduce((s,v,d)=>s+v*x[d],0)+
      this.rho*this.state[(h+this.state.length-1)%this.state.length]));
    return this.state.slice();
  }
}
/** Identical text-blind Fisher-Yates permutation in training and runtime. */
export function shuffledOrder(n:number):number[] {
  const order=Array.from({length:n},(_,i)=>i);let seed=1729;
  for(let i=n-1;i>0;i--) { seed=(Math.imul(seed,1664525)+1013904223)>>>0;
    const j=seed%(i+1);[order[i],order[j]]=[order[j],order[i]]; }
  return order;
}
export function readouts(model:FadingMemoryModel,raw:number[][]):number[][][] {
  const x=raw.map(r=>r.map((v,d)=>(v-model.center[d])/model.scale[d]));
  const reservoir=new Reservoir(model.input,model.rho), shuffled=new Reservoir(model.input,model.rho);
  const memories=x.map(v=>reservoir.push(v)), shuffledMemories:number[][]=Array(x.length);
  for(const i of shuffledOrder(x.length))shuffledMemories[i]=shuffled.push(x[i]);
  const instant=x.map(v=>model.input.map(w=>Math.tanh(w.reduce((s,a,d)=>s+a*v[d],0))));
  return [model.memory,model.instant,model.shuffled].map((head,a)=>x.map((v,t)=> {
    const row=[...v,...[memories,instant,shuffledMemories][a][t]];
    const logits=head.weight.map((w,p)=>w.reduce((s,v,d)=>s+v*row[d],head.bias[p]));
    const max=Math.max(...logits),normal=max+Math.log(logits.reduce((s,v)=>s+Math.exp(v-max),0));
    return logits.map(v=>v-normal);
  }));
}
const add=(a:number,b:number):number=> {
  const m=Math.max(a,b); return Number.isFinite(m)?m+Math.log(Math.exp(a-m)+Math.exp(b-m)):-Infinity;
};
/** Complete ordinary CTC forward sum, with the same frames and topology in every arm. */
export function ctcScore(rows:number[][],labels:number[]):number {
  if(!rows.length||!labels.length||labels.some(p=>p<=0||p>=rows[0].length))return -Infinity;
  const states=[0,...labels.flatMap(p=>[p,0])];
  let a=states.map(()=>-Infinity);a[0]=rows[0][0];a[1]=rows[0][labels[0]];
  for(let t=1;t<rows.length;t++)a=states.map((p,s)=> {
    let v=a[s];if(s>0)v=add(v,a[s-1]);
    if(s>1&&p!==0&&p!==states[s-2])v=add(v,a[s-2]);return v+rows[t][p];
  });
  return add(a.at(-1)!,a.at(-2)!);
}
/** Inputs are complete-recording readouts. Slice only after all reservoir state has been carried. */
export function memoryComparison(model:FadingMemoryModel,rows:number[][][],c:SequenceEvidenceCandidate) {
  if(!c.written||!c.rival||c.written===c.rival||!Number.isInteger(c.fromFrame)||c.fromFrame<0||
    !Number.isInteger(c.toFrame)||c.toFrame<=c.fromFrame||rows.length!==3||rows.some(a=>a.length<c.toFrame))return null;
  const ids=(s:string)=>[...s].map(p=>model.tokens.indexOf(p)+1);
  const pairs=rows.map(a=>[ctcScore(a.slice(c.fromFrame,c.toFrame),ids(c.written)),ctcScore(a.slice(c.fromFrame,c.toFrame),ids(c.rival))]);
  const scores={memoryWritten:pairs[0][0],memoryRival:pairs[0][1],instantWritten:pairs[1][0],instantRival:pairs[1][1],
    shuffledWritten:pairs[2][0],shuffledRival:pairs[2][1]};
  return {...c,...scores,veto:Object.values(scores).every(Number.isFinite)&&scores.memoryWritten>scores.memoryRival&&
    scores.instantWritten<scores.instantRival&&scores.shuffledWritten<scores.shuffledRival};
}
/** Full-session separate streaming state. The final linear input includes activation quantization.
 * Both arms retain the same frame clock; primary rows and caches are untouched. */
export async function fadingMemoryObservations(backend: InferenceBackend, manifest: ModelManifest, meta: FadingMemoryMetadata, audio: Float32Array) {
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


export function carryFadingMemoryEvidence(chunks:PhonemeEvidence[],certificates:FadingMemoryEvidence[]):PhonemeEvidence[] {
  if(!certificates.length)return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,fadingMemoryVetoes:certificates})});
}
export function memoryVerdictRemovals(live:CheckerEvent[],baseline:CheckerEvent[],reviewed:CheckerEvent[]):CheckerEvent[] {
  const key=(e:CheckerEvent)=>JSON.stringify([e.id,e.timeMs,e.location,e.expected,e.observed]);
  const kept=new Set(reviewed.map(key));
  const removed=new Set(baseline.filter(e=>e.errorClass==="wrong_word"&&!kept.has(key(e))).map(key));
  return live.filter(e=>e.errorClass==="wrong_word"&&removed.has(key(e)));
}
export async function addFadingMemoryEvidence(scorer:{model:FadingMemoryModel;backend:InferenceBackend}|undefined,manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(comparison:ReturnType<typeof memoryComparison>,c:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if(!scorer||!audio.length)return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence:c=>{candidates.set(JSON.stringify(c),c);return true;}});
  for(const e of insertionCells(chunks))checker.push(e,tracker.push(trackerInput(e)));checker.end(tracker.end());
  if(!candidates.size)return chunks;
  const features=await fadingMemoryObservations(scorer.backend,manifest,scorer.model,audio);
  const rows=readouts(scorer.model,features.encoder),certificates:FadingMemoryEvidence[]=[];
  for(const c of candidates.values()) {
    const result=memoryComparison(scorer.model,rows,c);inspect?.(result,c);
    if(result?.veto)certificates.push(result);
  }
  return carryFadingMemoryEvidence(chunks,certificates);
}
