import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, MultigramEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

/** None labels are canonical supervision only when the recording is usable and has no edits/repeats. */
export function eligibleMultigramTake(r: {kind?:string;split?:string;speaker?:string;exclude?:boolean|string|null;excluded?:boolean|string|null;scored?:boolean;events?:unknown[];non_errors?:unknown[]}, excluded:ReadonlySet<string>):boolean {
  return r.kind === "none" && r.split === "train" && !!r.speaker && !excluded.has(r.speaker) &&
    !r.exclude && !r.excluded && r.scored !== false && !r.events?.length && !r.non_errors?.length;
}


export interface MultigramMetadata {
  schema: 1; inventory: string; tokens: string; width: number;
  featureOutput: string; logitOutput: string; primarySha256: string;
}
export interface CtcHead { input: number[][]; inputBias: number[]; output: number[][]; outputBias: number[] }
export interface MultigramModel extends MultigramMetadata {
  units: string[]; center: number[]; scale: number[]; multigram: CtcHead; control: CtcHead;
}
export function validateMultigramModel(value: unknown, manifest: ModelManifest): MultigramModel {
  const m=value as MultigramModel;
  const vector=(r:unknown,n:number):r is number[]=>Array.isArray(r)&&r.length===n&&r.every(Number.isFinite);
  const matrix=(r:unknown,n:number,d:number)=>Array.isArray(r)&&r.length===n&&r.every(v=>vector(v,d));
  if(!m||m.schema!==1||m.inventory!==manifest.inventory||m.tokens!==manifest.tokens||
    m.primarySha256!==manifest.sha256||!Number.isSafeInteger(m.width)||m.width<=0||
    !m.featureOutput||!m.logitOutput||!vector(m.center,m.width)||!vector(m.scale,m.width)||m.scale.some(v=>v<=0))
    throw new Error("Invalid multigram readout metadata");
  if (!Array.isArray(m.units) || new Set(m.units).size !== m.units.length ||
    [...m.tokens].some((p,i)=>m.units[i]!==p) || m.units.some(u=>!u || [...u].length>3 || [...u].some(p=>!m.tokens.includes(p))))
    throw new Error("Invalid multigram inventory");
  const hidden=m.multigram?.inputBias?.length;
  if(!Number.isSafeInteger(hidden)||hidden<=0)throw new Error("Invalid multigram head width");
  for(const h of [m.multigram,m.control])if(!h||!matrix(h.input,hidden,m.width)||!vector(h.inputBias,hidden)||
    !matrix(h.output,m.units.length+1,hidden)||!vector(h.outputBias,m.units.length+1))
    throw new Error("Invalid or unequal-capacity CTC heads");
  return m;
}
export function readout(head: CtcHead, raw: number[][], center: number[], scale: number[]): number[][] {
  return raw.map(row => {
    const x=row.map((v,d)=>(v-center[d])/scale[d]);
    const hidden=head.input.map((w,h)=>Math.tanh(w.reduce((s,v,d)=>s+v*x[d],head.inputBias[h])));
    const logits=head.output.map((w,p)=>w.reduce((s,v,h)=>s+v*hidden[h],head.outputBias[p]));
    const max=Math.max(...logits), normal=max+Math.log(logits.reduce((s,v)=>s+Math.exp(v-max),0));
    return logits.map(v=>v-normal);
  });
}
const add=(a:number,b:number):number=> {
  const m=Math.max(a,b); return Number.isFinite(m)?m+Math.log(Math.exp(a-m)+Math.exp(b-m)):-Infinity;
};
/** Sum every CTC path whose collapsed units concatenate to the complete pronunciation.
 * A state records the consumed phone prefix and the last unit (or blank). Repeating
 * a nonblank keeps that state; advancing through the same unit requires a blank. */
export function multigramCtcScore(rows: number[][], pronunciation: string, units: string[]): number {
  if (!rows.length || !pronunciation || units.some(u=>!u) || new Set(units).size!==units.length ||
    rows.some(r=>r.length!==units.length+1 || r.some(v=>Number.isNaN(v)||v===Infinity))) return -Infinity;
  const phones=[...pronunciation], sequences=units.map(u=>[...u]);
  const edges: {from:number;to:number;unit:number}[]=[];
  for(let from=0;from<phones.length;from++) sequences.forEach((u,i)=> {
    if(u.every((p,k)=>phones[from+k]===p)) edges.push({from,to:from+u.length,unit:i+1});
  });
  const ending=Array.from({length:phones.length+1},()=>[] as number[]);
  edges.forEach((e,i)=>ending[e.to].push(i));
  let blanks=ending.map(()=>-Infinity), active=edges.map(()=>-Infinity); blanks[0]=0;
  for(const row of rows) {
    const nextBlank=blanks.map((v,j)=>ending[j].reduce((s,i)=>add(s,active[i]),v)+row[0]);
    const nextActive=edges.map((e,i)=> {
      let v=add(active[i],blanks[e.from]);
      for(const k of ending[e.from]) if(edges[k].unit!==e.unit)v=add(v,active[k]);
      return v+row[e.unit];
    });
    blanks=nextBlank; active=nextActive;
  }
  return ending[phones.length].reduce((s,i)=>add(s,active[i]),blanks[phones.length]);
}
export function multigramComparison(model: MultigramModel, features: number[][], c: SequenceEvidenceCandidate) {
  const {written,rival}=c;
  if(c.writtenParts.join("")!==written||c.rivalParts.join("")!==rival||!written||!rival||written===rival||
    [...written+rival].some(p=>!model.tokens.includes(p))||!Number.isInteger(c.fromFrame)||c.fromFrame<0||
    !Number.isInteger(c.toFrame)||!features.length||features.length!==c.toFrame-c.fromFrame||
    features.some(r=>r.length!==model.width||r.some(v=>!Number.isFinite(v)))) return null;
  const a=readout(model.multigram,features,model.center,model.scale),b=readout(model.control,features,model.center,model.scale);
  // The control has exactly the same number of parameters; extra outputs are
  // negative classes, never labels. Both scorers use the same normalized softmax.
  const single=(rows:number[][],s:string)=>multigramCtcScore(rows.map(r=>r.slice(0,model.tokens.length+1)),s,[...model.tokens]);
  const scores={multigramWritten:multigramCtcScore(a,written,model.units),multigramRival:multigramCtcScore(a,rival,model.units),
    controlWritten:single(b,written),controlRival:single(b,rival)};
  return {...c,...scores,veto:Object.values(scores).every(Number.isFinite)&&
    scores.multigramWritten>scores.multigramRival&&scores.controlWritten<scores.controlRival};
}
/** Full-session separate streaming state. The final linear input includes activation quantization.
 * Both arms retain the same frame clock; primary rows and caches are untouched. */
export async function multigramObservations(backend: InferenceBackend, manifest: ModelManifest, meta: MultigramMetadata, audio: Float32Array) {
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

export function carryMultigramEvidence(chunks:PhonemeEvidence[],certificates:MultigramEvidence[]):PhonemeEvidence[] {
  if(!certificates.length)return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,multigramVetoes:certificates})});
}
export function multigramVerdictRemovals(live:CheckerEvent[],baseline:CheckerEvent[],reviewed:CheckerEvent[]):CheckerEvent[] {
  const key=(e:CheckerEvent)=>JSON.stringify([e.id,e.timeMs,e.location,e.expected,e.observed]);
  const kept=new Set(reviewed.map(key));
  const removed=new Set(baseline.filter(e=>e.errorClass==="wrong_word"&&!kept.has(key(e))).map(key));
  return live.filter(e=>e.errorClass==="wrong_word"&&removed.has(key(e)));
}
export async function addMultigramEvidence(scorer:{model:MultigramModel;backend:InferenceBackend}|undefined,manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free"):Promise<PhonemeEvidence[]> {
  if(!scorer||!audio.length)return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence:c=>{candidates.set(JSON.stringify(c),c);}});
  for(const e of insertionCells(chunks))checker.push(e,tracker.push(trackerInput(e)));checker.end(tracker.end());
  if(!candidates.size)return chunks;
  const features=await multigramObservations(scorer.backend,manifest,scorer.model,audio),certificates:MultigramEvidence[]=[];
  for(const c of candidates.values()) {
    if(c.fromFrame<0||c.toFrame>features.encoder.length)continue;
    const result=multigramComparison(scorer.model,features.encoder.slice(c.fromFrame,c.toFrame),c);
    if(result?.veto)certificates.push(result);
  }
  return carryMultigramEvidence(chunks,certificates);
}
