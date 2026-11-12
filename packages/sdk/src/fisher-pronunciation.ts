import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, FisherPronunciationEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export function eligibleFisherTake(r: {kind?:string;split?:string;speaker?:string;exclude?:boolean|string|null;excluded?:boolean|string|null;scored?:boolean;events?:unknown[];non_errors?:unknown[]}, excluded:ReadonlySet<string>):boolean {
  return r.kind === "none" && r.split === "train" && !!r.speaker && !excluded.has(r.speaker) &&
    !r.exclude && !r.excluded && r.scored !== false && !r.events?.length && !r.non_errors?.length;
}
export interface FisherMetadata {
  schema: 1; inventory: string; tokens: string; width: number;
  featureOutput: string; logitOutput: string; primarySha256: string; parityWindows: number;
}
export interface FisherGaussian { mean:number[]; variance:number[]; count:number }
export interface FisherArm { phones:FisherGaussian[] }
export interface FisherModel extends FisherMetadata {
  center:number[]; projection:number[][]; dimensions:number; maxDuration:number;
  mixture:{weight:number;mean:number[];variance:number[]}[];
  fisherProjection:number[][]; controlProjection:number[][];
  fisher:FisherArm; control:FisherArm;
}
const logadd=(a:number,b:number):number=> {
  const m=Math.max(a,b);return Number.isFinite(m)?m+Math.log(Math.exp(a-m)+Math.exp(b-m)):-Infinity;
};
const project=(x:readonly number[],matrix:number[][]):number[]=>matrix.map(row=>row.reduce((s,v,d)=>s+v*x[d],0));
function normalize(x:number[]):number[] {
  const y=x.map(v=>Math.sign(v)*Math.sqrt(Math.abs(v))),norm=Math.hypot(...y);
  return y.map(v=>norm>0?v/norm:0);
}
/** Exact normalized mean/variance likelihood gradients of a diagonal background GMM.
 * Responsibilities depend only on acoustics; no pronunciation participates in this map. */
export function fisherFrame(model:Pick<FisherModel,"mixture">,x:number[]):number[] {
  const logs=model.mixture.map(g=>Math.log(g.weight)-.5*g.mean.reduce((s,m,d)=>s+Math.log(2*Math.PI*g.variance[d])+(x[d]-m)**2/g.variance[d],0));
  const z=logs.reduce(logadd,-Infinity),out:number[]=[];
  model.mixture.forEach((g,k)=> {
    const q=Math.exp(logs[k]-z),u=x.map((v,d)=>(v-g.mean[d])/Math.sqrt(g.variance[d]));
    out.push(...u.map(v=>q*v/Math.sqrt(g.weight)),...u.map(v=>q*(v*v-1)/Math.sqrt(2*g.weight)));
  });
  return out;
}
export function fisherSegment(model:FisherModel,raw:number[][]):{fisher:number[];control:number[]} {
  if (!raw.length) throw new Error("Empty acoustic segment");
  const x=raw.map(r=>project(r.map((v,d)=>v-model.center[d]),model.projection));
  const d=x[0].length,mean=Array(d).fill(0) as number[],gradient=Array(2*model.mixture.length*d).fill(0) as number[];
  for(const r of x) {
    r.forEach((v,j)=>mean[j]+=v/x.length);
    fisherFrame(model,r).forEach((v,j)=>gradient[j]+=v/x.length);
  }
  const pooled=mean.slice();
  for(let i=0;i<d;i++)for(let j=0;j<=i;j++)pooled.push(x.reduce((s,r)=>s+(r[i]-mean[i])*(r[j]-mean[j]),0)/x.length);
  // The fixed maps give both Gaussian readouts exactly the same parameter count.
  return {fisher:normalize(project(gradient,model.fisherProjection)),control:normalize(project(pooled,model.controlProjection))};
}
function segmentLogProb(arm:FisherArm,x:number[]):number[] {
  const values=arm.phones.map(g=>-.5*x.reduce((s,v,d)=>s+Math.log(2*Math.PI*g.variance[d])+(v-g.mean[d])**2/g.variance[d],0));
  const z=values.reduce(logadd,-Infinity);
  return values.map(v=>v-z+Math.log(values.length));
}
/** Semi-Markov complete pronunciation. Every phone consumes a nonempty interval;
 * intervening background intervals may be empty. All legal boundaries are summed,
 * and the shared uniform duration prior is conditioned on full completion. */
export function fisherBoundaryScore(values:number[][][],ids:number[],maxDuration:number):number {
  const tmax=values.length;
  if(!tmax||!ids.length)return -Infinity;
  let a=Array(tmax+1).fill(-Infinity),prior=a.slice();a[0]=prior[0]=0;
  for(const p of [0,...ids.flatMap(p=>[p,0])]) {
    const b=Array(tmax+1).fill(-Infinity),q=b.slice(),minimum=p===0?0:1;
    for(let end=0;end<=tmax;end++)for(let n=minimum;n<=Math.min(maxDuration,end);n++) {
      const score=n===0?0:values[end-1][n-1]?.[p]??-Infinity;
      const cost=Math.log(maxDuration-minimum+1);
      b[end]=logadd(b[end],a[end-n]+score-cost);q[end]=logadd(q[end],prior[end-n]-cost);
    }
    a=b;prior=q;
  }
  return Number.isFinite(a[tmax])&&Number.isFinite(prior[tmax])?a[tmax]-prior[tmax]:-Infinity;
}
function segmentFeatures(model:FisherModel,raw:number[][]):(start:number,end:number)=>{fisher:number[];control:number[]} {
  const x=raw.map(r=>project(r.map((v,d)=>v-model.center[d]),model.projection));
  const d=model.projection.length,pairs:[number,number][]=[];
  for(let i=0;i<d;i++)for(let j=0;j<=i;j++)pairs.push([i,j]);
  const prefixes=(rows:number[][])=> {
    const out=[rows[0].map(()=>0)];
    for(const row of rows)out.push(row.map((v,j)=>v+out[out.length-1][j]));
    return out;
  };
  const mean=prefixes(x),gradient=prefixes(x.map(r=>fisherFrame(model,r)));
  const covariance=prefixes(x.map(r=>pairs.map(([i,j])=>r[i]*r[j])));
  return (start,end)=> {
    const n=end-start,m=mean[end].map((v,j)=>(v-mean[start][j])/n);
    const g=gradient[end].map((v,j)=>(v-gradient[start][j])/n);
    const c=covariance[end].map((v,j)=>(v-covariance[start][j])/n-m[pairs[j][0]]*m[pairs[j][1]]);
    return {fisher:normalize(project(g,model.fisherProjection)),control:normalize(project([...m,...c],model.controlProjection))};
  };
}
function comparisonScores(model:FisherModel,raw:number[][],sequences:number[][]):{fisher:number[];control:number[]} {
  const values:{fisher:number[][][];control:number[][][]}={fisher:[],control:[]};
  const descriptor=segmentFeatures(model,raw);
  for(let end=1;end<=raw.length;end++) {
    const f:number[][]=[],c:number[][]=[];
    for(let n=1;n<=Math.min(end,model.maxDuration);n++) {
      const v=descriptor(end-n,end);
      f.push(segmentLogProb(model.fisher,v.fisher).map(s=>s*n));
      c.push(segmentLogProb(model.control,v.control).map(s=>s*n));
    }
    values.fisher.push(f);values.control.push(c);
  }
  return {fisher:sequences.map(s=>fisherBoundaryScore(values.fisher,s,model.maxDuration)),
    control:sequences.map(s=>fisherBoundaryScore(values.control,s,model.maxDuration))};
}
export function fisherComparison(model:FisherModel,raw:number[][],candidate:SequenceEvidenceCandidate):FisherPronunciationEvidence|null {
  const sequences=[candidate.written,candidate.rival].map(s=>[...s].map(p=>model.tokens.indexOf(p)+1));
  if(!Number.isInteger(candidate.fromFrame)||candidate.fromFrame<0||candidate.toFrame-candidate.fromFrame!==raw.length||
    !raw.length||raw.some(r=>r.length!==model.width||!r.every(Number.isFinite))||
    sequences.some(s=>!s.length||s.some(p=>p<1||model.fisher.phones[p].count<=0||model.control.phones[p].count<=0)))return null;
  const v=comparisonScores(model,raw,sequences);
  const result={...candidate,fisherWritten:v.fisher[0],fisherRival:v.fisher[1],controlWritten:v.control[0],controlRival:v.control[1]};
  return [result.fisherWritten,result.fisherRival,result.controlWritten,result.controlRival].every(Number.isFinite)?result:null;
}
export function fisherReversal(c:FisherPronunciationEvidence):boolean {
  return [c.fisherWritten,c.fisherRival,c.controlWritten,c.controlRival].every(Number.isFinite)&&
    c.fisherWritten>c.fisherRival&&c.controlWritten<c.controlRival;
}
export function validateFisherModel(value:unknown,manifest:ModelManifest):FisherModel {
  const m=value as FisherModel;
  const vector=(r:unknown,n:number):r is number[]=>Array.isArray(r)&&r.length===n&&r.every(Number.isFinite);
  const matrix=(v:unknown,rows:number,cols:number):v is number[][]=>Array.isArray(v)&&v.length===rows&&v.every(r=>vector(r,cols));
  if(!m||m.schema!==1||m.inventory!==manifest.inventory||m.tokens!==manifest.tokens||m.primarySha256!==manifest.sha256||
    !Number.isSafeInteger(m.width)||m.width<1||!vector(m.center,m.width)||
    !Array.isArray(m.projection)||!m.projection.length||!matrix(m.projection,m.projection.length,m.width)||
    !Number.isSafeInteger(m.dimensions)||m.dimensions<1||!Number.isSafeInteger(m.maxDuration)||m.maxDuration<1||m.maxDuration>128||
    !m.featureOutput||!m.logitOutput||m.parityWindows<8)throw new Error("Invalid Fisher pronunciation model");
  const d=m.projection.length;
  if(!Array.isArray(m.mixture)||!m.mixture.length||m.mixture.some(g=>!g||!Number.isFinite(g.weight)||g.weight<=0||
    !vector(g.mean,d)||!vector(g.variance,d)||g.variance.some(v=>v<=0))||Math.abs(m.mixture.reduce((s,g)=>s+g.weight,0)-1)>1e-6||
    !matrix(m.fisherProjection,m.dimensions,2*d*m.mixture.length)||!matrix(m.controlProjection,m.dimensions,d+d*(d+1)/2))throw new Error("Invalid Fisher background model");
  for(const arm of [m.fisher,m.control])if(!arm||!Array.isArray(arm.phones)||arm.phones.length!==m.tokens.length+1||
    arm.phones.some(g=>!g||!vector(g.mean,m.dimensions)||!vector(g.variance,m.dimensions)||g.variance.some(v=>v<=0)||
      !Number.isFinite(g.count)||g.count<0))throw new Error("Invalid Fisher pronunciation readout");
  return m;
}
export function fisherVerdictRemovals(live:CheckerEvent[],baseline:CheckerEvent[],reviewed:CheckerEvent[]):CheckerEvent[] {
  const identity=(e:CheckerEvent)=>JSON.stringify([e.id,e.timeMs,e.location,e.expected,e.observed]);
  const kept=new Set(reviewed.map(identity));
  const vetoed=new Set(baseline.filter(e=>e.errorClass==="wrong_word"&&!kept.has(identity(e))).map(identity));
  return live.filter(e=>e.errorClass==="wrong_word"&&vetoed.has(identity(e)));
}
export function carryFisherEvidence(chunks:PhonemeEvidence[],certificates:FisherPronunciationEvidence[]):PhonemeEvidence[] {
  if(!certificates.length)return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,fisherPronunciationVetoes:certificates})});
}
/** Replay a separate copy of the exact streaming encoder, including its caches and padding. */
export async function fisherObservations(backend:InferenceBackend,manifest:ModelManifest,meta:FisherMetadata,audio:Float32Array) {
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
    if(!Number.isSafeInteger(n)||n<=0||!h||h.type!=="float32"||h.data.length!==n*meta.width)throw new Error("Invalid frozen encoder geometry");
    for(let t=0;t<n&&raw.length<total;t++) {
      raw.push(Array.from(h.data.slice(t*meta.width,(t+1)*meta.width),Number));
      logProbs.push(Array.from(out.log_probs.data.slice(t*vocab,(t+1)*vocab),Number));
    }
  }
  return {raw,logProbs};
}
export async function addFisherEvidence(scorer:{model:FisherModel;backend:InferenceBackend}|undefined,manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:FisherPronunciationEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if(!scorer||!audio.length)return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>(),tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for(const e of insertionCells(chunks))checker.push(e,tracker.push(trackerInput(e)));checker.end(tracker.end());
  if(!candidates.size)return chunks;
  const features=await fisherObservations(scorer.backend,manifest,scorer.model,audio),certificates:FisherPronunciationEvidence[]=[];
  for(const c of candidates.values()) {
    const result=c.fromFrame>=0&&c.toFrame<=features.raw.length?fisherComparison(scorer.model,features.raw.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(result,c);if(result&&fisherReversal(result))certificates.push(result);
  }
  return carryFisherEvidence(chunks,certificates);
}
