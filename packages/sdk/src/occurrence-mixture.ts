import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, OccurrenceMixtureEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface OccurrenceMetadata {
  schema: 1; inventory: string; tokens: string; width: number;
  featureOutput: string; primarySha256: string;
}
export interface PhoneMixture {
  weights: number[]; means: number[][]; variances: number[][]; count: number;
}
export interface OccurrenceModel extends OccurrenceMetadata {
  dimensions: number; center: number[]; projection: number[][]; phones: PhoneMixture[];
}
const logadd = (a:number,b:number):number => {
  const m=Math.max(a,b);
  return Number.isFinite(m) ? m+Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
const sum = (values:number[]) => values.reduce(logadd,-Infinity);

/** Marginalize all complete CTC paths. A phone state has one latent component per
 * occurrence: self-loops retain it; entries sample its prior exactly once.
 * Blank emissions switch in both arms. The shared timing prior is conditioned on
 * completion, so neither pronunciation gains from having more boundary paths. */
export function occurrenceBoundaryScore(values:number[][][], weights:number[][], ids:number[], tied:boolean):number {
  if (!values.length || !ids.length || ids.some(p=>p<=0||p>=weights.length)) return -Infinity;
  const labels=[0,...ids.flatMap(p=>[p,0])];
  const edges=labels.map((p,s)=>[s,...(s+1<labels.length?[s+1]:[]),
    ...(s+2<labels.length&&labels[s+2]!==0&&labels[s+2]!==p?[s+2]:[])]);
  const degree=edges.map(e=>Math.log(e.length));
  const priors=weights.map(w=>w.map(Math.log));
  const mixed=(t:number,p:number)=>sum(values[t][p].map((v,k)=>v+priors[p][k]));
  const emission=(t:number,s:number,k:number)=>tied&&labels[s]!==0 ? values[t][labels[s]][k] : mixed(t,labels[s]);
  const entry=(s:number,k:number)=>tied&&labels[s]!==0 ? priors[labels[s]][k] : 0;
  let a=labels.map(p=>Array(tied&&p!==0?weights[p].length:1).fill(-Infinity) as number[]);
  let timing=labels.map(()=>-Infinity);
  for (const s of [0,1]) { a[s]=a[s].map((_,k)=>entry(s,k)+emission(0,s,k));timing[s]=0; }
  for (let t=1;t<values.length;t++) {
    const b=a.map(row=>row.map(()=>-Infinity)), q=timing.map(()=>-Infinity);
    for(let s=0;s<labels.length;s++) for(const next of edges[s]) {
      q[next]=logadd(q[next],timing[s]-degree[s]);
      if(next===s) for(let k=0;k<a[s].length;k++) b[s][k]=logadd(b[s][k],a[s][k]-degree[s]);
      else {
        const total=sum(a[s])-degree[s];
        for(let k=0;k<b[next].length;k++) b[next][k]=logadd(b[next][k],total+entry(next,k));
      }
    }
    a=b.map((row,s)=>row.map((v,k)=>v+emission(t,s,k)));timing=q;
  }
  const z=logadd(sum(a.at(-1)!),sum(a.at(-2)!)), prior=logadd(timing.at(-1)!,timing.at(-2)!);
  return Number.isFinite(z)&&Number.isFinite(prior)?z-prior:-Infinity;
}
export function occurrenceReversal(c:OccurrenceMixtureEvidence):boolean {
  return [c.tiedWritten,c.tiedRival,c.switchingWritten,c.switchingRival].every(Number.isFinite)&&
    c.tiedWritten>c.tiedRival&&c.switchingWritten<c.switchingRival;
}
export function occurrenceComparison(model:OccurrenceModel,raw:number[][],c:SequenceEvidenceCandidate):OccurrenceMixtureEvidence|null {
  const ids=[c.written,c.rival].map(s=>[...s].map(p=>model.tokens.indexOf(p)+1));
  if(!Number.isInteger(c.fromFrame)||c.fromFrame<0||!Number.isInteger(c.toFrame)||c.toFrame-c.fromFrame!==raw.length||
    !raw.length||raw.some(r=>r.length!==model.width||!r.every(Number.isFinite))||
    ids.some(s=>!s.length||s.some(p=>p<1||model.phones[p].count<=0)))return null;
  const used=new Set([0,...ids.flat()]);
  const values=raw.map(r=> {
    const x=model.projection.map(row=>row.reduce((v,a,d)=>v+a*(r[d]-model.center[d]),0));
    return model.phones.map((phone,p)=>!used.has(p)?phone.weights.map(()=>-Infinity):phone.means.map((mean,k)=>
      -.5*mean.reduce((v,m,d)=>v+Math.log(2*Math.PI*phone.variances[k][d])+(x[d]-m)**2/phone.variances[k][d],0)));
  });
  const weights=model.phones.map(p=>p.weights);
  const scores=(tied:boolean)=>ids.map(s=>occurrenceBoundaryScore(values,weights,s,tied));
  const a=scores(true),b=scores(false);
  const result={...c,tiedWritten:a[0],tiedRival:a[1],switchingWritten:b[0],switchingRival:b[1]};
  return [a[0],a[1],b[0],b[1]].every(Number.isFinite)?result:null;
}
export function validateOccurrenceModel(value:unknown,manifest:ModelManifest):OccurrenceModel {
  const m=value as OccurrenceModel;
  const vector=(v:unknown,n:number):v is number[]=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite);
  const matrix=(v:unknown,n:number,d:number):v is number[][]=>Array.isArray(v)&&v.length===n&&v.every(r=>vector(r,d));
  if(!m||m.schema!==1||m.tokens!==manifest.tokens||m.inventory!==manifest.inventory||m.primarySha256!==manifest.sha256||
    !m.featureOutput||!Number.isSafeInteger(m.width)||m.width<=0||!Number.isSafeInteger(m.dimensions)||
    m.dimensions<=0||m.dimensions>m.width||!vector(m.center,m.width)||!matrix(m.projection,m.dimensions,m.width)||
    !Array.isArray(m.phones)||m.phones.length!==m.tokens.length+1)throw new Error("Invalid occurrence mixture model");
  const k=m.phones[0]?.weights?.length;
  if(!Number.isSafeInteger(k)||k<1||k>8)throw new Error("Invalid occurrence mixture capacity");
  for(const p of m.phones)if(!p||!vector(p.weights,k)||p.weights.some(v=>v<=0)||
    Math.abs(p.weights.reduce((a,b)=>a+b,0)-1)>1e-8||!matrix(p.means,k,m.dimensions)||
    !matrix(p.variances,k,m.dimensions)||p.variances.some(r=>r.some(v=>v<=0))||
    !Number.isFinite(p.count)||p.count<0)throw new Error("Invalid phone mixture");
  return m;
}

/** None labels are canonical supervision only when the recording is usable and has no edits/repeats. */
export function eligibleOccurrenceTake(r: {kind?:string;split?:string;speaker?:string;exclude?:boolean|string|null;excluded?:boolean|string|null;scored?:boolean;events?:unknown[];non_errors?:unknown[]}, excluded:ReadonlySet<string>):boolean {
  return r.kind === "none" && r.split === "train" && !!r.speaker && !excluded.has(r.speaker) &&
    !r.exclude && !r.excluded && r.scored !== false && !r.events?.length && !r.non_errors?.length;
}


/** Only retract the exact live occurrence removed by an otherwise identical evidence replay.
 * Repeated words and later graph confirmations may share a spelling or issue id. */
export function occurrenceVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity = (e: CheckerEvent) => JSON.stringify([e.id, e.errorClass, e.timeMs, e.location, e.expected, e.observed]);
  const kept = new Set(reviewed.map(identity));
  const vetoed = new Set(baseline.filter(e => e.errorClass === "wrong_word" && !kept.has(identity(e))).map(identity));
  return live.filter(e => e.errorClass === "wrong_word" && vetoed.has(identity(e)));
}
/** Carry auxiliary judgments without changing any primary token or posterior. */
export function carryOccurrenceEvidence(chunks: PhonemeEvidence[], certificates: OccurrenceMixtureEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e,i) => i !== first ? e : {...e, phonemes:e.phonemes.map((p,j) => j ? p : {
    ...p, occurrenceMixtureVetoes:certificates,
  })});
}

/** Full-session separate streaming state. The final linear input includes activation quantization.
 * The shared emissions retain the same frame clock; primary rows and caches are untouched. */
export async function occurrenceObservations(backend: InferenceBackend, manifest: ModelManifest, meta: OccurrenceMetadata, audio: Float32Array) {
  const features = new Fbank().push(audio), logProbs: number[][] = [], occurrence: number[][] = [];
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
    const h=out[meta.featureOutput];
    if(!Number.isSafeInteger(n)||n<=0||!h||h.type!=="float32"||h.data.length!==n*meta.width)
      throw new Error("Invalid frozen feature geometry");
    for(let t=0;t<n && logProbs.length<total;t++) {
      occurrence.push(Array.from(h.data.slice(t*meta.width,(t+1)*meta.width),Number));
      logProbs.push(Array.from(out.log_probs.data.slice(t*vocab,(t+1)*vocab),Number));
    }
  }
  return {features:occurrence,logProbs};
}
export async function addOccurrenceEvidence(scorer: {model:OccurrenceModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:OccurrenceMixtureEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if (!scorer||!audio.length) return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>(),cells=insertionCells(chunks);
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e))); checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const features=await occurrenceObservations(scorer.backend,manifest,scorer.model,audio),certificates:OccurrenceMixtureEvidence[]=[];
  for (const c of candidates.values()) {
    const result=c.fromFrame>=0&&c.toFrame<=features.features.length ? occurrenceComparison(scorer.model,
      features.features.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(result,c); if (result&&occurrenceReversal(result)) certificates.push(result);
  }
  return carryOccurrenceEvidence(chunks,certificates);
}
