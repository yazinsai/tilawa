import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, PreactivationResidualEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

/** None labels are canonical supervision only when the recording is usable and has no edits/repeats. */
export function eligibleResidualTake(r: {kind?:string;split?:string;speaker?:string;exclude?:boolean;excluded?:boolean;scored?:boolean;events?:unknown[];non_errors?:unknown[]}, excluded:ReadonlySet<string>):boolean {
  return r.kind === "none" && r.split === "train" && !!r.speaker && !excluded.has(r.speaker) &&
    r.exclude !== true && r.excluded !== true && r.scored !== false && !r.events?.length && !r.non_errors?.length;
}

export interface ResidualEmission { mean: number[]; precision: number[][]; logDet: number; count: number }
export interface ResidualArm { center: number[]; scale: number[]; phones: ResidualEmission[] }
export interface ActivationSite {
  pre: string; post: string; width: number; kind: string; scope: string;
  preProjection: number[][]; postProjection: number[][]; nonlinearProjection: number[][];
}
export interface ResidualPredictor { center: number[]; scale: number[]; coefficients: number[][]; strength: number }
export interface ResidualMetadata { schema: 1; inventory: string; tokens: string; dimensions: number;
  sites: ActivationSite[]; primarySha256: string }
export interface ResidualModel extends ResidualMetadata {
  site: number; predictor: ResidualPredictor; residual: ResidualArm; control: ResidualArm;
}
const logadd = (a: number, b: number): number => {
  const m = Math.max(a,b); return Number.isFinite(m) ? m + Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
/** Complete CTC pronunciation, with all boundary paths marginalized. Legal outgoing arcs have
 * equal probability; conditioning the timing prior on completion avoids a length/path-count reward. */
export function residualBoundaryScore(values: number[][], ids: number[]): number {
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
function scores(arm: ResidualArm, raw: number[][], sequences: number[][]): number[] {
  const x=raw.map(r=>r.map((v,d)=>(v-arm.center[d])/arm.scale[d]));
  const used=[...new Set([0,...sequences.flat()])];
  const rows=x.map(r=> {
    const values=arm.phones.map(()=>-Infinity);
    for (const p of used) {
      const e=arm.phones[p]; let q=0;
      for (const row of e.precision) { let z=0; for (let d=0;d<r.length;d++) z+=row[d]*(r[d]-e.mean[d]); q+=z*z; }
      values[p]=-.5*(r.length*Math.log(2*Math.PI)+e.logDet+q);
    }
    return values;
  });
  return sequences.map(s=>residualBoundaryScore(rows,s));
}
export function residualReversal(c: PreactivationResidualEvidence): boolean {
  return [c.residualWritten,c.residualRival,c.controlWritten,c.controlRival].every(Number.isFinite) &&
    c.residualWritten>c.residualRival && c.controlWritten<c.controlRival;
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
export function carryResidualEvidence(chunks: PhonemeEvidence[], certificates: PreactivationResidualEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e,i) => i !== first ? e : {...e, phonemes:e.phonemes.map((p,j) => j ? p : {
    ...p, preactivationResidualVetoes:certificates,
  })});
}
export function residualComparison(model: ResidualModel, residual: number[][], control: number[][],
  candidate: SequenceEvidenceCandidate): PreactivationResidualEvidence | null {
  const sequences=[candidate.written,candidate.rival].map(s=>[...s].map(p=>model.tokens.indexOf(p)+1));
  const valid=(x:number[][])=>x.length>0&&x.every(r=>r.length===model.dimensions&&r.every(Number.isFinite));
  if (!Number.isInteger(candidate.fromFrame) || candidate.fromFrame < 0 ||
    candidate.toFrame-candidate.fromFrame !== residual.length || residual.length!==control.length || !valid(residual)||!valid(control) ||
    sequences.some(s=>!s.length||s.some(p=>p<1||model.residual.phones[p].count<=0||model.control.phones[p].count<=0))) return null;
  const a=scores(model.residual,residual,sequences), b=scores(model.control,control,sequences);
  const result={...candidate,residualWritten:a[0],residualRival:a[1],controlWritten:b[0],controlRival:b[1]};
  return [a[0],a[1],b[0],b[1]].every(Number.isFinite)?result:null;
}
export function validateResidualModel(value: unknown, manifest: ModelManifest): ResidualModel {
  const m = value as ResidualModel;
  const vector = (r: unknown, n: number): r is number[] => Array.isArray(r) && r.length === n && r.every(Number.isFinite);
  const matrix = (v: unknown, rows: number, cols: number): boolean => Array.isArray(v) && v.length === rows && v.every(r => vector(r, cols));
  if (!m || m.schema !== 1 || m.inventory !== manifest.inventory || m.tokens !== manifest.tokens ||
    m.primarySha256 !== manifest.sha256 || m.dimensions !== 16 || !Array.isArray(m.sites) || !m.sites.length ||
    !Number.isInteger(m.site) || m.site < 0 || m.site >= m.sites.length) throw new Error("Invalid preactivation residual model");
  for (const site of m.sites) if (!site || !Number.isSafeInteger(site.width) || site.width <= 0 ||
    ![site.pre, site.post, site.scope, site.kind].every(v => typeof v === "string" && v.length > 0) || site.pre === site.post ||
    !matrix(site.preProjection, 8, site.width) || !matrix(site.postProjection, 16, site.width) ||
    !matrix(site.nonlinearProjection, 64, site.width)) throw new Error("Invalid activation site");
  const width = m.sites[m.site].width, p = m.predictor;
  if (!p || !vector(p.center, width) || !vector(p.scale, width) || p.scale.some(v => v <= 0) ||
    !Number.isFinite(p.strength) || p.strength <= 0 || !matrix(p.coefficients, 8, width + 65))
    throw new Error("Invalid preactivation predictor");
  for (const arm of [m.residual, m.control]) {
    if (!arm || !vector(arm.center,16) || !vector(arm.scale,16) || arm.scale.some(v=>v<=0) ||
      !Array.isArray(arm.phones) || arm.phones.length !== m.tokens.length+1) throw new Error("Invalid residual emission arm");
    for (const e of arm.phones) if (!e || !vector(e.mean,16) || !matrix(e.precision,16,16) || !Number.isFinite(e.logDet) ||
      !Number.isFinite(e.count) || e.count < 0 || e.precision.some((r,i)=>r[i]<=0||r.some((v,j)=>j>i&&v!==0)))
      throw new Error("Invalid residual emission");
  }
  return m;
}
function project(x: readonly number[] | Float32Array, matrix: number[][]): number[] {
  return matrix.map(row => row.reduce((s,v,d) => s+v*x[d],0));
}
/** Predict from ALL corresponding post channels, then retain the unexplained pre coordinates.
 * This is residual relative to the training-frozen nonlinear regressor, not a claim that
 * every remaining coordinate is unrecoverable by an arbitrary predictor. */
export function residualFeatures(site: ActivationSite, predictor: ResidualPredictor, pre: number[], post: number[]) {
  if (pre.length !== site.width || post.length !== site.width || ![...pre,...post].every(Number.isFinite))
    throw new Error("Invalid paired activation row");
  const normalized = post.map((v,d) => (v-predictor.center[d])/predictor.scale[d]);
  const basis = [1, ...normalized, ...project(normalized,site.nonlinearProjection).map(Math.tanh)];
  const predicted = project(basis,predictor.coefficients), original = project(pre,site.preProjection);
  const control = project(post,site.postProjection);
  return {control, residual:[...control.slice(0,8), ...original.map((v,d)=>v-predicted[d])]};
}
/** A separate state instance retains the actual activation pairs before time-grid pooling. */
export async function activationObservations(backend: InferenceBackend, manifest: ModelManifest, meta: ResidualMetadata, audio: Float32Array) {
  const features = new Fbank().push(audio), logProbs: number[][] = [];
  const pairs = meta.sites.map(() => ({pre:[] as number[][],post:[] as number[][]}));
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
    if(!Number.isSafeInteger(n)||n<=0)throw new Error("Invalid extractor frame count");
    for(let t=0;t<n && logProbs.length<total;t++) {
      for(let l=0;l<meta.sites.length;l++) {
        const site=meta.sites[l], pre=out[site.pre],post=out[site.post];
        const rows=pre?.data.length/site.width, ratio=rows/n;
        if(!pre||!post||pre.type!=="float32"||post.type!=="float32"||pre.data.length!==post.data.length||
          pre.dims.length!==3||post.dims.length!==3||pre.dims.at(-1)!==site.width||post.dims.at(-1)!==site.width||
          pre.dims[1]!==1||post.dims[1]!==1||!Number.isSafeInteger(rows)||rows<1||
          !(Number.isSafeInteger(ratio)||Number.isSafeInteger(1/ratio)))throw new Error("Invalid paired activation geometry");
        // Keep each actual pre/post pair. The residual is computed BEFORE native-clock
        // averaging; pooling first could invent information loss in the activation.
        for(let j=0;j<Math.max(1,ratio);j++) {
          // Slower encoder stages repeat an actual paired observation onto the
          // native clock; faster stages retain each pair for later averaging.
          const start=(ratio>=1?t*ratio+j:Math.floor(t*ratio))*site.width;
          pairs[l].pre.push(Array.from(pre.data.slice(start,start+site.width),Number));
          pairs[l].post.push(Array.from(post.data.slice(start,start+site.width),Number));
        }
      }
      logProbs.push(Array.from(out.log_probs.data.slice(t*vocab,(t+1)*vocab),Number));
    }
  }
  return {pairs,logProbs};
}
export async function residualObservations(backend: InferenceBackend, manifest: ModelManifest, model: ResidualModel, audio: Float32Array) {
  const observations=await activationObservations(backend,manifest,model,audio);
  const pair=observations.pairs[model.site], n=observations.logProbs.length, ratio=pair.pre.length/n;
  const residual:number[][]=[],control:number[][]=[];
  for(let t=0;t<n;t++) {
    const a=Array(16).fill(0),b=Array(16).fill(0);
    for(let j=0;j<ratio;j++) {
      const row=residualFeatures(model.sites[model.site],model.predictor,pair.pre[t*ratio+j],pair.post[t*ratio+j]);
      for(let d=0;d<16;d++){a[d]+=row.residual[d]/ratio;b[d]+=row.control[d]/ratio;}
    }
    residual.push(a);control.push(b);
  }
  return {residual,control,logProbs:observations.logProbs};
}
export async function addResidualEvidence(scorer: {model:ResidualModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:PreactivationResidualEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if (!scorer||!audio.length) return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>(),cells=insertionCells(chunks);
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e))); checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const features=await residualObservations(scorer.backend,manifest,scorer.model,audio),certificates:PreactivationResidualEvidence[]=[];
  for (const c of candidates.values()) {
    const result=c.fromFrame>=0&&c.toFrame<=features.residual.length ? residualComparison(scorer.model,
      features.residual.slice(c.fromFrame,c.toFrame),features.control.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(result,c); if (result&&residualReversal(result)) certificates.push(result);
  }
  return carryResidualEvidence(chunks,certificates);
}
