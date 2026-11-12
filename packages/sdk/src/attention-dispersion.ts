import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, AttentionDispersionEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface AttentionEmission { mean: number[]; precision: number[][]; logDet: number; count: number }
export interface AttentionArm { center: number[]; scale: number[]; phones: AttentionEmission[] }
export interface AttentionHead {
  meanOutput: string; varianceOutput: string; head: number; heads: number; width: number;
}
export interface AttentionModel {
  schema: 1; inventory: string; tokens: string; dimensions: number;
  heads: AttentionHead[]; selected: number;
  dispersionProjection: number[][]; controlProjection: number[][];
  dispersion: AttentionArm; control: AttentionArm; primarySha256: string;
}
const logadd = (a: number, b: number): number => {
  const m = Math.max(a,b); return Number.isFinite(m) ? m + Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
/** Complete CTC pronunciation, with all boundary paths marginalized. Legal outgoing arcs have
 * equal probability; conditioning the timing prior on completion avoids a length/path-count reward. */
export function attentionBoundaryScore(values: number[][], ids: number[]): number {
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
function scores(arm: AttentionArm, raw: number[][], sequences: number[][]): number[] {
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
  return sequences.map(s=>attentionBoundaryScore(rows,s));
}
export function attentionReversal(c: AttentionDispersionEvidence): boolean {
  return [c.dispersionWritten,c.dispersionRival,c.controlWritten,c.controlRival].every(Number.isFinite) &&
    c.dispersionWritten>c.dispersionRival && c.controlWritten<c.controlRival;
}
/** Only retract the exact live occurrence removed by an otherwise identical evidence replay.
 * Repeated words and later graph confirmations may share a spelling or issue id. */
export function attentionVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity = (e: CheckerEvent) => JSON.stringify([e.id, e.timeMs, e.location, e.expected, e.observed]);
  const kept = new Set(reviewed.map(identity));
  const vetoed = new Set(baseline.filter(e => e.errorClass === "wrong_word" && !kept.has(identity(e))).map(identity));
  return live.filter(e => e.errorClass === "wrong_word" && vetoed.has(identity(e)));
}
/** Carry auxiliary judgments without changing any primary token or posterior. */
export function carryAttentionEvidence(chunks: PhonemeEvidence[], certificates: AttentionDispersionEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e,i) => i !== first ? e : {...e, phonemes:e.phonemes.map((p,j) => j ? p : {
    ...p, attentionDispersionVetoes:certificates,
  })});
}
export function attentionComparison(model: AttentionModel, dispersion: number[][], control: number[][],
  candidate: SequenceEvidenceCandidate): AttentionDispersionEvidence | null {
  const sequences=[candidate.written,candidate.rival].map(s=>[...s].map(p=>model.tokens.indexOf(p)+1));
  const valid=(x:number[][])=>x.length>0&&x.every(r=>r.length===model.dimensions&&r.every(Number.isFinite));
  if (dispersion.length!==control.length || !valid(dispersion)||!valid(control) ||
    sequences.some(s=>!s.length||s.some(p=>p<1||model.dispersion.phones[p].count<=0||model.control.phones[p].count<=0))) return null;
  const a=scores(model.dispersion,dispersion,sequences), b=scores(model.control,control,sequences);
  const result={...candidate,dispersionWritten:a[0],dispersionRival:a[1],controlWritten:b[0],controlRival:b[1]};
  return [a[0],a[1],b[0],b[1]].every(Number.isFinite)?result:null;
}
export function validateAttentionModel(value: unknown,manifest: ModelManifest): AttentionModel {
  const m=value as AttentionModel;
  const vector=(r:unknown,n:number):r is number[]=>Array.isArray(r)&&r.length===n&&r.every(Number.isFinite);
  if (!m||m.schema!==1||m.inventory!==manifest.inventory||m.tokens!==manifest.tokens||m.primarySha256!==manifest.sha256||
    m.dimensions!==16||!Array.isArray(m.heads)||!m.heads.length||
    !Number.isInteger(m.selected)||m.selected<0||m.selected>=m.heads.length||m.heads.some(h=>
      !h||typeof h.meanOutput!=="string"||!h.meanOutput.length||typeof h.varianceOutput!=="string"||!h.varianceOutput.length||
      !Number.isInteger(h.heads)||h.heads<=0||!Number.isInteger(h.head)||h.head<0||h.head>=h.heads||
      !Number.isInteger(h.width)||h.width<=0)||
    !Array.isArray(m.dispersionProjection)||m.dispersionProjection.length!==16||
    !m.dispersionProjection.every(r=>vector(r,2*m.heads[m.selected].width))||
    !Array.isArray(m.controlProjection)||m.controlProjection.length!==16||
    !m.controlProjection.every(r=>vector(r,m.heads[m.selected].width))) throw new Error("Invalid attention dispersion model");
  for (const arm of [m.dispersion,m.control]) {
    if (!vector(arm.center,16)||!vector(arm.scale,16)||arm.scale.some(v=>v<=0)||arm.phones.length!==m.tokens.length+1) throw new Error("Invalid attention emission arm");
    for (const e of arm.phones) {
      if (!vector(e.mean,16)||e.precision.length!==16||!e.precision.every(r=>vector(r,16))||!Number.isFinite(e.logDet)||
        !Number.isFinite(e.count)||e.count<0||e.precision.some((r,i)=>r[i]<=0||r.some((v,j)=>j>i&&v!==0))) throw new Error("Invalid attention emission");
    }
  }
  return m;
}
function project(x: ArrayLike<number>, matrix: number[][]): number[] { return matrix.map(row=>row.reduce((s,v,d)=>s+v*x[d],0)); }
/** Read one selected head on the native clock. Downsampled stages repeat their rows;
 * higher-rate stages average only within the architecture's integer clock ratio. */
export function attentionHeadRows(out: Record<string,Tensor>, head: AttentionHead, n: number) {
  const read=(name:string) => {
    const tensor=out[name];
    if (!tensor||tensor.type!=="float32"||tensor.dims.length!==3||tensor.dims[0]!==head.heads||
      tensor.dims[2]!==head.width) throw new Error("Invalid attention statistic geometry");
    const rows=tensor.dims[1];
    if (!rows||!n||(rows>=n?rows%n:n%rows)) throw new Error("Attention clock cannot map to primary frames");
    const data=tensor.data as Float32Array;
    return Array.from({length:n},(_,t)=> {
      const count=rows>=n?rows/n:1, from=rows>=n?t*count:Math.floor(t*rows/n);
      return Array.from({length:head.width},(_,d)=> {
        let v=0; for (let j=0;j<count;j++) v+=data[(head.head*rows+from+j)*head.width+d]/count;
        if (!Number.isFinite(v)) throw new Error("Nonfinite attention statistic");
        return v;
      });
    });
  };
  const mean=read(head.meanOutput),variance=read(head.varianceOutput);
  if (variance.some(r=>r.some(v=>v<0))) throw new Error("Negative weighted attention variance");
  return {mean,variance};
}
/** A separate graph/cache instance exposes existing weighted means and value dispersion.
 * In cache mode all architectural candidates are retained for training-only head selection. */
export async function attentionObservations(backend: InferenceBackend, manifest: ModelManifest, model: AttentionModel,
  audio: Float32Array, cacheAll=false) {
  const features=new Fbank().push(audio), dispersion:number[][]=[],control:number[][]=[],logProbs:number[][]=[];
  const candidates=model.heads.map(()=>({mean:[] as number[][],variance:[] as number[][]}));
  const total=Math.floor((Math.floor((features.length-7)/2)+1)/2);
  let states:Record<string,Tensor>=Object.fromEntries(manifest.states.map(s=>[s.name,{type:s.dtype,dims:s.shape,
    data:s.dtype==="int64"?new BigInt64Array(s.shape.reduce((a,b)=>a*b,1)):new Float32Array(s.shape.reduce((a,b)=>a*b,1))}]));
  const pad=new Float32Array(manifest.fbank.mels).fill(manifest.fbank.padValue);
  for (let f=0;logProbs.length<total;f+=manifest.shift) {
    const input=new Float32Array(manifest.window*manifest.fbank.mels);
    for (let i=0;i<manifest.window;i++) input.set(features[f+i]??pad,i*manifest.fbank.mels);
    const out=await backend.run({x:{type:"float32",dims:[1,manifest.window,manifest.fbank.mels],data:input},...states});
    states=Object.fromEntries(manifest.states.map(s=>[s.name,out[`new_${s.name}`]]));
    const n=out.log_probs.data.length/(manifest.tokens.length+1),keep=Math.min(n,total-logProbs.length);
    for (const h of cacheAll?model.heads.map((_,i)=>i):[model.selected]) {
      const raw=attentionHeadRows(out,model.heads[h],n);
      candidates[h].mean.push(...raw.mean.slice(0,keep));candidates[h].variance.push(...raw.variance.slice(0,keep));
      if (!cacheAll) for (let t=0;t<keep;t++) {
        dispersion.push(project([...raw.mean[t],...raw.variance[t]],model.dispersionProjection));
        control.push(project(raw.mean[t],model.controlProjection));
      }
    }
    const vocab=manifest.tokens.length+1;
    for (let t=0;t<keep;t++) logProbs.push(Array.from(out.log_probs.data.slice(t*vocab,(t+1)*vocab),Number));
  }
  return {dispersion,control,logProbs,candidates};
}
export async function addAttentionEvidence(scorer: {model:AttentionModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:AttentionDispersionEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if (!scorer||!audio.length) return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>(),cells=insertionCells(chunks);
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e))); checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const features=await attentionObservations(scorer.backend,manifest,scorer.model,audio),certificates:AttentionDispersionEvidence[]=[];
  for (const c of candidates.values()) {
    const result=c.fromFrame>=0&&c.toFrame<=features.dispersion.length ? attentionComparison(scorer.model,
      features.dispersion.slice(c.fromFrame,c.toFrame),features.control.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(result,c); if (result&&attentionReversal(result)) certificates.push(result);
  }
  return carryAttentionEvidence(chunks,certificates);
}
