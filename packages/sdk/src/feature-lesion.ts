import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, FeatureLesionEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { insertionCells } from "./seam-insertion.js";

export interface FeatureLesionMetadata {
  schema: 1; inventory: string; tokens: string; primarySha256: string;
  width: number; featureOutput: string; logitOutput: string;
  effectiveWeight: number[][]; bias: number[];
}
export interface FeatureLesionModel extends FeatureLesionMetadata {
  mean: number[];
  mask: number[];
}
const logadd = (a: number, b: number) => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  return Math.max(a,b) + Math.log1p(Math.exp(-Math.abs(a-b)));
};
/** Ordinary complete-pronunciation CTC: both arms use the shipped topology. */
export function featureLesionSequenceScore(values: number[][], ids: number[]): number {
  if (!values.length) return ids.length ? -Infinity : 0;
  if (ids.some(p => !Number.isInteger(p) || p < 1 || p >= values[0].length)) return -Infinity;
  const labels = [0, ...ids.flatMap(p => [p,0])];
  let prev = labels.map(() => -Infinity);
  prev[0] = values[0][0];
  if (ids.length) prev[1] = values[0][ids[0]];
  for (const row of values.slice(1)) {
    prev = labels.map((p,s) => {
      let v = prev[s];
      if (s) v = logadd(v,prev[s-1]);
      if (s > 1 && p && p !== labels[s-2]) v = logadd(v,prev[s-2]);
      return v + row[p];
    });
  }
  return ids.length ? logadd(prev.at(-1)!,prev.at(-2)!) : prev[0];
}
export function meanLesionedLogProbs(model: FeatureLesionModel, features: number[][], logits: number[][]): number[][] {
  return logits.map((row,t) => {
    const adjusted = row.map((v,p) => v + model.mask.reduce((sum,d) =>
      sum + (model.mean[d]-features[t][d])*model.effectiveWeight[d][p],0));
    const peak = Math.max(...adjusted), z = peak + Math.log(adjusted.reduce((s,v) => s+Math.exp(v-peak),0));
    return adjusted.map(v => v-z);
  });
}
export function featureLesionReversal(c: FeatureLesionEvidence): boolean {
  return [c.originalWritten,c.originalRival,c.lesionedWritten,c.lesionedRival].every(Number.isFinite) &&
    c.originalWritten < c.originalRival && c.lesionedWritten > c.lesionedRival;
}
export function featureLesionComparison(model: FeatureLesionModel, features: number[][], logits: number[][],
  original: number[][], candidate: SequenceEvidenceCandidate): FeatureLesionEvidence | null {
  const n = features.length, vocab = model.tokens.length+1;
  const valid = (x:number[][],width:number) => x.length === n && x.every(r => r.length===width && r.every(Number.isFinite));
  const sequences = [candidate.written,candidate.rival].map(s => [...s].map(p => model.tokens.indexOf(p)+1));
  if (!Number.isSafeInteger(candidate.fromFrame) || candidate.fromFrame<0 || !n ||
      candidate.toFrame-candidate.fromFrame!==n || !valid(features,model.width) || !valid(logits,vocab) || !valid(original,vocab) ||
      sequences.some(s => !s.length || s.some(p => p<1))) return null;
  const modified = meanLesionedLogProbs(model,features,logits);
  const c = {...candidate, originalWritten:featureLesionSequenceScore(original,sequences[0]),
    originalRival:featureLesionSequenceScore(original,sequences[1]),
    lesionedWritten:featureLesionSequenceScore(modified,sequences[0]),lesionedRival:featureLesionSequenceScore(modified,sequences[1])};
  return [c.originalWritten,c.originalRival,c.lesionedWritten,c.lesionedRival].every(Number.isFinite)?c:null;
}
export function validateFeatureLesionModel(value: unknown, manifest: ModelManifest): FeatureLesionModel {
  const m = value as FeatureLesionModel;
  const vector = (v:unknown,n:number):v is number[] => Array.isArray(v) && v.length===n && v.every(Number.isFinite);
  const vocab = manifest.tokens.length+1;
  if (!m || m.schema!==1 || m.inventory!==manifest.inventory || m.tokens!==manifest.tokens || m.primarySha256!==manifest.sha256 ||
      !Number.isSafeInteger(m.width) || m.width<1 || !m.featureOutput || !m.logitOutput ||
      !vector(m.mean,m.width) || !vector(m.bias,vocab) || !Array.isArray(m.effectiveWeight) ||
      m.effectiveWeight.length!==m.width || !m.effectiveWeight.every(r => vector(r,vocab)) ||
      !Array.isArray(m.mask) || !m.mask.length || m.mask.length>Math.max(1,Math.floor(m.width/32)) ||
      new Set(m.mask).size!==m.mask.length || m.mask.some(d => !Number.isInteger(d) || d<0 || d>=m.width)) {
    throw new Error("Invalid frozen feature lesion");
  }
  return m;
}
/** Separate full-stream state preserves the acoustic clock and the primary encoder's context. */
export async function featureLesionObservations(backend:InferenceBackend, manifest:ModelManifest,
  metadata:FeatureLesionMetadata, audio:Float32Array) {
  const fbank = new Fbank().push(audio), features:number[][]=[], logits:number[][]=[], logProbs:number[][]=[];
  const total = Math.max(0,Math.floor((Math.floor((fbank.length-7)/2)+1)/2));
  const vocab = manifest.tokens.length+1;
  let states:Record<string,Tensor> = Object.fromEntries(manifest.states.map(s => [s.name,{type:s.dtype,dims:s.shape,
    data:s.dtype === "int64" ? new BigInt64Array(s.shape.reduce((a,b)=>a*b,1)) : new Float32Array(s.shape.reduce((a,b)=>a*b,1))}]));
  const pad = new Float32Array(manifest.fbank.mels).fill(manifest.fbank.padValue);
  for (let f=0;features.length<total;f+=manifest.shift) {
    const data = new Float32Array(manifest.window*manifest.fbank.mels);
    for (let i=0;i<manifest.window;i++) data.set(fbank[f+i]??pad,i*manifest.fbank.mels);
    const out = await backend.run({x:{type:"float32",data,dims:[1,manifest.window,manifest.fbank.mels]},...states});
    states = Object.fromEntries(manifest.states.map(s => [s.name,out[`new_${s.name}`]]));
    const h = out[metadata.featureOutput], z = out[metadata.logitOutput], lp = out.log_probs;
    const n = lp.data.length/vocab;
    if (!h || !z || h.type!=="float32" || z.type!=="float32" || lp.type!=="float32" || !Number.isSafeInteger(n) || n<1 ||
        h.data.length!==n*metadata.width || z.data.length!==n*vocab) throw new Error("Invalid feature lesion clock");
    for (let t=0;t<n && features.length<total;t++) {
      features.push(Array.from((h.data as Float32Array).slice(t*metadata.width,(t+1)*metadata.width)));
      logits.push(Array.from((z.data as Float32Array).slice(t*vocab,(t+1)*vocab)));
      logProbs.push(Array.from((lp.data as Float32Array).slice(t*vocab,(t+1)*vocab)));
    }
  }
  return {features,logits,logProbs};
}
/** Read-only discovery of exactly the complete comparisons used to confirm string-path warnings. */
export function featureLesionRequests(chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,
  options:CheckerOptions={},mode:"free"|"memorization"="free") {
  const candidates = new Map<string,SequenceEvidenceCandidate>();
  const tracker = createTracker(index,{mode,phrases,acquireMs:1000});
  const checker = createChecker(index,{...options,phrases,sequenceEvidence(c) { candidates.set(JSON.stringify(c),c); return true; }});
  const events = [];
  for (const e of insertionCells(chunks)) { const t=tracker.push(trackerInput(e)); events.push(...t,...checker.push(e,t)); }
  const tail=tracker.end(); events.push(...tail,...checker.end(tail));
  return {candidates:[...candidates.values()],events,mistakes:checker.mistakes()};
}
export function carryFeatureLesionEvidence(chunks:PhonemeEvidence[],certificates:FeatureLesionEvidence[]):PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const at=chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e,i) => i!==at?e:{...e,phonemes:e.phonemes.map((p,j) => j?p:{...p,featureLesionVetoes:certificates})});
}
/** Only retract an identical live occurrence whose replayed string confirmation was vetoed. */
export function featureLesionVerdictRemovals(live:CheckerEvent[],baseline:CheckerEvent[],reviewed:CheckerEvent[]):CheckerEvent[] {
  const identity=(e:CheckerEvent)=>JSON.stringify([e.id,e.timeMs,e.location,e.expected,e.observed]);
  const kept=new Set(reviewed.map(identity));
  const removed=new Set(baseline.filter(e=>e.errorClass==="wrong_word"&&!kept.has(identity(e))).map(identity));
  return live.filter(e=>e.errorClass==="wrong_word"&&removed.has(identity(e)));
}
export async function addFeatureLesionEvidence(scorer:{model:FeatureLesionModel;backend:InferenceBackend}|undefined,
  manifest:ModelManifest,audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,
  options:CheckerOptions={},mode:"free"|"memorization"="free",
  inspect?:(c:FeatureLesionEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if (!scorer || !audio.length) return chunks;
  const {candidates}=featureLesionRequests(chunks,index,phrases,options,mode);
  if (!candidates.length) return chunks;
  const features=await featureLesionObservations(scorer.backend,manifest,scorer.model,audio);
  const certificates:FeatureLesionEvidence[]=[];
  for (const c of candidates) {
    const a=c.fromFrame,b=c.toFrame;
    const result=a>=0 && b<=features.features.length ? featureLesionComparison(scorer.model,
      features.features.slice(a,b),features.logits.slice(a,b),features.logProbs.slice(a,b),c):null;
    inspect?.(result,c);
    if (result && featureLesionReversal(result)) certificates.push(result);
  }
  return carryFeatureLesionEvidence(chunks,certificates);
}
