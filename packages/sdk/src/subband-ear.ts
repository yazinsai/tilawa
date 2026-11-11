import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, SubbandEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, LoadedModel, ModelManifest } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface SubbandModel {
  schema: 1; inventory: string; tokens: string; primarySha256: string;
  mels: 80; stride: 4; kernel: 9; bands: [number, number][];
  learnedParameters: number; controlLearnedParameters: number;
}
export function validateSubbandModel(value: unknown, manifest: ModelManifest): SubbandModel {
  const m = value as SubbandModel;
  if (!m || m.schema !== 1 || m.inventory !== manifest.inventory || m.tokens !== manifest.tokens ||
    m.primarySha256 !== manifest.sha256 || m.mels !== 80 || m.stride !== 4 || m.kernel !== 9 ||
    JSON.stringify(m.bands) !== JSON.stringify([[0,32],[16,48],[32,64],[48,80]]) ||
    !Number.isSafeInteger(m.learnedParameters) || m.learnedParameters <= 0 || m.learnedParameters !== m.controlLearnedParameters)
    throw new Error("Invalid subband ear configuration");
  return m;
}
const logadd = (a: number, b: number): number => {
  const m = Math.max(a,b); return Number.isFinite(m) ? m + Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
/** Ordinary CTC forward sum over the entire requested pronunciation, with no free filler or local phone choice. */
export function subbandCtcScore(rows: number[][], ids: number[]): number {
  if (!rows.length || !ids.length) return -Infinity;
  const labels = [0,...ids.flatMap(p=>[p,0])];
  let a = labels.map(()=>-Infinity); a[0]=rows[0][0]; a[1]=rows[0][ids[0]];
  for (let t=1;t<rows.length;t++) a=labels.map((p,s)=> {
    let v=a[s]; if (s>0) v=logadd(v,a[s-1]);
    if (s>1 && p!==0 && p!==labels[s-2]) v=logadd(v,a[s-2]);
    return v+rows[t][p];
  });
  return logadd(a[a.length-1],a[a.length-2]);
}
export function subbandReversal(c: SubbandEvidence): boolean {
  return [c.fusedWritten,c.fusedRival,c.controlWritten,c.controlRival].every(Number.isFinite) &&
    c.fusedWritten>c.fusedRival && c.controlWritten<c.controlRival;
}
export function subbandComparison(model: SubbandModel, fused: number[][], control: number[][],
  candidate: SequenceEvidenceCandidate): SubbandEvidence | null {
  const ids=[candidate.written,candidate.rival].map(s=>[...s].map(p=>model.tokens.indexOf(p)+1));
  const valid=(x:number[][])=>x.length>0 && x.every(r=>r.length===model.tokens.length+1 && r.every(Number.isFinite));
  if (fused.length!==control.length || !valid(fused) || !valid(control) || ids.some(s=>!s.length || s.includes(0))) return null;
  const a=ids.map(s=>subbandCtcScore(fused,s)), b=ids.map(s=>subbandCtcScore(control,s));
  return [...a,...b].every(Number.isFinite) ? {...candidate,fusedWritten:a[0],fusedRival:a[1],controlWritten:b[0],controlRival:b[1]} : null;
}
/** A repeated spelling or later graph confirmation is a different occurrence. */
export function subbandVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity=(e:CheckerEvent)=>JSON.stringify([e.id,e.timeMs,e.location,e.expected,e.observed]);
  const kept=new Set(reviewed.map(identity));
  const vetoed=new Set(baseline.filter(e=>e.errorClass==="wrong_word" && !kept.has(identity(e))).map(identity));
  return live.filter(e=>e.errorClass==="wrong_word" && vetoed.has(identity(e)));
}
export function carrySubbandEvidence(chunks: PhonemeEvidence[], certificates: SubbandEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first ? e : {...e,phonemes:e.phonemes.map((p,j)=>j ? p : {...p,subbandVetoes:certificates})});
}
/** Both graphs consume the same features, and return exactly the primary listener's native 40 ms frame count.
 * Band encoders retain independent representations until their learned reliability-weighted fusion. */
export async function subbandObservations(scorer: NonNullable<LoadedModel["subbandEar"]>, audio: Float32Array) {
  const features=new Fbank().push(audio), data=new Float32Array(features.length*80);
  features.forEach((row,i)=>data.set(row,i*80));
  if (features.length<9) return {fused:[] as number[][],control:[] as number[][]};
  const feeds={x:{type:"float32" as const,dims:[1,features.length,80],data}};
  const read=async (backend:InferenceBackend)=> {
    const out=(await backend.run(feeds)).log_probs;
    const n=Math.floor((features.length-9)/4)+1, v=scorer.model.tokens.length+1;
    if (!out || out.type!=="float32" || out.dims.length!==3 || out.dims[0]!==1 || out.dims[1]!==n || out.dims[2]!==v || out.data.length!==n*v)
      throw new Error("Invalid subband CTC output geometry");
    return Array.from({length:n},(_,i)=>Array.from(out.data.slice(i*v,(i+1)*v),Number));
  };
  // Sequential graphs avoid duplicating long-session activation storage.
  const fused=await read(scorer.fused), control=await read(scorer.control);
  return {fused,control};
}
export async function addSubbandEvidence(scorer: LoadedModel["subbandEar"], _manifest: ModelManifest,
  audio: Float32Array, chunks: PhonemeEvidence[], index: ReferenceIndex, phrases?: PhraseAsset,
  options: CheckerOptions={},mode:"free"|"memorization"="free",
  inspect?:(result:SubbandEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if (!scorer || !audio.length) return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of insertionCells(chunks)) checker.push(e,tracker.push(trackerInput(e))); checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const observations=await subbandObservations(scorer,audio),certificates:SubbandEvidence[]=[];
  for (const c of candidates.values()) {
    const result=c.fromFrame>=0 && c.toFrame<=observations.fused.length ? subbandComparison(scorer.model,
      observations.fused.slice(c.fromFrame,c.toFrame),observations.control.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(result,c); if (result && subbandReversal(result)) certificates.push(result);
  }
  return carrySubbandEvidence(chunks,certificates);
}
