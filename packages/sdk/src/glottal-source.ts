import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, GlottalSourceEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface PhoneReadout { weight: number[][]; bias: number[] }
export interface GlottalSourceMetadata {
  schema: 1; inventory: string; tokens: string; width: number;
  featureOutput: string; primarySha256: string;
}
export interface GlottalSourceModel extends GlottalSourceMetadata {
  invariant: PhoneReadout; control: PhoneReadout;
}
const logadd = (a: number, b: number): number => {
  const m = Math.max(a,b);
  return Number.isFinite(m) ? m + Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};

/** Standard complete-transcript CTC sum. Repeated occurrences require an intervening blank. */
export function pronunciationLogP(rows: number[][], ids: number[]): number {
  if (!rows.length || !ids.length || ids.some(i => i <= 0 || i >= rows[0].length)) return -Infinity;
  const labels = [0, ...ids.flatMap(p => [p,0])];
  let a = labels.map(() => -Infinity);
  a[0] = rows[0][0]; a[1] = rows[0][ids[0]];
  for (let t=1; t<rows.length; t++) {
    a = labels.map((p,s) => rows[t][p] + logadd(a[s], logadd(s ? a[s-1] : -Infinity,
      s>1 && p!==0 && p!==labels[s-2] ? a[s-2] : -Infinity)));
  }
  return logadd(a.at(-1)!,a.at(-2)!);
}
export function readoutRows(arm: PhoneReadout, features: number[][]): number[][] {
  return features.map(row => {
    const logits = arm.weight.map((w,p) => arm.bias[p] + w.reduce((s,v,d) => s+v*row[d],0));
    const m = Math.max(...logits), z = m + Math.log(logits.reduce((s,v) => s+Math.exp(v-m),0));
    return logits.map(v => v-z);
  });
}
export function glottalReversal(c: GlottalSourceEvidence): boolean {
  return [c.invariantWritten,c.invariantRival,c.controlWritten,c.controlRival].every(Number.isFinite) &&
    c.invariantWritten>c.invariantRival && c.controlWritten<c.controlRival;
}
export function glottalComparison(model: GlottalSourceModel, features: number[][],
  candidate: SequenceEvidenceCandidate): GlottalSourceEvidence | null {
  const ids = [candidate.written,candidate.rival].map(s => [...s].map(p => model.tokens.indexOf(p)+1));
  if (!Number.isSafeInteger(candidate.fromFrame) || candidate.fromFrame<0 ||
      !Number.isSafeInteger(candidate.toFrame) || candidate.toFrame-candidate.fromFrame !== features.length ||
      !features.length || features.some(r => r.length!==model.width || !r.every(Number.isFinite)) ||
      ids.some(s => !s.length || s.some(p => p<1))) return null;
  const a=readoutRows(model.invariant,features), b=readoutRows(model.control,features);
  const result={...candidate,invariantWritten:pronunciationLogP(a,ids[0]),invariantRival:pronunciationLogP(a,ids[1]),
    controlWritten:pronunciationLogP(b,ids[0]),controlRival:pronunciationLogP(b,ids[1])};
  return [result.invariantWritten,result.invariantRival,result.controlWritten,result.controlRival].every(Number.isFinite) ? result : null;
}
export function validateGlottalSourceModel(value: unknown, manifest: ModelManifest): GlottalSourceModel {
  const m=value as GlottalSourceModel;
  const vector=(r:unknown,n:number):r is number[] => Array.isArray(r)&&r.length===n&&r.every(Number.isFinite);
  if (!m || m.schema!==1 || m.inventory!==manifest.inventory || m.tokens!==manifest.tokens ||
      m.primarySha256!==manifest.sha256 || !Number.isSafeInteger(m.width) || m.width<=0 ||
      typeof m.featureOutput!=="string" || !m.featureOutput.length) throw new Error("Invalid glottal-source model");
  for (const a of [m.invariant,m.control]) if (!a || !vector(a.bias,m.tokens.length+1) ||
      !Array.isArray(a.weight) || a.weight.length!==a.bias.length || !a.weight.every(r => vector(r,m.width)))
    throw new Error("Invalid glottal-source phone readout");
  return m;
}

/** Separate full-session encoder state; the primary backend and cached states are untouched. */
export async function glottalObservations(backend: InferenceBackend, manifest: ModelManifest,
  meta: GlottalSourceMetadata, audio: Float32Array): Promise<number[][]> {
  const frames=new Fbank().push(audio), features:number[][]=[];
  const total=Math.max(0,Math.floor((Math.floor((frames.length-7)/2)+1)/2));
  let states:Record<string,Tensor>=Object.fromEntries(manifest.states.map(s => [s.name,{type:s.dtype,dims:s.shape,
    data:s.dtype==="int64" ? new BigInt64Array(s.shape.reduce((a,b)=>a*b,1)) : new Float32Array(s.shape.reduce((a,b)=>a*b,1))}]));
  const pad=new Float32Array(manifest.fbank.mels).fill(manifest.fbank.padValue);
  for (let f=0; features.length<total; f+=manifest.shift) {
    const input=new Float32Array(manifest.window*manifest.fbank.mels);
    for (let i=0;i<manifest.window;i++) input.set(frames[f+i]??pad,i*manifest.fbank.mels);
    const out=await backend.run({x:{type:"float32",dims:[1,manifest.window,manifest.fbank.mels],data:input},...states});
    states=Object.fromEntries(manifest.states.map(s=>[s.name,out[`new_${s.name}`]]));
    const h=out[meta.featureOutput], n=out.log_probs.data.length/(manifest.tokens.length+1);
    if (!Number.isSafeInteger(n)||n<=0||!h||h.type!=="float32"||h.data.length!==n*meta.width)
      throw new Error("Invalid glottal-source encoder geometry");
    for (let t=0;t<n && features.length<total;t++)
      features.push(Array.from(h.data.slice(t*meta.width,(t+1)*meta.width),Number));
  }
  return features;
}
/** Only metadata is added. Primary token and posterior values are retained exactly. */
export function carryGlottalSourceEvidence(chunks: PhonemeEvidence[], certificates: GlottalSourceEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first ? e : {...e,phonemes:e.phonemes.map((p,j)=>j ? p : {...p,glottalSourceVetoes:certificates})});
}
/** Exact occurrence identity prevents a repeated spelling or later graph event being retracted. */
export function glottalVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity=(e:CheckerEvent)=>JSON.stringify([e.id,e.timeMs,e.location,e.expected,e.observed]);
  const kept=new Set(reviewed.map(identity));
  const removed=new Set(baseline.filter(e=>e.errorClass==="wrong_word"&&!kept.has(identity(e))).map(identity));
  return live.filter(e=>e.errorClass==="wrong_word"&&removed.has(identity(e)));
}
export async function addGlottalSourceEvidence(scorer:{model:GlottalSourceModel;backend:InferenceBackend}|undefined,
  manifest:ModelManifest,audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,
  options:CheckerOptions={},mode:"free"|"memorization"="free",
  inspect?:(c:GlottalSourceEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if (!scorer || !audio.length) return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of insertionCells(chunks)) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const h=await glottalObservations(scorer.backend,manifest,scorer.model,audio), certificates:GlottalSourceEvidence[]=[];
  for (const c of candidates.values()) {
    const result=c.fromFrame>=0 && c.toFrame<=h.length ? glottalComparison(scorer.model,h.slice(c.fromFrame,c.toFrame),c) : null;
    inspect?.(result,c);
    if (result&&glottalReversal(result)) certificates.push(result);
  }
  return carryGlottalSourceEvidence(chunks,certificates);
}
