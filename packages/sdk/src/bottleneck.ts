import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, BottleneckEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

/** None labels are canonical supervision only when the recording is usable and has no edits/repeats. */
export function eligibleBottleneckTake(r: {kind?:string;split?:string;speaker?:string;exclude?:boolean|string|null;excluded?:boolean|string|null;scored?:boolean;events?:unknown[];non_errors?:unknown[]}, excluded:ReadonlySet<string>):boolean {
  return r.kind === "none" && r.split === "train" && !!r.speaker && !excluded.has(r.speaker) &&
    !r.exclude && !r.excluded && r.scored !== false && !r.events?.length && !r.non_errors?.length;
}

export interface BottleneckArm {
  encoder: number[][]; encoderBias: number[];
  output: number[][]; outputBias: number[];
}
export interface BottleneckMetadata {
  schema: 1; inventory: string; tokens: string; width: number;
  featureOutput: string; primarySha256: string;
}
export interface BottleneckModel extends BottleneckMetadata {
  center: number[]; scale: number[]; latent: number; fixedNoiseStd: number;
  bottleneck: BottleneckArm; deterministic: BottleneckArm; fixedNoise: BottleneckArm;
}
const logadd = (a: number, b: number): number => {
  const m = Math.max(a,b); return Number.isFinite(m) ? m + Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
/** Complete CTC pronunciation, with all boundary paths marginalized. Legal outgoing arcs have
 * equal probability; conditioning the timing prior on completion avoids a length/path-count reward. */
export function bottleneckBoundaryScore(values: number[][], ids: number[]): number {
  if (!values.length || !ids.length || !ids.every(p => Number.isInteger(p) && p > 0) ||
      !values.every(r => r.length === values[0].length && r.every(Number.isFinite)) ||
      ids.some(p => p >= values[0].length)) return -Infinity;
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
/** Third-degree spherical-radial Gaussian quadrature: deterministic, with exact
 * first and second moments. Integrate probabilities, never logits or log probabilities. */
export function bottleneckQuadrature(mean: number[], std: number[]): number[][] {
  const radius = Math.sqrt(mean.length);
  return mean.flatMap((_,d) => [-1,1].map(sign => mean.map((m,j) => m + (j===d ? sign*radius*std[d] : 0))));
}
/** Each control uses both encoder branches, with the same learned parameter count.
 * Their second branch enriches the deterministic mean; in the VIB it specifies variance. */
export function bottleneckRows(model: BottleneckModel, arm: BottleneckArm, raw: number[][],
  kind: "bottleneck" | "deterministic" | "fixedNoise"): number[][] {
  return raw.map(row => {
    const x = row.map((v,d) => (v-model.center[d])/model.scale[d]);
    const h = arm.encoder.map((w,i) => w.reduce((sum,v,d) => sum+v*x[d],arm.encoderBias[i]));
    const mean = h.slice(0,model.latent).map((m,d) => kind === "bottleneck" ? m : m+Math.tanh(h[d+model.latent]));
    const std = mean.map((_,d) => kind === "bottleneck" ? Math.exp(.5*Math.max(-8,Math.min(4,h[d+model.latent]))) : model.fixedNoiseStd);
    const nodes = kind === "deterministic" ? [mean] : bottleneckQuadrature(mean,std);
    const total = Array.from({length:model.tokens.length+1},()=>-Infinity);
    for (const node of nodes) {
      const logits = arm.output.map((w,i) => w.reduce((sum,v,d) => sum+v*node[d],arm.outputBias[i]));
      const max = Math.max(...logits), z = max+Math.log(logits.reduce((sum,v) => sum+Math.exp(v-max),0));
      logits.forEach((v,i) => {total[i]=logadd(total[i],v-z-Math.log(nodes.length));});
    }
    return total;
  });
}
function scores(model: BottleneckModel, arm: BottleneckArm, raw: number[][], sequences: number[][],
  kind: "bottleneck" | "deterministic" | "fixedNoise"): number[] {
  const rows = bottleneckRows(model,arm,raw,kind);
  return sequences.map(s => bottleneckBoundaryScore(rows,s));
}
export function bottleneckReversal(c: BottleneckEvidence): boolean {
  return [c.bottleneckWritten,c.bottleneckRival,c.fixedNoiseWritten,c.fixedNoiseRival,c.deterministicWritten,c.deterministicRival].every(Number.isFinite) &&
    c.bottleneckWritten>c.bottleneckRival && c.fixedNoiseWritten<c.fixedNoiseRival && c.deterministicWritten<c.deterministicRival;
}
/** Only retract the exact live occurrence removed by an otherwise identical evidence replay.
 * Repeated words and later graph confirmations may share a spelling or issue id. */
export function bottleneckVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity = (e: CheckerEvent) => JSON.stringify([e.id, e.timeMs, e.location, e.expected, e.observed]);
  const kept = new Set(reviewed.map(identity));
  const vetoed = new Set(baseline.filter(e => e.errorClass === "wrong_word" && !kept.has(identity(e))).map(identity));
  return live.filter(e => e.errorClass === "wrong_word" && vetoed.has(identity(e)));
}
/** Carry auxiliary judgments without changing any primary token or posterior. */
export function carryBottleneckEvidence(chunks: PhonemeEvidence[], certificates: BottleneckEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e,i) => i !== first ? e : {...e, phonemes:e.phonemes.map((p,j) => j ? p : {
    ...p, bottleneckVetoes:certificates,
  })});
}
export function bottleneckComparison(model: BottleneckModel, raw: number[][],
  candidate: SequenceEvidenceCandidate): BottleneckEvidence | null {
  const sequences = [candidate.written,candidate.rival].map(s => [...s].map(p => model.tokens.indexOf(p)+1));
  if (!Number.isInteger(candidate.fromFrame) || !Number.isInteger(candidate.toFrame) || candidate.fromFrame < 0 ||
      candidate.toFrame-candidate.fromFrame !== raw.length || !raw.length ||
      !raw.every(r => r.length===model.width && r.every(Number.isFinite)) ||
      sequences.some(s => !s.length || s.some(p => p<1))) return null;
  const a = scores(model,model.bottleneck,raw,sequences,"bottleneck"), b = scores(model,model.fixedNoise,raw,sequences,"fixedNoise"),
    d = scores(model,model.deterministic,raw,sequences,"deterministic");
  const result = {...candidate,bottleneckWritten:a[0],bottleneckRival:a[1],fixedNoiseWritten:b[0],fixedNoiseRival:b[1],
    deterministicWritten:d[0],deterministicRival:d[1]};
  return [a[0],a[1],b[0],b[1],d[0],d[1]].every(Number.isFinite) ? result : null;
}
export function validateBottleneckModel(value: unknown, manifest: ModelManifest): BottleneckModel {
  const m = value as BottleneckModel;
  const vector = (r: unknown, n: number): r is number[] => Array.isArray(r) && r.length===n && r.every(Number.isFinite);
  const matrix = (v: unknown, rows: number, cols: number): v is number[][] => Array.isArray(v) && v.length===rows && v.every(r => vector(r,cols));
  if (!m || m.schema!==1 || m.inventory!==manifest.inventory || m.tokens!==manifest.tokens ||
      m.primarySha256!==manifest.sha256 || !Number.isSafeInteger(m.width) || m.width<1 ||
      !Number.isSafeInteger(m.latent) || m.latent<1 || m.latent>64 ||
      !Number.isFinite(m.fixedNoiseStd) || m.fixedNoiseStd<=0 ||
      typeof m.featureOutput!=="string" || !m.featureOutput ||
      !vector(m.center,m.width) || !vector(m.scale,m.width) ||
      m.scale.some(v => v<=0)) throw new Error("Invalid bottleneck model");
  for (const arm of [m.bottleneck,m.deterministic,m.fixedNoise]) {
    if (!arm || !matrix(arm.encoder,2*m.latent,m.width) || !vector(arm.encoderBias,2*m.latent) ||
        !matrix(arm.output,m.tokens.length+1,m.latent) || !vector(arm.outputBias,m.tokens.length+1))
      throw new Error("Invalid bottleneck readout");
  }
  return m;
}
/** Full-session separate streaming state. The final linear input includes activation quantization.
 * All arms retain the same frame clock; primary rows and caches are untouched. */
export async function bottleneckObservations(backend: InferenceBackend, manifest: ModelManifest, meta: BottleneckMetadata, audio: Float32Array) {
  const features = new Fbank().push(audio), logProbs: number[][] = [], featuresOut: number[][] = [];
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
    if(!Number.isSafeInteger(n)||n<=0||!h||h.type!=="float32"||
       h.data.length!==n*meta.width) throw new Error("Invalid final projection geometry");
    for(let t=0;t<n && logProbs.length<total;t++) {
      featuresOut.push(Array.from(h.data.slice(t*meta.width,(t+1)*meta.width),Number));
      logProbs.push(Array.from(out.log_probs.data.slice(t*vocab,(t+1)*vocab),Number));
    }
  }
  return {features:featuresOut,logProbs};
}
export async function addBottleneckEvidence(scorer: {model:BottleneckModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:BottleneckEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if (!scorer||!audio.length) return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>(),cells=insertionCells(chunks);
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e))); checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const features=await bottleneckObservations(scorer.backend,manifest,scorer.model,audio),certificates:BottleneckEvidence[]=[];
  for (const c of candidates.values()) {
    const result=c.fromFrame>=0&&c.toFrame<=features.features.length ? bottleneckComparison(scorer.model,
      features.features.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(result,c); if (result&&bottleneckReversal(result)) certificates.push(result);
  }
  return carryBottleneckEvidence(chunks,certificates);
}
