import { createChecker, createTracker, acousticSpanVeto, trackerInput } from "@tilawa/core";
import type { CheckerOptions, AcousticSpanEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface AcousticSpanModel {
  schema: 1; inventory: string; tokens: string; dimensions: number; primarySha256: string;
  featureOutput: string; projection: number[][]; center: number[]; scale: number[];
}
export function validateAcousticSpanModel(value: unknown, manifest: ModelManifest): AcousticSpanModel {
  const m = value as AcousticSpanModel;
  const vector = (r: unknown, n: number): r is number[] => Array.isArray(r) && r.length === n && r.every(Number.isFinite);
  if (!m || m.schema !== 1 || m.inventory !== manifest.inventory || m.tokens !== manifest.tokens ||
    m.primarySha256 !== manifest.sha256 || m.dimensions !== 32 || typeof m.featureOutput !== "string" ||
    !m.featureOutput || m.projection?.length !== 32 || !m.projection[0]?.length ||
    !m.projection.every(r => vector(r, m.projection[0].length)) || !vector(m.center, 32) ||
    !vector(m.scale, 32) || m.scale.some(v => v <= 0)) throw new Error("Invalid acoustic-span model");
  return m;
}
/** Frozen pre-head observations, with separate state and the primary native frame clock. */
export async function acousticSpanFeatures(backend:InferenceBackend,manifest:ModelManifest,model:AcousticSpanModel,audio:Float32Array) {
  const features=new Fbank().push(audio),observations:number[][]=[],logProbs:number[][]=[];
  const total=Math.max(0,Math.floor((Math.floor((features.length-7)/2)+1)/2));
  let states:Record<string,Tensor>=Object.fromEntries(manifest.states.map(s=>[s.name,{type:s.dtype,dims:s.shape,
    data:s.dtype==="int64"?new BigInt64Array(s.shape.reduce((a,b)=>a*b,1)):new Float32Array(s.shape.reduce((a,b)=>a*b,1))}]));
  const pad=new Float32Array(manifest.fbank.mels).fill(manifest.fbank.padValue);
  for(let f=0;observations.length<total;f+=manifest.shift) {
    const input=new Float32Array(manifest.window*manifest.fbank.mels);
    for(let i=0;i<manifest.window;i++)input.set(features[f+i]??pad,i*manifest.fbank.mels);
    const out=await backend.run({x:{type:"float32",dims:[1,manifest.window,manifest.fbank.mels],data:input},...states});
    states=Object.fromEntries(manifest.states.map(s=>[s.name,out[`new_${s.name}`]]));
    const vocab=manifest.tokens.length+1,n=out.log_probs.data.length/vocab;
    const tensor=out[model.featureOutput],width=model.projection[0].length;
    if(!tensor||tensor.type!=="float32"||tensor.data.length!==n*width)throw new Error("Invalid frozen feature geometry");
    for(let t=0;t<n&&observations.length<total;t++) {
      const row=tensor.data.slice(t*width,(t+1)*width) as Float32Array;
      observations.push(model.projection.map(w=>w.reduce((s,v,d)=>s+v*row[d],0)));
      logProbs.push(Array.from(out.log_probs.data.slice(t*vocab,(t+1)*vocab),Number));
    }
  }
  return {observations,logProbs};
}

export interface CompatibilityHead {
  embedding: number[][];
  weight: number[][];
  bias: number[];
  output: number[];
  intercept: number;
}
export function validateCompatibilityHead(value: unknown, vocabulary: number): CompatibilityHead {
  const h = value as CompatibilityHead;
  const vector = (x: unknown, n: number): x is number[] => Array.isArray(x) && x.length === n && x.every(Number.isFinite);
  if (!h || h.embedding?.length !== vocabulary || !h.embedding.every(r => vector(r, 16)) ||
    h.weight?.length !== 32 || !h.weight.every(r => vector(r, 80)) || !vector(h.bias, 32) ||
    !vector(h.output, 32) || !Number.isFinite(h.intercept)) throw new Error("Invalid compatibility head");
  return h;
}
const logadd = (a: number, b: number): number => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
};
/** Uniform boundary prior over all ordered, nonempty phone segments.
 * Every frame belongs to exactly one phone and every candidate phone consumes frames.
 * Divide by the same prior partition sum to remove the number-of-timings advantage. */
export function marginalCompatibility(emissions: number[][]): number | null {
  const phones = emissions.length, frames = emissions[0]?.length ?? 0;
  if (!phones || frames < phones || !emissions.every(r => r.length === frames && r.every(Number.isFinite))) return null;
  let score = new Array(phones).fill(-Infinity), prior = [...score];
  score[0] = emissions[0][0] / frames; prior[0] = 0;
  for (let t = 1; t < frames; t++) {
    const next = new Array(phones).fill(-Infinity), base = [...next];
    for (let p = 0; p < phones; p++) {
      next[p] = logadd(score[p], p ? score[p - 1] : -Infinity) + emissions[p][t] / frames;
      base[p] = logadd(prior[p], p ? prior[p - 1] : -Infinity);
    }
    score = next; prior = base;
  }
  return score[phones - 1] - prior[phones - 1];
}
export function compatibilityScore(h: CompatibilityHead, m: AcousticSpanModel, x: number[][], pronunciation: string): number | null {
  const ids = [...pronunciation].map(p => m.tokens.indexOf(p) + 1);
  if (!ids.length || ids.some(p => p < 1) || x.length < ids.length ||
    !x.every(r => r.length === 32 && r.every(Number.isFinite))) return null;
  // Equal-prior uncertain outer boundaries, in addition to the internal partition sum.
  const boundaries = [[0, x.length], [1, x.length], [0, x.length - 1]].filter(([a, b]) => b - a >= ids.length);
  const scores = boundaries.map(([a, b]) => {
    const normalized = x.slice(a, b).map(r => r.map((v, d) => (v - m.center[d]) / m.scale[d]));
    const emissions = ids.map(id => normalized.map((r, t) => {
      const input = [...r, ...r.map((v, d) => v - normalized[Math.max(0, t - 1)][d]), ...h.embedding[id]];
      return h.weight.reduce((s, w, i) => s + h.output[i] * Math.tanh(h.bias[i] +
        w.reduce((n, v, d) => n + v * input[d], 0)), h.intercept);
    }));
    return marginalCompatibility(emissions)!;
  });
  return scores.reduce(logadd, -Infinity) - Math.log(scores.length);
}
export function carryAcousticSpanEvidence(chunks: PhonemeEvidence[], certificates: AcousticSpanEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e, i) => i !== first ? e : { ...e, phonemes: e.phonemes.map((p, j) => j ? p :
    { ...p, acousticSpanVetoes: [...(p.acousticSpanVetoes ?? []), ...certificates] }) });
}
export async function addAcousticSpanEvidence(scorer: {
  model: AcousticSpanModel; backend: InferenceBackend; contrast: CompatibilityHead; control: CompatibilityHead;
} | undefined, manifest: ModelManifest, audio: Float32Array, chunks: PhonemeEvidence[], index: ReferenceIndex,
  phrases?: PhraseAsset, options: CheckerOptions = {}, mode: "free" | "memorization" = "free",
  inspect?: (c: AcousticSpanEvidence | null, request: SequenceEvidenceCandidate) => void): Promise<PhonemeEvidence[]> {
  if (!scorer || !audio.length) return chunks;
  const candidates = new Map<string, SequenceEvidenceCandidate>();
  const tracker = createTracker(index, { mode, phrases, acquireMs: 1000 });
  const checker = createChecker(index, { ...options, phrases, sequenceEvidence(c) {
    candidates.set(JSON.stringify(c), c); return true;
  } });
  for (const e of insertionCells(chunks)) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const features = await acousticSpanFeatures(scorer.backend, manifest, scorer.model, audio);
  const certificates: AcousticSpanEvidence[] = [];
  for (const c of candidates.values()) {
    let result: AcousticSpanEvidence | null = null;
    if (c.fromFrame >= 0 && c.toFrame <= features.observations.length) {
      const x = features.observations.slice(c.fromFrame, c.toFrame);
      const scores = [scorer.contrast, scorer.control].flatMap(h =>
        [c.written, c.rival].map(p => compatibilityScore(h, scorer.model, x, p)));
      if (scores.every(v => v !== null)) result = { ...c,
        contrastMargin: scores[0]! - scores[1]!, controlMargin: scores[2]! - scores[3]! };
    }
    inspect?.(result, c);
    if (result && acousticSpanVeto(result)) certificates.push(result);
  }
  return carryAcousticSpanEvidence(chunks, certificates);
}
