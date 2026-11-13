import { createChecker, createTracker, sequenceEvidenceKey, trackerInput, wordScrubVeto } from "@tilawa/core";
import type { CheckerOptions, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate, WordScrubEvidence } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

const vector = (x: unknown, n: number): x is number[] => Array.isArray(x) && x.length === n && x.every(Number.isFinite);
const matrix = (x: unknown, n: number, d: number): x is number[][] => Array.isArray(x) && x.length === n && x.every(r => vector(r, d));
export interface PhoneReadout { weight: number[][]; bias: number[]; residual: number[][]; output: number[][]; intercept: number[] }
export interface WordScrubModel {
  schema: 1; inventory: string; tokens: string; primarySha256: string; featureOutput: string;
  projection: number[][]; center: number[]; scale: number[];
  residual: PhoneReadout; original: PhoneReadout; shuffled: PhoneReadout;
}
export interface WordScrubScorer { model: WordScrubModel; backend: InferenceBackend }
export function validateWordScrubModel(value: unknown, manifest: ModelManifest): WordScrubModel {
  const m = value as WordScrubModel, v = manifest.tokens.length + 1;
  const head = (h: PhoneReadout) => h && matrix(h.weight, 32, 32) && vector(h.bias, 32) &&
    matrix(h.residual, 32, 32) && matrix(h.output, v, 32) && vector(h.intercept, v);
  if (!m || m.schema !== 1 || m.inventory !== manifest.inventory || m.tokens !== manifest.tokens ||
    m.primarySha256 !== manifest.sha256 || !m.featureOutput || typeof m.featureOutput !== "string" ||
    !m.projection?.[0]?.length || !matrix(m.projection, 32, m.projection[0].length) ||
    !vector(m.center, 32) || !vector(m.scale, 32) || m.scale.some(s => s <= 0) ||
    !head(m.residual) || !head(m.original) || !head(m.shuffled)) throw new Error("Invalid word-scrub model");
  return m;
}
const logadd = (a: number, b: number): number => a === -Infinity ? b : b === -Infinity ? a :
  Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));

/** Separate frozen graph and cache; never feeds features into the primary decode. */
export async function wordScrubFeatures(backend: InferenceBackend, manifest: ModelManifest,
  model: Pick<WordScrubModel, "featureOutput" | "projection">, audio: Float32Array) {
  const features = new Fbank().push(audio), observations: number[][] = [], logProbs: number[][] = [];
  const total = Math.max(0, Math.floor((Math.floor((features.length - 7) / 2) + 1) / 2));
  let states: Record<string, Tensor> = Object.fromEntries(manifest.states.map(s => [s.name, { type: s.dtype, dims: s.shape,
    data: s.dtype === "int64" ? new BigInt64Array(s.shape.reduce((a, b) => a * b, 1)) : new Float32Array(s.shape.reduce((a, b) => a * b, 1)) }]));
  const pad = new Float32Array(manifest.fbank.mels).fill(manifest.fbank.padValue);
  for (let f = 0; observations.length < total; f += manifest.shift) {
    const input = new Float32Array(manifest.window * manifest.fbank.mels);
    for (let i = 0; i < manifest.window; i++) input.set(features[f + i] ?? pad, i * manifest.fbank.mels);
    const out = await backend.run({ x: { type: "float32", dims: [1, manifest.window, manifest.fbank.mels], data: input }, ...states });
    states = Object.fromEntries(manifest.states.map(s => [s.name, out[`new_${s.name}`]]));
    const vocab = manifest.tokens.length + 1, n = out.log_probs.data.length / vocab;
    const tensor = out[model.featureOutput], width = model.projection[0].length;
    if (!n || !tensor || tensor.type !== "float32" || tensor.data.length !== n * width) throw new Error("Invalid frozen feature geometry");
    for (let t = 0; t < n && observations.length < total; t++) {
      const row = tensor.data.slice(t * width, (t + 1) * width) as Float32Array;
      observations.push(model.projection.map(w => w.reduce((s, v, d) => s + v * row[d], 0)));
      logProbs.push(Array.from(out.log_probs.data.slice(t * vocab, (t + 1) * vocab), Number));
    }
  }
  return { observations, logProbs };
}

export function readoutLogProbs(h: PhoneReadout, x: number[][]): number[][] {
  return x.map(r => {
    const hidden = h.weight.map((w, i) => Math.tanh(w.reduce((s, v, d) => s + v * r[d], h.bias[i])));
    const z = r.map((v, d) => v + h.residual[d].reduce((sum, w, i) => sum + w * hidden[i], 0));
    const logits = h.output.map((w, i) => w.reduce((s, v, d) => s + v * z[d], h.intercept[i]));
    const norm = logits.reduce(logadd, -Infinity); return logits.map(v => v - norm);
  });
}
/** Complete CTC chain, summing every phone placement. No phone can be omitted. */
export function sequenceLogProbability(lp: number[][], ids: number[]): number {
  if (!lp.length || !ids.length || ids.some(i => i < 1 || i >= lp[0].length)) return -Infinity;
  const labels = [0, ...ids.flatMap(i => [i, 0])];
  let a = labels.map(() => -Infinity); a[0] = lp[0][0]; a[1] = lp[0][ids[0]];
  for (let t = 1; t < lp.length; t++) a = labels.map((id, s) => {
    let p = a[s]; if (s) p = logadd(p, a[s - 1]);
    if (s > 1 && id && id !== labels[s - 2]) p = logadd(p, a[s - 2]);
    return p + lp[t][id];
  });
  return logadd(a.at(-1)!, a.at(-2)!);
}

export function scrubComparisons(m: WordScrubModel, x: number[][],
  c: SequenceEvidenceCandidate): WordScrubEvidence | null {
  const ids = [c.written, c.rival].map(s => [...s].map(p => m.tokens.indexOf(p) + 1));
  if (ids.some(r => !r.length || r.some(i => i < 1)) || !x.length || !x.every(r => vector(r, 32))) return null;
  const normalized = x.map(r => r.map((v, d) => (v - m.center[d]) / m.scale[d]));
  const margins = [m.residual, m.original, m.shuffled].map(head => {
    const lp = readoutLogProbs(head, normalized);
    return sequenceLogProbability(lp, ids[0]) - sequenceLogProbability(lp, ids[1]);
  });
  const result = { ...c, residualMargin: margins[0], originalMargin: margins[1], shuffledMargin: margins[2] };
  return margins.every(Number.isFinite) ? result : null;
}

export function carryWordScrubEvidence(chunks: PhonemeEvidence[], certificates: WordScrubEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e, i) => i !== first ? e : { ...e, phonemes: e.phonemes.map((p, j) => j ? p :
    { ...p, wordScrubVetoes: [...(p.wordScrubVetoes ?? []), ...certificates] }) });
}
export async function addWordScrubEvidence(scorer: WordScrubScorer | undefined, manifest: ModelManifest,
  audio: Float32Array, chunks: PhonemeEvidence[], index: ReferenceIndex, phrases?: PhraseAsset,
  options: CheckerOptions = {}, mode: "free" | "memorization" = "free",
  inspect?: (result: WordScrubEvidence | null, request: SequenceEvidenceCandidate) => void): Promise<PhonemeEvidence[]> {
  if (!scorer || !audio.length) return chunks;
  const candidates = new Map<string, SequenceEvidenceCandidate>();
  const tracker = createTracker(index, { mode, phrases, acquireMs: 1000 });
  const checker = createChecker(index, { ...options, phrases, sequenceEvidence(c) {
    candidates.set(sequenceEvidenceKey(c), c); return true;
  } });
  for (const e of insertionCells(chunks)) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const features = await wordScrubFeatures(scorer.backend, manifest, scorer.model, audio);
  const certificates: WordScrubEvidence[] = [];
  for (const c of candidates.values()) {
    const result = c.fromFrame >= 0 && c.toFrame <= features.observations.length ? scrubComparisons(scorer.model,
      features.observations.slice(c.fromFrame, c.toFrame), c) : null;
    inspect?.(result, c);
    if (result && wordScrubVeto(result)) certificates.push(result);
  }
  return carryWordScrubEvidence(chunks, certificates);
}
