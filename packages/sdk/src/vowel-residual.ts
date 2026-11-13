import { createChecker, createTracker, sequenceEvidenceKey, trackerInput, vowelResidualVeto } from "@tilawa/core";
import type { CheckerOptions, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate, VowelResidualEvidence } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

const vector = (x: unknown, n: number): x is number[] => Array.isArray(x) && x.length === n && x.every(Number.isFinite);
const matrix = (x: unknown, n: number, d: number): x is number[][] => Array.isArray(x) && x.length === n && x.every(r => vector(r, d));
export interface PhoneReadout { weight: number[][]; bias: number[]; output: number[][]; intercept: number[] }
export interface VowelResidualModel {
  schema: 1; inventory: string; tokens: string; primarySha256: string; featureOutput: string;
  projection: number[][]; center: number[]; scale: number[];
  /** Disjoint acoustic vowel-quality ids and a normalized bounded duration prior. */
  vowels: number[][]; durations: number[];
  predictor: number[][]; shuffledPredictor: number[][];
  residual: PhoneReadout; original: PhoneReadout; shuffled: PhoneReadout;
}
export interface VowelResidualScorer { model: VowelResidualModel; backend: InferenceBackend }
export function validateVowelResidualModel(value: unknown, manifest: ModelManifest): VowelResidualModel {
  const m = value as VowelResidualModel, v = manifest.tokens.length + 1;
  const head = (h: PhoneReadout) => h && matrix(h.weight, 32, 32) && vector(h.bias, 32) &&
    matrix(h.output, v, 32) && vector(h.intercept, v);
  if (!m || m.schema !== 1 || m.inventory !== manifest.inventory || m.tokens !== manifest.tokens ||
    m.primarySha256 !== manifest.sha256 || !m.featureOutput || typeof m.featureOutput !== "string" ||
    !m.projection?.[0]?.length || !matrix(m.projection, 32, m.projection[0].length) ||
    !vector(m.center, 32) || !vector(m.scale, 32) || m.scale.some(s => s <= 0) ||
    !matrix(m.predictor, 32, 10) || !matrix(m.shuffledPredictor, 32, 10) ||
    !Array.isArray(m.vowels) || m.vowels.length !== 3 || m.vowels.some(r => !Array.isArray(r) || !r.length ||
      r.some(i => !Number.isSafeInteger(i) || i <= 0 || i >= v)) ||
    new Set(m.vowels.flat()).size !== m.vowels.flat().length ||
    !Array.isArray(m.durations) || !m.durations.length || m.durations.length > 75 ||
    m.durations.some(p => !Number.isFinite(p) || p <= 0) || Math.abs(m.durations.reduce((a, b) => a + b, 0) - 1) > 1e-6 ||
    !head(m.residual) || !head(m.original) || !head(m.shuffled)) throw new Error("Invalid vowel-residual model");
  return m;
}
const logadd = (a: number, b: number): number => a === -Infinity ? b : b === -Infinity ? a :
  Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));

/** Acoustic-only semi-Markov vowel inference. Sum all bounded vowel intervals,
 * qualities and boundaries once. The phase moments are E[u] and E[u²], where
 * u is normalized position within each possible vowel interval. Blank is neutral
 * between the four contexts; neither pronunciation is an argument to this code. */
export function vowelContext(logProbs: number[][], vowels: number[][], durations: number[]): number[][] {
  const n = logProbs.length;
  if (!n) return [];
  const vowelIds = new Set(vowels.flat());
  const emissions = logProbs.map(r => {
    const p = r.map(Math.exp), blank = p[0] / 4;
    return [p.reduce((s, x, i) => s + (i && !vowelIds.has(i) ? x : 0), blank),
      ...vowels.map(ids => ids.reduce((s, i) => s + p[i], blank))].map(p => Math.log(Math.max(1e-12, p)));
  });
  const prefix = Array.from({ length: 3 }, (_, q) => {
    const p = [0]; for (const r of emissions) p.push(p.at(-1)! + r[q + 1]); return p;
  });
  const each = (a: number, visit: (b: number, q: number, score: number) => void) => {
    visit(a + 1, -1, Math.log(.7) + emissions[a][0]);
    for (let len = 1; len <= durations.length && a + len <= n; len++) for (let q = 0; q < 3; q++)
      visit(a + len, q, Math.log(.1 * durations[len - 1]) + prefix[q][a + len] - prefix[q][a]);
  };
  const forward = new Array(n + 1).fill(-Infinity), backward = [...forward];
  forward[0] = 0; backward[n] = 0;
  for (let a = 0; a < n; a++) each(a, (b, _q, s) => { forward[b] = logadd(forward[b], forward[a] + s); });
  for (let a = n - 1; a >= 0; a--) each(a, (b, _q, s) => { backward[a] = logadd(backward[a], s + backward[b]); });
  const context = Array.from({ length: n }, () => new Array(10).fill(0));
  // Range-add polynomial moments rather than iterating every interval's frames.
  const diff = Array.from({ length: n + 1 }, () => new Float64Array(18));
  for (let a = 0; a < n; a++) each(a, (b, q, s) => {
    const mass = Math.exp(forward[a] + s + backward[b] - forward[n]);
    if (q < 0) { context[a][0] += mass; return; }
    const len = b - a, c = (.5 - a) / len, slope = 1 / len;
    const co = [mass, mass * c, mass * slope, mass * c * c, 2 * mass * c * slope, mass * slope * slope];
    for (let i = 0; i < 6; i++) { diff[a][6 * q + i] += co[i]; diff[b][6 * q + i] -= co[i]; }
  });
  const running = new Float64Array(18);
  for (let t = 0; t < n; t++) {
    for (let i = 0; i < 18; i++) running[i] += diff[t][i];
    for (let q = 0; q < 3; q++) {
      const r = 6 * q, c = 1 + 3 * q;
      context[t][c] = running[r];
      context[t][c + 1] = running[r + 1] + t * running[r + 2];
      context[t][c + 2] = running[r + 3] + t * running[r + 4] + t * t * running[r + 5];
    }
  }
  return context;
}

/** Separate frozen graph and cache; never feeds features into the primary decode. */
export async function vowelResidualFeatures(backend: InferenceBackend, manifest: ModelManifest,
  model: Pick<VowelResidualModel, "featureOutput" | "projection">, audio: Float32Array) {
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
    const z = h.weight.map((w, i) => Math.tanh(w.reduce((s, v, d) => s + v * r[d], h.bias[i])));
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

export function residualComparisons(m: VowelResidualModel, x: number[][], context: number[][],
  c: SequenceEvidenceCandidate): VowelResidualEvidence | null {
  const ids = [c.written, c.rival].map(s => [...s].map(p => m.tokens.indexOf(p) + 1));
  if (ids.some(r => !r.length || r.some(i => i < 1)) || !x.length || x.length !== context.length ||
    !x.every(r => vector(r, 32)) || !context.every(r => vector(r, 10))) return null;
  const normalized = x.map(r => r.map((v, d) => (v - m.center[d]) / m.scale[d]));
  const remove = (predictor: number[][]) => normalized.map((r, t) => r.map((v, d) =>
    v - predictor[d].reduce((s, w, i) => s + w * context[t][i], 0)));
  const lp = [readoutLogProbs(m.residual, remove(m.predictor)), readoutLogProbs(m.original, normalized),
    readoutLogProbs(m.shuffled, remove(m.shuffledPredictor))];
  const margins = lp.map(rows => sequenceLogProbability(rows, ids[0]) - sequenceLogProbability(rows, ids[1]));
  // Neutralize every non-consonant identity while retaining its mass and blank.
  // The full chains must still complete, but vowels cannot supply positive identity evidence.
  const softIds = [...m.tokens].flatMap((p, i) => "ءبتثجحخدذرزسشصضطظعغفقكلمنهوي".includes(p) ? [] : [i + 1]);
  const neutral = lp[0].map(r => {
    const row = [...r], mean = softIds.reduce((s, i) => logadd(s, r[i]), -Infinity) - Math.log(softIds.length);
    for (const i of softIds) row[i] = mean; return row;
  });
  const result = { ...c, residualMargin: margins[0], originalMargin: margins[1], shuffledMargin: margins[2],
    consonantMargin: sequenceLogProbability(neutral, ids[0]) - sequenceLogProbability(neutral, ids[1]) };
  return [result.residualMargin, result.originalMargin, result.shuffledMargin, result.consonantMargin].every(Number.isFinite) ? result : null;
}

export function carryVowelResidualEvidence(chunks: PhonemeEvidence[], certificates: VowelResidualEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e, i) => i !== first ? e : { ...e, phonemes: e.phonemes.map((p, j) => j ? p :
    { ...p, vowelResidualVetoes: [...(p.vowelResidualVetoes ?? []), ...certificates] }) });
}
export async function addVowelResidualEvidence(scorer: VowelResidualScorer | undefined, manifest: ModelManifest,
  audio: Float32Array, chunks: PhonemeEvidence[], index: ReferenceIndex, phrases?: PhraseAsset,
  options: CheckerOptions = {}, mode: "free" | "memorization" = "free",
  inspect?: (result: VowelResidualEvidence | null, request: SequenceEvidenceCandidate) => void): Promise<PhonemeEvidence[]> {
  if (!scorer || !audio.length) return chunks;
  const candidates = new Map<string, SequenceEvidenceCandidate>();
  const tracker = createTracker(index, { mode, phrases, acquireMs: 1000 });
  const checker = createChecker(index, { ...options, phrases, sequenceEvidence(c) {
    candidates.set(sequenceEvidenceKey(c), c); return true;
  } });
  for (const e of insertionCells(chunks)) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const features = await vowelResidualFeatures(scorer.backend, manifest, scorer.model, audio);
  const context = vowelContext(features.logProbs, scorer.model.vowels, scorer.model.durations);
  const certificates: VowelResidualEvidence[] = [];
  for (const c of candidates.values()) {
    const result = c.fromFrame >= 0 && c.toFrame <= context.length ? residualComparisons(scorer.model,
      features.observations.slice(c.fromFrame, c.toFrame), context.slice(c.fromFrame, c.toFrame), c) : null;
    inspect?.(result, c);
    if (result && vowelResidualVeto(result)) certificates.push(result);
  }
  return carryVowelResidualEvidence(chunks, certificates);
}
