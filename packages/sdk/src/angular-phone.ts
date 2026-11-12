import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { AngularPhoneEvidence, CheckerEvent, CheckerOptions, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface AngularModel {
  schema: 1;
  inventory: string;
  tokens: string;
  dimensions: number;
  primarySha256: string;
  featureOutput: string;
  angular: { vectors: number[][]; concentration: number };
  control: { weights: number[][]; bias: number[] };
  /** Minimum/maximum native norm among acoustically occupied training frames. */
  normRange: [number, number];
}
const dot = (a: number[], b: number[]) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const norm = (x: number[]) => Math.sqrt(dot(x, x));
const logadd = (a: number, b: number): number => {
  const m = Math.max(a, b);
  return Number.isFinite(m) ? m + Math.log(Math.exp(a - m) + Math.exp(b - m)) : -Infinity;
};

/** Ordinary complete CTC forward sum, including repeated-phone blank constraints. */
export function featureBoundaryScore(values: number[][], ids: number[]): number {
  if (!values.length || !ids.length) return -Infinity;
  const labels = [0, ...ids.flatMap(p => [p, 0])];
  let a = labels.map(() => -Infinity);
  a[0] = values[0][0];
  a[1] = values[0][ids[0]];
  for (let t = 1; t < values.length; t++) {
    a = labels.map((p, s) => {
      let mass = a[s];
      if (s > 0) mass = logadd(mass, a[s - 1]);
      if (s > 1 && p !== 0 && p !== labels[s - 2]) mass = logadd(mass, a[s - 2]);
      return mass + values[t][p];
    });
  }
  return logadd(a[a.length - 1], a[a.length - 2]);
}

/** Only identity changes: the primary blank and total nonblank mass are exact inputs. */
export function identityRows(m: AngularModel, raw: number[][], primary: number[][], arm: "angular" | "control"): number[][] {
  if (raw.length !== primary.length) throw new Error("Features and primary frames differ");
  return raw.map((x, t) => {
    const blank = primary[t][0];
    if (Number.isNaN(blank) || blank > 0) throw new Error("Invalid primary blank log probability");
    const magnitude = norm(x);
    if (!(magnitude > 0) || !Number.isFinite(magnitude)) throw new Error("Undefined feature direction");
    const logits = arm === "angular"
      ? m.angular.vectors.map(w => m.angular.concentration * dot(w, x) / magnitude)
      : m.control.weights.map((w, i) => dot(w, x) + m.control.bias[i]);
    const max = Math.max(...logits);
    const z = max + Math.log(logits.reduce((sum, v) => sum + Math.exp(v - max), 0));
    // Stable log(1-exp(blank)); no floor that could manufacture speech mass.
    const speech = blank < -Math.LN2 ? Math.log1p(-Math.exp(blank)) : Math.log(-Math.expm1(blank));
    return [blank, ...logits.map(v => speech + v - z)];
  });
}
export function featureScores(m: AngularModel, x: number[][], primary: number[][], sequences: number[][], arm: "angular" | "control"): number[] {
  const rows = identityRows(m, x, primary, arm);
  return sequences.map(s => featureBoundaryScore(rows, s));
}
export function featureReversal(c: AngularPhoneEvidence): boolean {
  return [c.angularWritten, c.angularRival, c.controlWritten, c.controlRival].every(Number.isFinite) &&
    c.angularWritten > c.angularRival && c.controlWritten < c.controlRival;
}
export function featureComparison(m: AngularModel, x: number[][], primary: number[][], candidate: SequenceEvidenceCandidate): AngularPhoneEvidence | null {
  const sequences = [candidate.written, candidate.rival].map(s => [...s].map(p => m.tokens.indexOf(p) + 1));
  if (!x.length || x.length !== primary.length || !x.every(r => r.length === m.dimensions && r.every(Number.isFinite) && norm(r) > 0) ||
    primary.some(r => r.length !== m.tokens.length + 1 || r.some(v => Number.isNaN(v)) || r[0] > 0) ||
    sequences.some(s => !s.length || s.some(p => p < 1))) return null;
  const a = featureScores(m, x, primary, sequences, "angular"), b = featureScores(m, x, primary, sequences, "control");
  const c = { ...candidate, angularWritten: a[0], angularRival: a[1], controlWritten: b[0], controlRival: b[1] };
  return [a[0], a[1], b[0], b[1]].every(Number.isFinite) ? c : null;
}

/** Analysis only, never selection or a warning gate. Preserve every vector direction and
 * primary blank while sweeping common radii inside the observed training interval. */
export function magnitudeProbe(m: AngularModel, x: number[][], primary: number[][], candidate: SequenceEvidenceCandidate) {
  const original = featureComparison(m, x, primary, candidate);
  if (!original) return null;
  const [low, high] = m.normRange;
  const sweep = Array.from({ length: 33 }, (_, i) => {
    const radius = Math.exp(Math.log(low) + (Math.log(high) - Math.log(low)) * i / 32);
    const scaled = x.map(r => r.map(v => v * radius / norm(r)));
    const c = featureComparison(m, scaled, primary, candidate)!;
    return { radius, affineMargin: c.controlWritten - c.controlRival, angularMargin: c.angularWritten - c.angularRival };
  });
  return { original, sweep, magnitudeSensitive: original.controlWritten < original.controlRival &&
    sweep.some(s => s.affineMargin > 0) && sweep.some(s => s.affineMargin < 0),
    angularWritten: original.angularWritten > original.angularRival };
}
export function angularVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity = (e: CheckerEvent) => JSON.stringify([e.id, e.timeMs, e.location, e.expected, e.observed]);
  const kept = new Set(reviewed.map(identity));
  const vetoed = new Set(baseline.filter(e => e.errorClass === "wrong_word" && !kept.has(identity(e))).map(identity));
  return live.filter(e => e.errorClass === "wrong_word" && vetoed.has(identity(e)));
}
export function carryAngularEvidence(chunks: PhonemeEvidence[], certificates: AngularPhoneEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e, i) => i !== first ? e : { ...e, phonemes: e.phonemes.map((p, j) => j ? p : { ...p, angularPhoneVetoes: certificates }) });
}
export function validateAngularModel(value: unknown, manifest: ModelManifest): AngularModel {
  const m = value as AngularModel;
  const vector = (r: unknown, n: number): r is number[] => Array.isArray(r) && r.length === n && r.every(Number.isFinite);
  if (!m || m.schema !== 1 || m.inventory !== manifest.inventory || m.tokens !== manifest.tokens || m.primarySha256 !== manifest.sha256 ||
    !Number.isSafeInteger(m.dimensions) || m.dimensions <= 0 || typeof m.featureOutput !== "string" || !m.featureOutput ||
    !vector(m.normRange, 2) || m.normRange[0] <= 0 || m.normRange[1] < m.normRange[0] ||
    !m.angular || !Number.isFinite(m.angular.concentration) || m.angular.concentration <= 0 ||
    m.angular.vectors?.length !== m.tokens.length || !m.angular.vectors.every(r => vector(r, m.dimensions) && Math.abs(norm(r) - 1) < 1e-5) ||
    !m.control || m.control.weights?.length !== m.tokens.length || !m.control.weights.every(r => vector(r, m.dimensions)) ||
    !vector(m.control.bias, m.tokens.length)) throw new Error("Invalid angular feature model");
  return m;
}

/** A separate native state instance exposes raw pre-head features on the primary clock. */
export async function featureObservations(backend: InferenceBackend, manifest: ModelManifest, model: AngularModel, audio: Float32Array) {
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
    const tensor = out[model.featureOutput], width = model.dimensions;
    if (!Number.isInteger(n) || n <= 0 || !tensor || tensor.type !== "float32" || tensor.data.length !== n * width)
      throw new Error("Invalid pre-head feature geometry");
    for (let t = 0; t < n && observations.length < total; t++) {
      observations.push(Array.from(tensor.data.slice(t * width, (t + 1) * width), Number));
      logProbs.push(Array.from(out.log_probs.data.slice(t * vocab, (t + 1) * vocab), Number));
    }
  }
  return { observations, logProbs };
}
export async function addAngularEvidence(scorer: { model: AngularModel; backend: InferenceBackend } | undefined, manifest: ModelManifest,
  audio: Float32Array, chunks: PhonemeEvidence[], index: ReferenceIndex, phrases?: PhraseAsset, options: CheckerOptions = {},
  mode: "free" | "memorization" = "free", inspect?: (c: AngularPhoneEvidence | null, request: SequenceEvidenceCandidate, probe: ReturnType<typeof magnitudeProbe>) => void): Promise<PhonemeEvidence[]> {
  if (!scorer || !audio.length) return chunks;
  const candidates = new Map<string, SequenceEvidenceCandidate>();
  const tracker = createTracker(index, { mode, phrases, acquireMs: 1000 });
  const checker = createChecker(index, { ...options, phrases, sequenceEvidence(c) { candidates.set(JSON.stringify(c), c); return true; } });
  for (const e of insertionCells(chunks)) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const features = await featureObservations(scorer.backend, manifest, scorer.model, audio), certificates: AngularPhoneEvidence[] = [];
  for (const c of candidates.values()) {
    const valid = c.fromFrame >= 0 && c.toFrame <= features.observations.length;
    const x = features.observations.slice(c.fromFrame, c.toFrame), primary = features.logProbs.slice(c.fromFrame, c.toFrame);
    const result = valid ? featureComparison(scorer.model, x, primary, c) : null;
    if (inspect) inspect(result, c, valid ? magnitudeProbe(scorer.model, x, primary, c) : null);
    if (result && featureReversal(result)) certificates.push(result);
  }
  return carryAngularEvidence(chunks, certificates);
}
