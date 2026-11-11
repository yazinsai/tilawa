import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, TemporalEnvelopeEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { insertionCells } from "./seam-insertion.js";

export interface EnvelopeEmission { mean: number[]; variance: number[] }
export interface EnvelopeModel {
  schema: 1;
  inventory: string;
  tokens: string;
  frameMs: 40;
  order: 2 | 4;
  ridge: number;
  maxFrames: number;
  phoneCounts: number[];
  envelope: EnvelopeEmission[];
  control: EnvelopeEmission[];
  envelopeScale: number[];
  controlScale: number[];
}
export interface EnvelopeFrame { envelope: number[]; control: number[] }

const BANDS = [80, 400, 900, 1600, 2800, 4500, 7600];
const SIZE = 2048, HOP = 640, CENTER = 200;

/** Radix-two FFT also implements the even-extension DCT used by FDLP. */
export function envelopeFFT(real: Float64Array, imag: Float64Array): void {
  const n = real.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [real[i], real[j]] = [real[j], real[i]]; [imag[i], imag[j]] = [imag[j], imag[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const angle = -2 * Math.PI / len, wr = Math.cos(angle), wi = Math.sin(angle);
    for (let start = 0; start < n; start += len) {
      let r = 1, s = 0;
      for (let j = 0; j < len / 2; j++) {
        const a = start + j, b = a + len / 2;
        const x = real[b] * r - imag[b] * s, y = real[b] * s + imag[b] * r;
        real[b] = real[a] - x; imag[b] = imag[a] - y; real[a] += x; imag[a] += y;
        [r, s] = [r * wr - s * wi, r * wi + s * wr];
      }
    }
  }
}

/** Stable reflection coefficients of prediction along frequency, not time.
 * Its all-pole spectrum is the subband's temporal energy envelope. */
export function frequencyPrediction(coefficients: number[], order: number, ridge: number): number[] {
  const correlation = Array.from({ length: order + 1 }, (_, lag) =>
    coefficients.slice(lag).reduce((s, x, i) => s + x * coefficients[i], 0));
  let error = correlation[0] * (1 + ridge) + 1e-12;
  let a: number[] = [], reflection: number[] = [];
  for (let k = 0; k < order; k++) {
    const numerator = correlation[k + 1] + a.reduce((s, v, j) => s + v * correlation[k - j], 0);
    const r = Math.max(-.999999, Math.min(.999999, -numerator / error));
    a = [...a.map((v, j) => v + r * a[k - 1 - j]), r];
    error *= 1 - r * r;
    reflection.push(Math.atanh(r));
  }
  return reflection;
}

/** Multi-frame FDLP and an equal-dimensional ordinary frame-energy view.
 * Both use the same samples, bands and energy normalization. */
export function envelopeFrame(window: Float64Array, order: 2 | 4, ridge: number): EnvelopeFrame {
  const mean = window.reduce((s, x) => s + x, 0) / SIZE;
  const x = Float64Array.from(window, v => v - mean);
  const energy = x.reduce((s, v) => s + v * v, 0) / SIZE + 1e-12;
  const re = new Float64Array(2 * SIZE), im = new Float64Array(2 * SIZE);
  for (let i = 0; i < SIZE; i++) re[i] = re[2 * SIZE - 1 - i] = x[i];
  envelopeFFT(re, im);
  const dct = Float64Array.from({ length: SIZE }, (_, k) =>
    (re[k] * Math.cos(Math.PI * k / (2 * SIZE)) + im[k] * Math.sin(Math.PI * k / (2 * SIZE))) / 2);
  const envelope: number[] = [Math.log(energy)], control: number[] = [Math.log(energy)];
  for (let band = 0; band < 6; band++) {
    const lo = Math.ceil(BANDS[band] * 2 * SIZE / 16000), hi = Math.floor(BANDS[band + 1] * 2 * SIZE / 16000);
    const c = Array.from(dct.subarray(lo, hi), (v, k) => v * (.54 - .46 * Math.cos(2 * Math.PI * k / (hi - lo - 1))));
    const power = c.reduce((s, v) => s + v * v, 0) / ((hi - lo) * SIZE * energy);
    envelope.push(Math.log(power + 1e-12), ...frequencyPrediction(c, order, ridge));
  }
  // Conventional 25 ms spectra at uniformly spaced positions inside the same
  // 128 ms interval. Each band has order+1 observations, matching the pole arm.
  const snapshots: number[][] = [];
  for (let t = 0; t <= order; t++) {
    const off = Math.round(t * (SIZE - 400) / order);
    const r = new Float64Array(512), s = new Float64Array(512);
    for (let j = 0; j < 400; j++) r[j] = x[off + j] * (.5 - .5 * Math.cos(2 * Math.PI * j / 399));
    envelopeFFT(r, s);
    snapshots.push(Array.from({ length: 6 }, (_, b) => {
      const lo = Math.ceil(BANDS[b] * 512 / 16000), hi = Math.floor(BANDS[b + 1] * 512 / 16000);
      let power = 0;
      for (let k = lo; k < hi; k++) power += r[k] * r[k] + s[k] * s[k];
      return Math.log(power / ((hi - lo) * 400 * energy) + 1e-12);
    }));
  }
  for (let b = 0; b < 6; b++) for (const frame of snapshots) control.push(frame[b]);
  return { envelope, control };
}

export class EnvelopeObservations {
  readonly frames: EnvelopeFrame[] = [];
  private audio: number[] = [];
  private base = 0;
  constructor(private readonly order: 2 | 4, private readonly ridge: number) {}
  push(samples: Float32Array, final = false): void {
    for (const x of samples) this.audio.push(x);
    const end = this.base + this.audio.length;
    while (final ? this.frames.length * HOP + 400 <= end : this.frames.length * HOP + CENTER + SIZE / 2 <= end) {
      const start = this.frames.length * HOP + CENTER - SIZE / 2;
      const window = Float64Array.from({ length: SIZE }, (_, j) => this.audio[start + j - this.base] ?? 0);
      this.frames.push(envelopeFrame(window, this.order, this.ridge));
    }
    const discard = Math.max(0, this.frames.length * HOP + CENTER - SIZE / 2 - this.base);
    this.audio = this.audio.slice(discard); this.base += discard;
  }
}
export function envelopeFeatures(audio: Float32Array, order: 2 | 4, ridge: number): EnvelopeFrame[] {
  const o = new EnvelopeObservations(order, ridge); o.push(audio, true); return o.frames;
}
export function envelopeLogadd(a: number, b: number): number {
  const m = Math.max(a, b);
  return Number.isFinite(m) ? m + Math.log(Math.exp(a - m) + Math.exp(b - m)) : -Infinity;
}

/** Left-to-right phone/silence graph. All phones are obligatory, including
 * genuine repetitions. Sum boundaries; normalize by the identical timing law's
 * partition, so sequence length alone provides no acoustic advantage. */
export function scoreEnvelopeSequence(m: EnvelopeModel, raw: EnvelopeFrame[], pronunciation: string, temporal: boolean): number | null {
  const ids = [...pronunciation].map(p => m.tokens.indexOf(p) + 1);
  if (!ids.length || !raw.length || raw.length > m.maxFrames || ids.some(id => !id || !m.phoneCounts[id - 1])) return null;
  const states = [0, ...ids.flatMap(id => [id, 0])], size = states.length;
  const emissions = temporal ? m.envelope : m.control, scale = temporal ? m.envelopeScale : m.controlScale;
  let score = Array<number>(size).fill(-Infinity), prior = score.slice();
  score[0] = prior[0] = Math.log(.5); score[1] = prior[1] = Math.log(.5);
  for (let t = 0; t < raw.length; t++) {
    const x = temporal ? raw[t].envelope : raw[t].control;
    if (x.length !== scale.length || x.some(v => !Number.isFinite(v))) return null;
    if (t) {
      const next = score.map(() => -Infinity), nextPrior = next.slice();
      for (let s = 0; s < size; s++) {
        const destinations = s === size - 1 ? [[s, 1]] : s % 2 ? [[s, .6], [s + 1, .2], ...(s + 2 < size ? [[s + 2, .2]] : [[s + 1, .2]])] : [[s, .6], [s + 1, .4]];
        for (const [to, p] of destinations) {
          next[to] = envelopeLogadd(next[to], score[s] + Math.log(p));
          nextPrior[to] = envelopeLogadd(nextPrior[to], prior[s] + Math.log(p));
        }
      }
      score = next; prior = nextPrior;
    }
    const row = emissions.map(e => x.reduce((s, v, d) => s - .5 * (Math.log(2 * Math.PI * e.variance[d]) + (v / scale[d] - e.mean[d]) ** 2 / e.variance[d]), 0));
    score = score.map((v, s) => v + row[states[s]]);
  }
  const value = envelopeLogadd(score[size - 1], score[size - 2]) - envelopeLogadd(prior[size - 1], prior[size - 2]);
  return Number.isFinite(value) ? value : null;
}
export function envelopeComparison(m: EnvelopeModel, raw: EnvelopeFrame[], c: SequenceEvidenceCandidate): TemporalEnvelopeEvidence | null {
  const controlWritten = scoreEnvelopeSequence(m, raw, c.written, false), controlRival = scoreEnvelopeSequence(m, raw, c.rival, false);
  const envelopeWritten = scoreEnvelopeSequence(m, raw, c.written, true), envelopeRival = scoreEnvelopeSequence(m, raw, c.rival, true);
  if (controlWritten === null || controlRival === null || envelopeWritten === null || envelopeRival === null) return null;
  return { ...c, controlWritten, controlRival, envelopeWritten, envelopeRival };
}
export const envelopeReversal = (c: TemporalEnvelopeEvidence): boolean =>
  [c.controlWritten, c.controlRival, c.envelopeWritten, c.envelopeRival].every(Number.isFinite) &&
  c.controlWritten < c.controlRival && c.envelopeWritten > c.envelopeRival;

export function validateEnvelopeModel(value: unknown, inventory: string, tokens: string): EnvelopeModel {
  const m = value as EnvelopeModel;
  const vector = (v: number[], n: number, positive = false) => Array.isArray(v) && v.length === n && v.every(x => Number.isFinite(x) && (!positive || x > 0));
  if (!m || m.schema !== 1 || m.inventory !== inventory || m.tokens !== tokens || m.frameMs !== 40 ||
    ![2, 4].includes(m.order) || !Number.isFinite(m.ridge) || m.ridge <= 0 || m.ridge > 1 ||
    !Number.isInteger(m.maxFrames) || m.maxFrames < 1 || m.maxFrames > 4096 ||
    !vector(m.phoneCounts, tokens.length) || m.phoneCounts.some(c => c < 0)) throw new Error("Invalid temporal envelope model");
  const dimensions = 1 + 6 * (m.order + 1);
  for (const arm of ["envelope", "control"] as const) {
    if (!vector(m[`${arm}Scale`], dimensions, true) || !Array.isArray(m[arm]) || m[arm].length !== tokens.length + 1 ||
      m[arm].some(e => !e || !vector(e.mean, dimensions) || !vector(e.variance, dimensions, true))) throw new Error("Invalid temporal envelope emissions");
  }
  return m;
}

/** Discover only existing confirmations and certify their exact complete comparison. */
export function addEnvelopeEvidence(model: EnvelopeModel | undefined, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset, options: CheckerOptions = {}, mode: "free" | "memorization" = "free",
  inspect?: (c: TemporalEnvelopeEvidence | null, candidate: SequenceEvidenceCandidate) => void): PhonemeEvidence[] {
  if (!model || !audio.length) return chunks;
  const cells = insertionCells(chunks);
  if (!cells.length) return chunks;
  if (cells[0].frameMs !== model.frameMs) throw new Error("Envelope scorer clock differs from primary evidence");
  const candidates = new Map<string, SequenceEvidenceCandidate>();
  const tracker = createTracker(index, { mode, phrases, acquireMs: 1000 });
  const checker = createChecker(index, { ...options, phrases, sequenceEvidence(c) { candidates.set(JSON.stringify(c), c); return true; } });
  for (const e of cells) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const features = envelopeFeatures(audio, model.order, model.ridge), certificates: TemporalEnvelopeEvidence[] = [];
  for (const c of candidates.values()) {
    const comparison = c.fromFrame >= 0 && c.toFrame <= features.length ? envelopeComparison(model, features.slice(c.fromFrame, c.toFrame), c) : null;
    inspect?.(comparison, c);
    if (comparison && envelopeReversal(comparison)) certificates.push(comparison);
  }
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e, i) => i !== first ? e : { ...e, phonemes: e.phonemes.map((p, j) => j ? p : { ...p, envelopeVetoes: [...(p.envelopeVetoes ?? []), ...certificates] }) });
}
