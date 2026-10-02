/**
 * Logistic slip head over frozen Zipformer encoder frames.
 *
 * A word's score is the probe's L2 logistic on the concatenated mean and max
 * of its tracker span, padded by one frame. The head decides whether to flag;
 * the kind still comes from the aligner. Weights are the a0w refit
 * (`slip-weights.json`). No encoder frames → the session never reads this.
 */

import BUNDLED from "./slip-weights.json" with { type: "json" };

export const ENCODER_DIM = 512;
/** Frames added on each side of a tracker word span. Matches the probe. */
export const SLIP_PAD = 1;
/** Spans longer than this are cropped to the middle. Matches `build_units.py`. */
export const SLIP_SPAN_CAP = 80;

export type SlipSensitivity = "strict" | "high";

export interface SlipHead {
  /** Train-set mean of `[mean ‖ max]`, length 1024. */
  mu: Float32Array;
  /** Train-set std, zeros already replaced by 1. Length 1024. */
  sd: Float32Array;
  /** Logistic coefficients on the standardized features. Length 1024. */
  coef: Float32Array;
  intercept: number;
  /** P(slip) cutoff: no extra false flags. Test-informed (0.987). */
  strict: number;
  /** P(slip) cutoff: sensitivity trade-off, about +0.10 flags per minute. */
  high: number;
}

export interface SlipWeightJson {
  mu: number[];
  sd: number[];
  coef: number[];
  intercept: number;
  thresholds: { strict: number; high: number };
}

/** Ring of encoder rows indexed by decoder frame. Cleared with the decoder. */
export class EncoderFrames {
  readonly dim: number;
  readonly capacity: number;
  private readonly data: Float32Array;
  private start = 0;
  private end = 0;

  constructor(dim = ENCODER_DIM, capacity = 20000) {
    this.dim = dim;
    this.capacity = capacity;
    this.data = new Float32Array(dim * capacity);
  }

  get firstFrame(): number {
    return this.start;
  }

  get endFrame(): number {
    return this.end;
  }

  /** Append `frames` rows (row-major `[frames, dim]`) starting at decoder frame `firstFrame`. */
  push(rows: ArrayLike<number>, frames: number, firstFrame: number): void {
    if (firstFrame !== this.end) this.clear(firstFrame);
    const d = this.dim;
    for (let t = 0; t < frames; t++) {
      const slot = this.end % this.capacity;
      const base = slot * d;
      const src = t * d;
      for (let k = 0; k < d; k++) this.data[base + k] = rows[src + k] ?? 0;
      this.end++;
      if (this.end - this.start > this.capacity) this.start = this.end - this.capacity;
    }
  }

  clear(at = 0): void {
    this.start = at;
    this.end = at;
  }

  has(from: number, to: number): boolean {
    return from >= this.start && to <= this.end && from <= to;
  }

  row(frame: number, into: Float32Array, offset: number): void {
    const base = (frame % this.capacity) * this.dim;
    for (let k = 0; k < this.dim; k++) into[offset + k] = this.data[base + k]!;
  }
}

export function slipHeadFromJson(raw: SlipWeightJson): SlipHead {
  const n = ENCODER_DIM * 2;
  if (raw.mu.length !== n || raw.sd.length !== n || raw.coef.length !== n) {
    throw new Error(`slip head expects ${n} features, got mu ${raw.mu.length}`);
  }
  return {
    mu: Float32Array.from(raw.mu),
    sd: Float32Array.from(raw.sd),
    coef: Float32Array.from(raw.coef),
    intercept: raw.intercept,
    strict: raw.thresholds.strict,
    high: raw.thresholds.high,
  };
}

/** Shipped a0w logistic. Thresholds are the probe's OOF clean-pool cutoffs. */
export const A0W_SLIP_HEAD: SlipHead = slipHeadFromJson(BUNDLED as SlipWeightJson);

export function slipThreshold(head: SlipHead, sensitivity: SlipSensitivity): number {
  return sensitivity === "high" ? head.high : head.strict;
}

/**
 * Mean and max over encoder frames `[a, b)`, padded by `pad`, cropped the
 * same way the probe built its training rows. Null when the ring no longer
 * holds the slice.
 */
export function poolEncoderSpan(
  enc: EncoderFrames,
  a: number,
  b: number,
  pad = SLIP_PAD,
): { mean: Float32Array; max: Float32Array } | null {
  if (b < a) {
    const t = a;
    a = b;
    b = t;
  }
  if (b - a > SLIP_SPAN_CAP) {
    const mid = (a + b) >> 1;
    a = mid - SLIP_SPAN_CAP / 2;
    b = mid + SLIP_SPAN_CAP / 2;
  }
  const lo = Math.max(0, a - pad);
  const hi = Math.min(enc.endFrame, Math.max(lo + 1, b + pad));
  if (hi <= lo || !enc.has(lo, hi)) return null;
  const dim = enc.dim;
  const mean = new Float32Array(dim);
  const max = new Float32Array(dim);
  max.fill(-Infinity);
  const row = new Float32Array(dim);
  const n = hi - lo;
  for (let t = lo; t < hi; t++) {
    enc.row(t, row, 0);
    for (let k = 0; k < dim; k++) {
      const v = row[k]!;
      mean[k] += v;
      if (v > max[k]!) max[k] = v;
    }
  }
  for (let k = 0; k < dim; k++) mean[k] /= n;
  return { mean, max };
}

/** sklearn `predict_proba` of the standardized `[mean ‖ max]` vector. */
export function slipProbability(head: SlipHead, mean: Float32Array, max: Float32Array): number {
  let logit = head.intercept;
  const n = mean.length;
  for (let i = 0; i < n; i++) {
    const sd = head.sd[i]!;
    logit += ((mean[i]! - head.mu[i]!) / (sd === 0 ? 1 : sd)) * head.coef[i]!;
  }
  for (let i = 0; i < n; i++) {
    const j = n + i;
    const sd = head.sd[j]!;
    logit += ((max[i]! - head.mu[j]!) / (sd === 0 ? 1 : sd)) * head.coef[j]!;
  }
  if (logit >= 0) return 1 / (1 + Math.exp(-logit));
  const z = Math.exp(logit);
  return z / (1 + z);
}

export interface TrailWordSpan {
  surah: number;
  ayah: number;
  word: number;
  wordIndex: number;
  /** Encoder-frame interval `[a, b)` of the tracker's chars, before padding. */
  a: number;
  b: number;
  skipped: boolean;
}

/** The bits of a tracker `units_from_track` needs. */
export interface TrailLike {
  heard: readonly { frame: number }[];
  trail: readonly number[];
  firstWord: number;
  len: number;
  localWordOfPos: ArrayLike<number>;
  corpus: {
    wordSurah: ArrayLike<number>;
    wordAyah: ArrayLike<number>;
    wordInAyah: ArrayLike<number>;
  };
}

/**
 * Tracker chars → word spans in encoder frames. A skipped word takes its
 * share of the gap between the neighbours the tracker did emit. Same split
 * as `lab/scripts/head_probe/common.py` `units_from_track`.
 */
export function wordSpansFromTrail(tracker: TrailLike): TrailWordSpan[] {
  const words = new Map<number, number[]>();
  const n = Math.min(tracker.heard.length, tracker.trail.length);
  for (let i = 0; i < n; i++) {
    const cell = tracker.trail[i]!;
    const local = cell <= 0 ? 0 : tracker.localWordOfPos[Math.min(cell, tracker.len) - 1]!;
    const w = tracker.firstWord + local;
    let fs = words.get(w);
    if (!fs) words.set(w, (fs = []));
    fs.push(tracker.heard[i]!.frame);
  }
  const order = [...words.keys()].sort((p, q) => {
    const fp = words.get(p)!;
    const fq = words.get(q)!;
    return Math.min(...fp) - Math.min(...fq);
  });
  const c = tracker.corpus;
  const span = (w: number, a: number, b: number, skipped: boolean): TrailWordSpan => ({
    surah: c.wordSurah[w]!,
    ayah: c.wordAyah[w]!,
    word: c.wordInAyah[w]!,
    wordIndex: w,
    a,
    b,
    skipped,
  });
  const out: TrailWordSpan[] = [];
  for (let i = 0; i < order.length; i++) {
    const w = order[i]!;
    const fs = words.get(w)!;
    let a = Infinity;
    let b = -Infinity;
    for (const fr of fs) {
      if (fr < a) a = fr;
      if (fr > b) b = fr;
    }
    b += 1;
    out.push(span(w, a, b, false));
    if (i + 1 >= order.length) continue;
    const nxt = order[i + 1]!;
    if (c.wordSurah[nxt] !== c.wordSurah[w] || c.wordAyah[nxt] !== c.wordAyah[w]) continue;
    const nskip = c.wordInAyah[nxt]! - c.wordInAyah[w]! - 1;
    if (nskip <= 0 || nxt !== w + nskip + 1) continue;
    const gapA = b;
    const nfs = words.get(nxt)!;
    let gapB = Infinity;
    for (const fr of nfs) if (fr < gapB) gapB = fr;
    const width = Math.max(nskip, gapB - gapA);
    for (let j = 0; j < nskip; j++) {
      let sa = gapA + Math.floor((width * j) / nskip);
      let sb = gapA + Math.floor((width * (j + 1)) / nskip);
      if (sb <= sa) sb = sa + 1;
      out.push(span(w + 1 + j, sa, sb, true));
    }
  }
  return out;
}
