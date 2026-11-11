import { PosteriorFrames } from "./ctc.js";
import type { DecodedPhoneme } from "./evidence.js";

/** Speaker-separated, transcript-marginal training; no location or phone identity features. */
export interface BoundaryModel {
  mean: readonly number[];
  scale: readonly number[];
  weights: readonly number[];
  intercept: number;
  prior: number;
  temperature: number;
}

export interface BoundaryLikelihood {
  /** Log likelihood ratios against the pooled acoustic distribution, in nats. */
  boundary: number;
  within: number;
}

/**
 * Causal seam features: the previous spike's tail, the intervening rows, and the current spike's head.
 * Missing retained rows mean no evidence, rather than an invented blank/pause. Kept in sync with training.
 */
export function boundaryFeatures(
  frames: PosteriorFrames,
  previous: DecodedPhoneme,
  current: DecodedPhoneme,
  frameMs: number,
): number[] | null {
  if (current.start <= previous.end) return null;
  const left = Math.max(previous.start, previous.end - 3);
  const right = Math.min(current.end + 1, current.start + 4);
  if (left < frames.startFrame || right > frames.endFrame) return null;
  const summary = (from: number, to: number) => {
    const mean = new Float64Array(frames.row(left).length);
    let blank = 0, entropy = 0, peak = 0, maxBlank = 0;
    const n = to - from;
    for (let f = from; f < to; f++) {
      const row = frames.row(f);
      let h = 0, top = 0;
      for (let i = 0; i < row.length; i++) {
        const p = Math.exp(row[i]);
        mean[i] += p / n;
        h -= p * row[i];
        if (i) top = Math.max(top, p);
      }
      const b = Math.exp(row[0]);
      blank += b / n;
      entropy += h / Math.log(row.length) / n;
      peak += top / n;
      maxBlank = Math.max(maxBlank, b);
    }
    return { mean, blank, entropy, peak, maxBlank };
  };
  const a = summary(left, previous.end + 1);
  // No gap is an observed zero, with no blank row inserted.
  const g = summary(previous.end + 1, current.start);
  const b = summary(current.start, right);
  let overlap = 0, dot = 0, normA = 0, normB = 0;
  for (let i = 1; i < a.mean.length; i++) {
    overlap += Math.min(a.mean[i], b.mean[i]);
    dot += a.mean[i] * b.mean[i];
    normA += a.mean[i] ** 2;
    normB += b.mean[i] ** 2;
  }
  return [Math.log1p((current.start - previous.end - 1) * frameMs / 1000),
    a.blank, g.blank, b.blank, a.entropy, g.entropy, b.entropy,
    a.peak, g.peak, b.peak, overlap, dot / Math.max(1e-12, Math.sqrt(normA * normB)), g.maxBlank];
}

/** Divide out the training prior so choosing more boundaries receives no free prior credit. */
export function boundaryLikelihood(features: readonly number[], model: BoundaryModel): BoundaryLikelihood {
  let z = model.intercept;
  for (let i = 0; i < features.length; i++) z += model.weights[i] * (features[i] - model.mean[i]) / model.scale[i];
  z /= model.temperature;
  const softplus = (x: number) => Math.max(0, x) + Math.log1p(Math.exp(-Math.abs(x)));
  return { boundary: -softplus(-z) - Math.log(model.prior), within: -softplus(z) - Math.log1p(-model.prior) };
}
