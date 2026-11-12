import { PosteriorFrames } from "./ctc.js";
import type { DecodedPhoneme } from "./evidence.js";

/** Frozen speaker-separated interruption training; no reference-match features. */
export interface InterruptionModel {
  mean: readonly number[];
  scale: readonly number[];
  weights: readonly number[];
  intercept: number;
  temperature: number;
  threshold: number;
}

export interface WordAttemptState {
  /** Text matching alone does not say whether the speaker stopped. */
  lexical: "incomplete" | "complete";
  acoustic: "interrupted" | "continuous" | "unavailable";
}

/**
 * Causal seam features: the previous spike's tail, the intervening rows, and the current spike's head.
 * Missing retained rows mean no evidence, rather than an invented blank/pause. Kept in sync with training.
 */
export function interruptionFeatures(
  frames: PosteriorFrames,
  previous: DecodedPhoneme,
  current: DecodedPhoneme,
  frameMs: number,
): number[] | null {
  if (!Number.isFinite(frameMs) || frameMs <= 0 || current.start <= previous.end || frames.startFrame < 0) return null;
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

/** Probability that speech was interrupted, independently of its text match. */
export function interruptionProbability(features: readonly number[], model: InterruptionModel): number {
  if (features.length !== model.weights.length || model.mean.length !== features.length || model.scale.length !== features.length ||
    !Number.isFinite(model.temperature) || model.temperature <= 0 || model.scale.some(s => !Number.isFinite(s) || s <= 0) ||
    [...features, ...model.weights, ...model.mean, model.intercept].some(x => !Number.isFinite(x))) throw new Error("Invalid interruption model or features");
  let z = model.intercept;
  for (let i = 0; i < features.length; i++) z += model.weights[i] * (features[i] - model.mean[i]) / model.scale[i];
  z /= model.temperature;
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

export function wordAttemptState(coveredPhones: number, length: number, probability: number | null, model: InterruptionModel): WordAttemptState {
  if (!(model.threshold > 0 && model.threshold < 1)) throw new Error("Invalid interruption threshold");
  if (probability !== null && (!Number.isFinite(probability) || probability < 0 || probability > 1)) throw new Error("Invalid interruption probability");
  return { lexical: coveredPhones < length ? "incomplete" : "complete",
    acoustic: probability === null ? "unavailable" : probability >= model.threshold ? "interrupted" : "continuous" };
}

/** A partial lexical match cannot satisfy the acoustic cutoff exception. */
export function restartAfterAttempt(state: WordAttemptState, pause: boolean): boolean {
  return pause || state.acoustic === "interrupted";
}
