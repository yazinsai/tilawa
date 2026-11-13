/** Causal within-word summaries; no location, speaker, or recording identifiers enter the model. */
export interface WordPrefix {
  readonly word: number;
  readonly entry: number;
  readonly progress: number;
  readonly length: number;
  readonly observations: number;
  readonly matches: number;
  readonly editCost: number;
  readonly lastCost: number;
  readonly stalled: number;
}

export interface CompletionExample {
  prefix: WordPrefix;
  /** Future observed tokens until completion, abandonment, or the recording's right censor. */
  delay: number;
  outcome: "complete" | "abandon" | "censored";
  weight?: number;
}

export interface CompletionModel {
  weights: number[];
  endpointOnly: boolean;
  /** Competing event waiting times, in observed tokens. */
  completeRate: number;
  abandonRate: number;
}

const sigmoid = (z: number) => z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
const bounded = (x: number) => Math.max(1e-12, Math.min(1 - 1e-12, x));

/** Eight features in both models. Endpoint control sees only current progress, length, and accumulated cost. */
export function completionFeatures(p: WordPrefix, endpointOnly: boolean): number[] {
  const length = Math.max(1, p.length), progress = p.progress / length;
  const cost = p.editCost / 10 / length;
  return endpointOnly
    ? [1, progress, 1 / length, cost, progress * progress, cost * cost, progress * cost, progress / length]
    : [1, progress, 1 / length, cost, p.observations / length, p.matches / Math.max(1, p.observations), p.lastCost / 10, p.stalled / length];
}

/** Probability that the partial lexical match still has a transcript-consistent completion ahead of it. */
export function completionProbability(model: CompletionModel, p: WordPrefix): number {
  const x = completionFeatures(p, model.endpointOnly);
  return bounded(sigmoid(x.reduce((z, v, i) => z + v * model.weights[i], 0)));
}

const survival = (rate: number, delay: number) => Math.pow(1 - rate, Math.max(0, delay));
const event = (rate: number, delay: number) => rate * survival(rate, Math.max(0, delay - 1));

/** Censored examples supply a survival likelihood, never an abandonment label. */
export function completionLogLikelihood(model: CompletionModel, examples: readonly CompletionExample[]): number {
  return examples.reduce((sum, e) => {
    const p = completionProbability(model, e.prefix);
    const likelihood = e.outcome === "complete" ? p * event(model.completeRate, e.delay)
      : e.outcome === "abandon" ? (1 - p) * event(model.abandonRate, e.delay)
      : p * survival(model.completeRate, e.delay) + (1 - p) * survival(model.abandonRate, e.delay);
    return sum + (e.weight ?? 1) * Math.log(Math.max(1e-300, likelihood));
  }, 0);
}

/** Ridge logistic cure mixture with geometric competing waiting times. EM includes right-censored prefixes. */
export function fitCompletionModel(examples: readonly CompletionExample[], endpointOnly: boolean, ridge: number): CompletionModel {
  if (!examples.length || !examples.some(e => e.outcome === "complete") || !examples.some(e => e.outcome === "abandon")) {
    throw new Error("Completion training requires observed completions and abandonments");
  }
  const model: CompletionModel = { weights: Array(8).fill(0), endpointOnly, completeRate: 0.1, abandonRate: 0.1 };
  const xs = examples.map(e => completionFeatures(e.prefix, endpointOnly));
  const rowNorms = xs.map(x => x.reduce((s, v) => s + Math.abs(v), 0));
  const probability = (i: number) => bounded(sigmoid(xs[i].reduce((z, v, j) => z + v * model.weights[j], 0)));
  // Fixed training compute; the regularizer is selected only in speaker-separated training partitions.
  for (let em = 0; em < 40; em++) {
    const targets = examples.map((e, i) => {
      if (e.outcome !== "censored") return e.outcome === "complete" ? 1 : 0;
      const p = probability(i);
      const c = p * survival(model.completeRate, e.delay), a = (1 - p) * survival(model.abandonRate, e.delay);
      return c / Math.max(1e-300, c + a);
    });
    for (let iter = 0; iter < 30; iter++) {
      const grad = model.weights.map((w, j) => j === 0 ? 0 : -ridge * w);
      const curvature = model.weights.map((_, j) => j === 0 ? 1e-6 : ridge);
      for (let i = 0; i < examples.length; i++) {
        const p = probability(i), weight = examples[i].weight ?? 1;
        for (let j = 0; j < 8; j++) {
          grad[j] += weight * (targets[i] - p) * xs[i][j];
          // A diagonal majorizer bounds the logistic Hessian's row sums.
          curvature[j] += weight * 0.25 * Math.abs(xs[i][j]) * rowNorms[i];
        }
      }
      for (let j = 0; j < 8; j++) model.weights[j] += grad[j] / curvature[j];
    }
    let ce = 0.5, ae = 0.5, ct = 1, at = 1;
    examples.forEach((e, i) => {
      const w = e.weight ?? 1, p = targets[i];
      ct += w * p * Math.max(1, e.delay); at += w * (1 - p) * Math.max(1, e.delay);
      if (e.outcome === "complete") ce += w;
      if (e.outcome === "abandon") ae += w;
    });
    model.completeRate = bounded(ce / ct); model.abandonRate = bounded(ae / at);
  }
  return model;
}

/**
 * A departure marginalizes a word already completed through recognition loss versus a pending word whose suffix
 * must be discharged. Ordinary open frontiers remain free. Suffix cost uses the unchanged phone deletion channel;
 * 3 nats/edit is the tracker's existing evidence conversion, rather than a practice-fitted multiplier.
 */
export function completionDepartureCost(model: CompletionModel, p: WordPrefix, suffixCost: number): number {
  if (p.progress >= p.length || suffixCost <= 0) return 0;
  const pending = completionProbability(model, p);
  return -Math.log((1 - pending) + pending * Math.exp(-3 * suffixCost / 10)) * 10 / 3;
}
