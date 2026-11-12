/** A training-only fit to the maximum of dependent logit-normal frame probabilities. */
export interface RunConfidenceCalibration {
  schema: 1;
  modelSha256: string;
  inventory: string;
  mode: "selection" | "peak";
  /** Probability that an emitted occurrence is correct before observing its maximum. */
  prior: number;
  correct: { mean: number; sd: number };
  incorrect: { mean: number; sd: number };
  dependence: { correlation: number; shrinkage: number };
}

const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));

/** Positive lag-one dependence, shrunk to the training estimate for short fragments. */
export function runCorrelation(probabilities: readonly number[], prior: number, shrinkage: number): number {
  const n = probabilities.length;
  if (n < 3) return prior;
  const xs = probabilities.map(p => Math.log(clamp(p, 1e-6, 1 - 1e-6) / (1 - clamp(p, 1e-6, 1 - 1e-6))));
  const mean = xs.reduce((a, x) => a + x, 0) / n;
  const variance = xs.reduce((a, x) => a + (x - mean) ** 2, 0);
  const differences = xs.slice(1).reduce((a, x, i) => a + (x - xs[i]) ** 2, 0);
  const rho = variance > 1e-12 ? clamp(1 - differences / (2 * variance), 0, 0.95) : prior;
  const weight = (n - 2) / (n - 2 + shrinkage);
  return weight * rho + (1 - weight) * prior;
}

/** Normal log CDF via a relative-error erfc approximation, evaluated in log space. */
function logNormalCdf(x: number): number {
  const a = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + a / 2);
  // Numerical Recipes erfc approximation; relative error below 1.2e-7.
  const logTail = Math.log(t / 2) - a * a - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
    t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
      t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
  return x >= 0 ? Math.log1p(-Math.exp(logTail)) : logTail;
}

export function occurrenceConfidence(probabilities: readonly number[], fit: RunConfidenceCalibration): number {
  if (!probabilities.length) throw new Error("A confidence run must contain a frame");
  const peak = clamp(Math.max(...probabilities), 1e-6, 1 - 1e-6);
  const x = Math.log(peak / (1 - peak));
  const rho = runCorrelation(probabilities, fit.dependence.correlation, fit.dependence.shrinkage);
  // AR(1) effective sample size, with the first opportunity always retained.
  const opportunities = fit.mode === "peak" ? 1 : 1 + (probabilities.length - 1) * (1 - rho) / (1 + rho);
  const density = ({ mean, sd }: { mean: number; sd: number }) => {
    const z = (x - mean) / sd;
    // log(n) and the logit Jacobian cancel between correctness classes.
    return -Math.log(sd) - z * z / 2 + (opportunities - 1) * logNormalCdf(z);
  };
  const odds = Math.log(fit.prior / (1 - fit.prior)) + density(fit.correct) - density(fit.incorrect);
  return odds >= 0 ? 1 / (1 + Math.exp(-odds)) : Math.exp(odds) / (1 + Math.exp(odds));
}

/** Reject incompatible or malformed assets before they can reinterpret a different listener's outputs. */
export function validateRunConfidence(value: unknown, modelSha256: string, inventory: string): RunConfidenceCalibration {
  const c = value as RunConfidenceCalibration;
  if (c?.schema !== 1 || c.modelSha256 !== modelSha256 || c.inventory !== inventory ||
      !["selection", "peak"].includes(c.mode) || !(c.prior > 0 && c.prior < 1) ||
      ![c.correct?.mean, c.incorrect?.mean].every(Number.isFinite) ||
      ![c.correct?.sd, c.incorrect?.sd].every(x => Number.isFinite(x) && x > 0) ||
      !(c.dependence?.correlation >= 0 && c.dependence.correlation <= 0.95) ||
      !(Number.isFinite(c.dependence?.shrinkage) && c.dependence.shrinkage >= 0)) {
    throw new Error("Invalid or incompatible run-confidence calibration");
  }
  return c;
}
