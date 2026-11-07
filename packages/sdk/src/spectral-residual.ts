import type { SpectralResidualRequest } from "@tilawa/core";
import { Fbank } from "./fbank.js";

export interface SpectralResidualParameters {
  schema: 1;
  tokens: string;
  /** Orthonormal FFT-bin directions annihilated by the unchanged mel matrix. */
  projection: number[][];
  /** Conditional prediction from intercept and normalized mel powers. */
  predictor: number[][];
  scale: number[];
  means: number[][];
  variance: number[];
  covered: boolean[];
}
export const logadd = (a: number, b: number): number => a === -Infinity ? b : b === -Infinity ? a :
  Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));

/** CTC placements with a uniform prior over all legal paths, including repeated-label blank separation. */
export function residualLikelihood(rows: number[][], labels: number[]): number {
  if (!rows.length || !labels.length) return -Infinity;
  const states = [0, ...labels.flatMap(p => [p, 0])];
  function forward(emissions: boolean): number {
    let prev = new Array(states.length).fill(-Infinity);
    for (let t = 0; t < rows.length; t++) {
      const next = new Array(states.length).fill(-Infinity);
      for (let s = 0; s < states.length; s++) {
        let mass = t === 0 ? (s <= 1 ? 0 : -Infinity) : prev[s];
        if (t > 0 && s > 0) mass = logadd(mass, prev[s - 1]);
        if (t > 0 && s > 1 && states[s] && states[s] !== states[s - 2]) mass = logadd(mass, prev[s - 2]);
        next[s] = mass + (emissions ? rows[t][states[s]] : 0);
      }
      prev = next;
    }
    return logadd(prev.at(-1)!, prev.at(-2)!);
  }
  const count = forward(false);
  return Number.isFinite(count) ? forward(true) - count : -Infinity;
}

/** Every original 25 ms window and 10 ms shift is retained, with no crop re-phasing. */
export function spectralObservations(audio: Float32Array, p: SpectralResidualParameters): Float32Array[] {
  const out: Float32Array[] = [];
  new Fbank({ onSpectrum(power, mel) {
    const energies = Array.from(mel, x => Math.exp(x));
    const total = energies.reduce((a, b) => a + b, 0);
    const conditioned = [1, ...energies.map(x => x / total)];
    const residual = Float32Array.from(p.projection.map((axis, j) => {
      const observed = axis.reduce((sum, v, k) => sum + v * power[k] / total, 0);
      const prediction = conditioned.reduce((sum, v, m) => sum + v * p.predictor[m][j], 0);
      return (observed - prediction) / p.scale[j];
    }));
    out.push(residual);
  } }).push(audio);
  return out;
}

export function createSpectralResidualScorer(p: SpectralResidualParameters) {
  const d = p.scale.length, vocab = p.tokens.length + 1;
  const finite = (xs: number[]) => xs.every(Number.isFinite);
  if (p.schema !== 1 || !d || p.projection.length !== d || p.projection.some(a => a.length !== 256 || !finite(a)) ||
    p.predictor.length !== 81 || p.predictor.some(a => a.length !== d || !finite(a)) ||
    p.means.length !== vocab || p.means.some(a => a.length !== d || !finite(a)) ||
    p.variance.length !== d || p.covered.length !== vocab ||
    p.scale.some(x => !Number.isFinite(x) || x <= 0) || p.variance.some(x => !Number.isFinite(x) || x <= 0))
    throw new Error("Invalid within-band spectral residual model");
  const ids = (s: string) => [...s].map(c => p.tokens.indexOf(c) + 1);
  return {
    parameters: p,
    compare(features: Float32Array[], r: SpectralResidualRequest, frameMs = 40) {
      const written = ids(r.written), accused = ids(r.accused);
      const from = r.fromFrame * frameMs / 10, to = r.toFrame * frameMs / 10;
      if (![from, to].every(Number.isInteger) || from < 0 || to > features.length || to <= from ||
        [...written, ...accused].some(id => !id || !p.covered[id]) || !p.covered[0]) return null;
      const rows = features.slice(from, to).map(x => {
        if (x.length !== d || !Array.from(x).every(Number.isFinite)) throw new Error("Invalid spectral observation");
        return p.means.map(mean => -.5 * mean.reduce((sum, v, j) =>
          sum + Math.log(2 * Math.PI * p.variance[j]) + (x[j] - v) ** 2 / p.variance[j], 0));
      });
      const writtenLogLikelihood = residualLikelihood(rows, written), accusedLogLikelihood = residualLikelihood(rows, accused);
      return Number.isFinite(writtenLogLikelihood) && Number.isFinite(accusedLogLikelihood)
        ? { writtenLogLikelihood, accusedLogLikelihood } : null;
    },
  };
}
