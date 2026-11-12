import { deletionCost, insertionCost, subCost, UNIT } from "../src/fuzzy.js";

const add = (a: number, b: number): number => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  return Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));
};

/** Training-only full-sequence edit smoother. Endpoints and latent word boundaries are summed, never Viterbi labels. */
export function smoothLocations(reference: string, heard: string): number[][] {
  if (!reference || !heard) throw new Error("Empty smoothing stream");
  const n = reference.length + 1, t = heard.length;
  const del = Array.from({ length: n }, (_, j) => j ? deletionCost(reference, j - 1) / UNIT : 0);
  let a = new Float64Array(n).fill(-Infinity);
  a[0] = 0;
  for (let j = 1; j < n; j++) a[j] = a[j - 1] - del[j];
  const forward: Float64Array[] = [a];
  for (const p of heard) {
    const b = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      b[j] = a[j] - insertionCost(reference, j, p) / UNIT;
      if (j) b[j] = add(b[j], a[j - 1] - subCost(reference[j - 1], p) / UNIT);
      if (j) b[j] = add(b[j], b[j - 1] - del[j]);
    }
    a = b;
    forward.push(a);
  }
  const logz = a[n - 1];
  let b = new Float64Array(n).fill(-Infinity);
  b[n - 1] = 0;
  const posterior: number[][] = new Array(t + 1);
  for (let i = t; i >= 0; i--) {
    posterior[i] = Array.from(b, (v, j) => Math.exp(v + forward[i][j] - logz));
    if (!i) break;
    const closure = new Float64Array(b);
    for (let j = n - 2; j >= 0; j--) closure[j] = add(closure[j], closure[j + 1] - del[j + 1]);
    const p = heard[i - 1], prev = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      prev[j] = closure[j] - insertionCost(reference, j, p) / UNIT;
      if (j + 1 < n) prev[j] = add(prev[j], closure[j + 1] - subCost(reference[j], p) / UNIT);
    }
    b = prev;
  }
  return posterior;
}

/** Distribution of the number of heard tokens before an internal reference boundary. */
export function boundaryDistribution(posterior: number[][], cut: number): number[] {
  const cdf = posterior.map(row => row.slice(0, cut + 1).reduce((a, b) => a + b, 0));
  const mass = cdf.map((p, i) => Math.max(0, p - (cdf[i + 1] ?? 0)));
  // Boundary traversed in the initial deletion closure: zero observations precede it.
  mass[0] += Math.max(0, 1 - cdf[0]);
  const sum = mass.reduce((a, b) => a + b, 0);
  return mass.map(p => p / sum);
}

export function sampleBoundary(mass: number[], quantile: number): number {
  let total = 0;
  for (let i = 0; i < mass.length; i++) { total += mass[i]; if (total >= quantile) return i; }
  return mass.length - 1;
}
