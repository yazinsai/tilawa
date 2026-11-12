import { UNIT, deletionCost, insertionCost, subCost } from "./fuzzy.js";

/** Structural classes shared with rhythm-fit.py; no lexical or location identities. */
export const RHYTHM_CLASSES = 8;
export function rhythmClasses(ref: string): Uint8Array {
  const out = new Uint8Array(ref.length);
  for (let i = 0; i < ref.length;) {
    let end = i + 1;
    while (ref[end] === ref[i]) end++;
    const p = ref[i];
    const c = "َُِ".includes(p) ? 1 : "اۦۥ".includes(p) ? 2
      : "ں۾".includes(p) || ("نم".includes(p) && end - i >= 3) ? 4
      : "نم".includes(p) ? 3 : "لريو".includes(p) ? 5
      : "فثذزسشصظهحخغعء".includes(p) ? 6 : 7;
    out.fill(c, i, end);
    i = end;
  }
  return out;
}

export interface RhythmModel {
  mean: number[];
  sd: number[];
  /** Gaussian Markov copula correlations, indexed previous/current structural class. */
  rho: number[][];
}

/** Density ratio of a bivariate normal to its SAME two standard-normal marginals. */
export function rhythmCopula(z1: number, z2: number, rho: number): number {
  if (rho === 0) return 0;
  const v = 1 - rho * rho;
  return -0.5 * Math.log(v) + (2 * rho * z1 * z2 - rho * rho * (z1 * z1 + z2 * z2)) / (2 * v);
}

export interface RhythmRow {
  lo: number;
  cost: Float64Array;
  /** Pre-update winning state, the unchanged source of the local graph's jump arcs. */
  best: number;
  p: string;
  duration: number;
}

const add = (a: number, b: number): number => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const hi = Math.max(a, b);
  return hi + Math.log1p(Math.exp(Math.min(a, b) - hi));
};

/**
 * Marginalize token-to-phone correspondence on the SAME recorded local graph twice. Movement and deletion charges
 * are shared. The independent partition supplies each sound's marginal, so only the adjacent copula survives the
 * ratio. Rows cover one common observed horizon, including insertions and speech before a run restart.
 */
export function rhythmRatios(
  ref: string, classes: Uint8Array, rows: readonly RhythmRow[], model: RhythmModel,
  wordStarts: ReadonlySet<number>, entry: (at: number) => number,
  crossing: (from: number, best: number) => number, jumpCost: number, jumpDistance: number,
): Float64Array {
  const last = rows.at(-1);
  if (!last) return new Float64Array(0);
  const out = new Float64Array(last.cost.length);
  if (rows.length < 2 || rows.some(r => !Number.isFinite(r.duration) || !(r.duration > 0))) return out;
  const logD = rows.map(r => Math.log(r.duration));
  const tempo = logD.reduce((a, b) => a + b, 0) / logD.length;
  const x = logD.map(d => d - tempo);
  const K = RHYTHM_CLASSES;
  // The first observation has no preceding timed token. Integrate its correspondence, with no invented pair.
  const initial = rows[0];
  let lo = initial.lo;
  let joint = new Float64Array(initial.cost.length * K).fill(-Infinity);
  let independent = new Float64Array(joint.length).fill(-Infinity);
  for (let i = 0; i < initial.cost.length; i++) joint[i * K] = independent[i * K] = -initial.cost[i] / UNIT;
  for (let t = 0; t < rows.length; t++) {
    const row = rows[t], n = row.cost.length;
    const j = new Float64Array(n * K).fill(-Infinity);
    const d = new Float64Array(j.length).fill(-Infinity);
    const z = model.mean.map((m, c) => (x[t] - m) / model.sd[c]);
    const prevZ = t ? model.mean.map((m, c) => (x[t - 1] - m) / model.sd[c]) : [];
    const emit = (dest: number, source: number, c: number, charge: number) => {
      if (source < 0 || source >= joint.length / K) return;
      const marginal = -0.5 * z[c] ** 2 - Math.log(model.sd[c]);
      for (let a = 0; a < K; a++) {
        const src = source * K + a;
        const copula = t ? rhythmCopula(prevZ[a], z[c], model.rho[a][c]) : 0;
        const dst = dest * K + c;
        j[dst] = add(j[dst], joint[src] - charge / UNIT + marginal + copula);
        d[dst] = add(d[dst], independent[src] - charge / UNIT + marginal);
      }
    };
    for (let i = 0; i < n; i++) {
      const at = row.lo + i;
      emit(i, at - lo, 0, insertionCost(ref, at, row.p));
      if (i > 0) {
        const from = at - 1, s = subCost(ref[from], row.p) + entry(from);
        emit(i, from - lo, classes[from], s);
        if (from !== row.best && wordStarts.has(from)) {
          emit(i, row.best - lo, classes[from], jumpCost + Math.abs(from - row.best) * jumpDistance
            + s + (entry(from) ? 0 : crossing(from, row.best)));
        }
        const charge = (deletionCost(ref, from) + entry(from)) / UNIT;
        for (let a = 0; a < K; a++) {
          j[i * K + a] = add(j[i * K + a], j[(i - 1) * K + a] - charge);
          d[i * K + a] = add(d[i * K + a], d[(i - 1) * K + a] - charge);
        }
      }
    }
    // Shared normalization changes neither partition ratio nor candidate ordering, and prevents underflow.
    const scale = Math.max(...d);
    for (let i = 0; i < j.length; i++) { j[i] -= scale; d[i] -= scale; }
    joint = j; independent = d; lo = row.lo;
  }
  for (let i = 0; i < out.length; i++) {
    let j = -Infinity, d = -Infinity;
    for (let a = 0; a < K; a++) { j = add(j, joint[i * K + a]); d = add(d, independent[i * K + a]); }
    out[i] = Number.isFinite(j - d) && j !== d ? -(j - d) * UNIT : 0;
  }
  return out;
}
