import { UNIT, deletionCost, insertionCost, subCost } from "./fuzzy.js";

/** Structural classes shared with word-rhythm-fit.py; no lexical or location identities. */
export function wordRhythmClasses(ref: string): Uint8Array {
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

export interface WordRhythmModel {
  mean: number[];
  sd: number[];
  /** [left short/long][right short/long][one/multiple consonants]. */
  rho: number[];
}

export interface WordRhythmStructure {
  classes: Uint8Array;
  word: Int32Array;
  nucleus: Int32Array;
  /** Nucleus index -> first reference phone. */
  starts: number[];
  /** Nucleus index -> incoming copula parameter index, or -1 at a word's first nucleus. */
  links: number[];
}

/** One contiguous vowel articulation is one nucleus, including pronunciation-expanded length. */
export function wordRhythmStructure(ref: string, words: readonly { start: number; end: number }[]): WordRhythmStructure {
  const classes = wordRhythmClasses(ref);
  const word = new Int32Array(ref.length).fill(-1);
  const nucleus = new Int32Array(ref.length).fill(-1);
  const starts: number[] = [], links: number[] = [];
  words.forEach((w, wi) => {
    word.fill(wi, w.start, w.end);
    let previous = -1, previousEnd = w.start;
    for (let i = w.start; i < w.end;) {
      if (!"َُِاۦۥ".includes(ref[i])) { i++; continue; }
      let end = i + 1;
      while (end < w.end && "َُِاۦۥ".includes(ref[end])) end++;
      const long = [...ref.slice(i, end)].some(p => "اۦۥ".includes(p));
      const id = starts.length;
      starts.push(i);
      const leftLong = previous >= 0 && classes[starts[previous]] === 2;
      links.push(previous < 0 ? -1 : Number(leftLong) * 4 + Number(long) * 2 + Number(i - previousEnd > 1));
      // The whole nucleus shares the length class; repeated duration symbols retain their original edit charges.
      classes.fill(long ? 2 : 1, i, end);
      nucleus.fill(id, i, end);
      previous = id; previousEnd = end; i = end;
    }
  });
  return { classes, word, nucleus, starts, links };
}

/** Normalized Gaussian copula density; both individual duration marginals are unchanged. */
export function wordRhythmCopula(z1: number, z2: number, rho: number): number {
  if (rho === 0) return 0;
  const v = 1 - rho * rho;
  return -.5 * Math.log(v) + (2 * rho * z1 * z2 - rho * rho * (z1*z1 + z2*z2)) / (2*v);
}

/** Integrating a missing nucleus in a Gaussian Markov chain multiplies its adjacent correlations. */
export function nucleusCorrelation(structure: WordRhythmStructure, model: WordRhythmModel, from: number, to: number): number {
  if (from < 0 || to <= from || structure.word[structure.starts[from]] !== structure.word[structure.starts[to]]) return 0;
  let rho = 1;
  for (let n = from + 1; n <= to; n++) rho *= model.rho[structure.links[n]];
  return rho;
}

export interface WordRhythmRow {
  lo: number;
  cost: Float64Array;
  /** Pre-update source of the unchanged local graph's movement arcs. */
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
interface State { nucleus: number; time: number; joint: number; independent: number }
type Cell = Map<string, State>;
const accumulate = (cell: Cell, n: number, t: number, joint: number, independent: number) => {
  if (joint === -Infinity && independent === -Infinity) return;
  const key = `${n}/${t}`, old = cell.get(key);
  cell.set(key, { nucleus: n, time: t, joint: add(old?.joint ?? -Infinity, joint), independent: add(old?.independent ?? -Infinity, independent) });
};

/**
 * Sum word-grouped and independent timing likelihoods on the same recorded local edit/movement graph. Histories
 * differ only in nucleus placement, so missing emissions and uncertain word divisions retain separate mass.
 * The score is a likelihood ratio; neither extra edit paths nor looser individual durations earn free evidence.
 */
export function wordRhythmRatios(
  ref: string, structure: WordRhythmStructure, rows: readonly WordRhythmRow[], model: WordRhythmModel,
  wordStarts: ReadonlySet<number>, entry: (at: number) => number,
  crossing: (from: number, best: number) => number, jumpCost: number, jumpDistance: number,
): Float64Array {
  const last = rows.at(-1);
  if (!last) return new Float64Array(0);
  const out = new Float64Array(last.cost.length);
  if (rows.length < 2 || rows.some(r => !Number.isFinite(r.duration) || r.duration <= 0)
    || model.rho.every(r => r === 0)) return out;
  const logD = rows.map(r => Math.log(r.duration));
  const rate = logD.reduce((a, b) => a + b, 0) / logD.length;
  const z = logD.map(x => model.mean.map((m, c) => (x - rate - m) / model.sd[c]));
  let lo = rows[0].lo;
  let prev: Cell[] = Array.from(rows[0].cost, c => new Map([["-1/-1", { nucleus: -1, time: -1, joint: -c/UNIT, independent: -c/UNIT }]]));
  for (let t = 0; t < rows.length; t++) {
    const row = rows[t], n = row.cost.length;
    const cells: Cell[] = Array.from({ length: n }, () => new Map());
    const emit = (dest: number, source: number, at: number, charge: number, movement = false) => {
      const src = prev[source - lo];
      if (!src) return;
      const c = at < 0 ? 0 : structure.classes[at];
      const ni = at < 0 ? -1 : structure.nucleus[at];
      const marginal = -.5 * z[t][c] ** 2 - Math.log(model.sd[c]);
      for (const state of src.values()) {
        const sameWord = state.nucleus >= 0 && !movement && (at < 0
          || structure.word[at] === structure.word[structure.starts[state.nucleus]]);
        const old = sameWord ? state.nucleus : -1;
        const rho = ni >= 0 ? nucleusCorrelation(structure, model, old, ni) : 0;
        const copula = rho === 0 ? 0 : wordRhythmCopula(z[state.time][structure.classes[structure.starts[old]]], z[t][c], rho);
        const nextN = ni < 0 ? old : ni;
        const nextT = ni < 0 ? (sameWord ? state.time : -1) : t;
        accumulate(cells[dest], nextN, nextT, state.joint - charge/UNIT + marginal + copula,
          state.independent - charge/UNIT + marginal);
      }
    };
    for (let i = 0; i < n; i++) {
      const at = row.lo + i;
      emit(i, at, -1, insertionCost(ref, at, row.p));
      if (i === 0) continue;
      const from = at - 1, s = subCost(ref[from], row.p) + entry(from);
      emit(i, from, from, s);
      if (from !== row.best && wordStarts.has(from)) emit(i, row.best, from,
        jumpCost + Math.abs(from - row.best)*jumpDistance + s + (entry(from) ? 0 : crossing(from, row.best)), true);
      const charge = (deletionCost(ref, from) + entry(from))/UNIT;
      for (const state of cells[i-1].values()) {
        const keep = state.nucleus >= 0 && structure.word[from] === structure.word[structure.starts[state.nucleus]];
        accumulate(cells[i], keep ? state.nucleus : -1, keep ? state.time : -1,
          state.joint - charge, state.independent - charge);
      }
    }
    let scale = -Infinity;
    for (const cell of cells) for (const s of cell.values()) scale = Math.max(scale, s.independent);
    if (!Number.isFinite(scale)) return out;
    for (const cell of cells) for (const s of cell.values()) { s.joint -= scale; s.independent -= scale; }
    prev = cells; lo = row.lo;
  }
  for (let i = 0; i < out.length; i++) {
    let joint = -Infinity, independent = -Infinity;
    for (const s of prev[i].values()) { joint = add(joint, s.joint); independent = add(independent, s.independent); }
    const lr = joint - independent;
    // Same evidence conversion used by tracker posterior scoring: three nats per plain edit.
    out[i] = Number.isFinite(lr) ? -lr * UNIT / 3 : 0;
  }
  return out;
}
