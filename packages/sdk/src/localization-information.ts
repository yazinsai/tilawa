/** Stored posterior rows, including blank at index zero. No acoustic re-decode. */
export interface LocalizationRows {
  readonly startFrame: number;
  readonly endFrame: number;
  row(frame: number): Float64Array;
}

export interface BoundaryDistribution {
  fromFrame: number;
  /** Row s contains log P(start=s, end=e+1), for e=s..T-1. */
  joint: Float64Array[];
  logPartition: number;
}

const add = (a: number, b: number): number => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  return Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));
};

/** Same CTC chain and deduplicated contiguous repeat arcs as PosteriorFrames.nll. */
function graph(labels: readonly number[], repeats: readonly (readonly [number, number])[]) {
  const phones = Array.from({ length: 2 * labels.length + 1 }, (_, s) => s % 2 ? labels[(s - 1) / 2] : 0);
  const incoming = phones.map((p, s) => new Set([s, ...(s ? [s - 1] : []),
    ...(s > 1 && s % 2 && p !== phones[s - 2] ? [s - 2] : [])]));
  for (const [a, b] of repeats) {
    if (!Number.isInteger(a) || !Number.isInteger(b) || a < 0 || b < a || b >= labels.length) continue;
    const target = 2 * a + 1;
    for (const source of [2 * b + 1, 2 * b + 2]) if (source !== target) incoming[target].add(source);
  }
  const before = incoming.map(xs => [...xs]);
  const after: number[][] = phones.map(() => []);
  before.forEach((xs, s) => xs.forEach(p => after[p].push(s)));
  return { phones, before, after };
}

/**
 * Joint boundaries of the final contiguous visit to one mandatory word in a whole-ayah forced read.
 * The start is its first phone's first frame; the exclusive end follows its last phone's last frame.
 * Blanks between phones belong to the visit; facing-edge blanks do not. Internal repeat arcs remain
 * available. An earlier visit, including an ayah reread, is summed into the prefix. The suffix cannot
 * re-enter the word, so every complete forced path contributes exactly once, even with repeats.
 *
 * The control neutralizes non-blank identities ONLY in this word's states: each gets the mean
 * non-blank probability of that row. Blank probabilities, all other words and graph arcs stay fixed.
 */
export function wordBoundaryDistribution(rows: LocalizationRows, labels: readonly number[],
  word: readonly [number, number], from: number, to: number,
  repeats: readonly (readonly [number, number])[] = [], neutral = false): BoundaryDistribution | null {
  const [a, b] = word;
  if (!Number.isInteger(from) || !Number.isInteger(to) || from < rows.startFrame || to > rows.endFrame || to <= from ||
    !Number.isInteger(a) || !Number.isInteger(b) || a < 0 || b < a || b >= labels.length) return null;
  // A word/ayah repeat may contain the word or be wholly outside it. An arc cutting through
  // a word would permit partial visits, which do not have the two complete-word boundaries.
  if (repeats.some(([x, y]) => Number.isInteger(x) && Number.isInteger(y) && x >= 0 && y >= x && y < labels.length &&
    ((x > a && x <= b) || (y >= a && y < b)))) return null;
  const { phones, before, after } = graph(labels, repeats);
  const T = to - from, S = phones.length, first = 2 * a + 1, last = 2 * b + 1, W = last - first + 1;
  const inside = (s: number) => s >= first && s <= last;
  const emissions = Array.from({ length: T }, (_, t) => {
    const r = rows.row(from + t);
    if (labels.some(id => !Number.isInteger(id) || id <= 0 || id >= r.length)) throw new RangeError("Phone outside posterior inventory");
    let mean = -Infinity;
    if (neutral) {
      for (let id = 1; id < r.length; id++) mean = add(mean, r[id]);
      mean -= Math.log(r.length - 1);
    }
    return Float64Array.from(phones, (id, s) => neutral && inside(s) && id ? mean : r[id]);
  });
  const entry = new Float64Array(T).fill(-Infinity);
  let alpha = new Float64Array(S).fill(-Infinity);
  // Virtual initial state before frame zero: only blank zero and phone one can start.
  for (let t = 0; t < T; t++) {
    let start = t === 0 && first === 1 ? 0 : -Infinity;
    if (t) for (const p of before[first]) if (!inside(p)) start = add(start, alpha[p]);
    entry[t] = start;
    const next = new Float64Array(S).fill(-Infinity);
    for (let s = 0; s < S; s++) {
      let z = t === 0 && s <= 1 ? 0 : -Infinity;
      if (t) for (const p of before[s]) z = add(z, alpha[p]);
      next[s] = z + emissions[t][s];
    }
    alpha = next;
  }
  const logPartition = add(alpha[S - 1], alpha[S - 2]);
  if (!Number.isFinite(logPartition)) return null;
  const exit = new Float64Array(T).fill(-Infinity);
  let beta = new Float64Array(S).fill(-Infinity);
  for (let t = T - 1; t >= 0; t--) {
    let end = t === T - 1 && last === S - 2 ? 0 : -Infinity;
    if (t < T - 1) for (const s of after[last]) if (!inside(s)) end = add(end, emissions[t + 1][s] + beta[s]);
    exit[t] = end;
    const next = new Float64Array(S).fill(-Infinity);
    for (let s = 0; s < S; s++) if (!inside(s)) {
      let z = t === T - 1 && s >= S - 2 ? 0 : -Infinity;
      if (t < T - 1) for (const q of after[s]) if (!inside(q)) z = add(z, emissions[t + 1][q] + beta[q]);
      next[s] = z;
    }
    beta = next;
  }
  const local = before.slice(first, last + 1).map(xs => xs.filter(inside).map(p => p - first));
  const joint: Float64Array[] = [];
  for (let start = 0; start < T; start++) {
    const scores = new Float64Array(T - start).fill(-Infinity);
    if (entry[start] === -Infinity) { joint.push(scores); continue; }
    let prev = new Float64Array(W).fill(-Infinity);
    let next = new Float64Array(W);
    prev[0] = entry[start] + emissions[start][first];
    for (let end = start; end < T; end++) {
      if (end > start) {
        for (let s = 0; s < W; s++) {
          let z = -Infinity;
          for (const p of local[s]) z = add(z, prev[p]);
          next[s] = z + emissions[end][first + s];
        }
        [prev, next] = [next, prev];
      }
      scores[end - start] = prev[W - 1] + exit[end] - logPartition;
    }
    joint.push(scores);
  }
  return { fromFrame: from, joint, logPartition };
}

/** KL(actual joint boundaries || identity-neutral control), in nats. */
export function localizationInformation(rows: LocalizationRows, labels: readonly number[],
  word: readonly [number, number], from: number, to: number,
  repeats: readonly (readonly [number, number])[] = []): number | null {
  const actual = wordBoundaryDistribution(rows, labels, word, from, to, repeats);
  const control = wordBoundaryDistribution(rows, labels, word, from, to, repeats, true);
  if (!actual || !control) return null;
  let kl = 0;
  for (let s = 0; s < actual.joint.length; s++) for (let e = 0; e < actual.joint[s].length; e++) {
    const p = actual.joint[s][e], q = control.joint[s][e];
    if (p !== -Infinity) kl += Math.exp(p) * (p - q);
  }
  return Math.max(0, kl);
}
