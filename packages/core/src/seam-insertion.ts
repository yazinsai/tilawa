import type { PosteriorFrames } from "./ctc.js";

const add = (a: number, b: number): number => a === -Infinity ? b : b === -Infinity ? a
  : Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));

type Arc = { src: number; dst: number; cost: number; insertion: boolean };
type Word = { first: number[]; last: number[]; blank: number };

/**
 * Closed seam CTC lattice with an optional exact word between the two text sides.
 * Each word may have attested pronunciation variants. Word and contiguous side
 * repeats use today's repeat transitions. No repeat crosses the optional word,
 * so its entrance is traversed at most once. Its marginal is the forward-backward
 * mass through that entrance, summed over all frame timings, divided by total mass.
 * Blank runs are shared at boundaries, including the no-insertion boundary; there
 * is no epsilon path that counts the same blank timing twice. No phone-loop filler.
 */
export function seamInsertionMarginal(
  frames: Pick<PosteriorFrames, "startFrame" | "endFrame" | "row">,
  before: readonly (readonly (readonly number[])[])[],
  candidate: readonly (readonly number[])[],
  after: readonly (readonly (readonly number[])[])[],
  from: number,
  to: number,
  insertionCost: number,
  repeats = true,
): number | null {
  if (!before.length || !after.length || !candidate.length || !Number.isFinite(insertionCost) ||
    from < frames.startFrame || to > frames.endFrame || to <= from) return null;
  const groups = [...before, candidate, ...after];
  if (groups.some(g => !g.length || g.some(f => !f.length || f.some(p => !Number.isInteger(p) || p <= 0)))) return null;
  const labels = [0];
  const arcs: Arc[] = [];
  const seen = new Set<string>();
  const state = (label: number) => { labels.push(label); return labels.length - 1; };
  const arc = (src: number, dst: number, cost = 0, insertion = false) => {
    const key = `${src}:${dst}:${cost}:${insertion}`;
    if (!seen.has(key)) { seen.add(key); arcs.push({ src, dst, cost, insertion }); }
  };
  const connect = (sources: number[], first: number[], cost = 0, insertion = false, repeat = false) => {
    for (const src of sources) for (const dst of first) {
      // Ordinary CTC needs a blank between equal labels. Existing repeat arcs
      // permit re-entering a different state even when its label is equal.
      if (src !== dst && (repeat || labels[src] === 0 || labels[src] !== labels[dst])) arc(src, dst, cost, insertion);
    }
  };
  const word = (forms: readonly (readonly number[])[], sources: number[], inserted = false): Word => {
    const blank = state(0), first: number[] = [], last: number[] = [];
    const unique = new Map(forms.map(f => [JSON.stringify(f), f]));
    for (const form of unique.values()) {
      let prev = -1;
      for (let i = 0; i < form.length; i++) {
        const current = state(form[i]);
        if (!i) first.push(current);
        else {
          const between = state(0);
          arc(prev, between); arc(between, current);
          if (labels[prev] !== labels[current]) arc(prev, current);
        }
        prev = current;
      }
      last.push(prev); arc(prev, blank);
    }
    connect(sources, first, inserted ? insertionCost : 0, inserted);
    if (repeats) connect([...last, blank], first, 0, false, true);
    return { first, last, blank };
  };
  const side = (forms: typeof before, sources: number[]): Word => {
    let first: number[] = [], tail: Word | undefined;
    for (const variants of forms) {
      tail = word(variants, sources);
      if (!first.length) first = tail.first;
      sources = [...tail.last, tail.blank];
    }
    if (repeats) connect(sources, first, 0, false, true);
    return tail!;
  };
  const left = side(before, [0]);
  const boundary = [...left.last, left.blank];
  const extra = word(candidate, boundary, true);
  // Direct no-word transitions avoid an epsilon skip and duplicate blank runs.
  const right = side(after, [...boundary, ...extra.last, extra.blank]);
  const finals = [...right.last, right.blank];
  for (let s = 0; s < labels.length; s++) arc(s, s);
  const S = labels.length, T = to - from;
  const alpha: Float64Array[] = [new Float64Array(S).fill(-Infinity)];
  alpha[0][0] = 0;
  for (let t = 0; t < T; t++) {
    const prev = alpha[t], next = new Float64Array(S).fill(-Infinity), row = frames.row(from + t);
    if (labels.some(p => p >= row.length)) return null;
    for (const a of arcs) next[a.dst] = add(next[a.dst], prev[a.src] - a.cost + row[labels[a.dst]]);
    alpha.push(next);
  }
  let z = -Infinity;
  for (const s of finals) z = add(z, alpha[T][s]);
  if (!Number.isFinite(z)) return null;
  let beta = new Float64Array(S).fill(-Infinity), used = -Infinity;
  for (const s of finals) beta[s] = 0;
  for (let t = T - 1; t >= 0; t--) {
    const prev = new Float64Array(S).fill(-Infinity), row = frames.row(from + t);
    for (const a of arcs) {
      const b = -a.cost + row[labels[a.dst]] + beta[a.dst];
      prev[a.src] = add(prev[a.src], b);
      if (a.insertion) used = add(used, alpha[t][a.src] + b);
    }
    beta = prev;
  }
  return Math.max(0, Math.min(1, Math.exp(used - z)));
}
