import type { PosteriorFrames } from "./ctc.js";

const add = (a: number, b: number): number => a === -Infinity ? b : b === -Infinity ? a
  : Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));

type FrameRows = Pick<PosteriorFrames, "startFrame" | "endFrame" | "row">;
type Forms = readonly (readonly (readonly number[])[])[];
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
  return score(frames, lattice(before, candidate, after, insertionCost, repeats), from, to)?.marginal ?? null;
}

function lattice(before: Forms, candidate: readonly (readonly number[])[] | null, after: Forms,
  insertionCost: number, repeats: boolean, loop?: { phones: number[]; consonants: Set<number> }) {
  const groups = [...before, ...(candidate ? [candidate] : []), ...after];
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
  const loopStates = new Set<number>();
  const phoneLoop = (): Word => {
    // State tracks the number of CTC consonant onsets, capped at two. Staying
    // on a consonant holds it; a blank then the same consonant starts a new one.
    const blanks = [state(0), state(0), state(0)];
    const nodes = Array.from({ length: 3 }, (_, count) => new Map(
      loop!.phones.filter(p => count > 0 || !loop!.consonants.has(p)).map(p => [p, state(p)])));
    for (const b of blanks) loopStates.add(b);
    for (const list of nodes) for (const s of list.values()) loopStates.add(s);
    for (const p of loop!.phones) connect(boundary, [nodes[Number(loop!.consonants.has(p))].get(p)!], insertionCost, true);
    for (let count = 0; count <= 2; count++) {
      for (const [p, src] of nodes[count]) {
        arc(src, blanks[count]);
        for (const q of loop!.phones) if (p !== q)
          arc(src, nodes[Math.min(2, count + Number(loop!.consonants.has(q)))].get(q)!);
      }
      for (const p of loop!.phones)
        arc(blanks[count], nodes[Math.min(2, count + Number(loop!.consonants.has(p)))].get(p)!);
    }
    return { first: [], last: [...nodes[2].values()], blank: blanks[2] };
  };
  const extra = candidate ? word(candidate, boundary, true) : phoneLoop();
  // Direct no-word transitions avoid an epsilon skip and duplicate blank runs.
  const right = side(after, [...boundary, ...extra.last, extra.blank]);
  const finals = [...right.last, right.blank];
  for (let s = 0; s < labels.length; s++) arc(s, s);
  return { labels, arcs, finals, loopStates };
}

/** Identity-free insertion with at least two consonant onsets. The returned
 * Viterbi span is conditioned on using the inserted arc, not a fixed token window. */
export function seamPhoneLoop(frames: FrameRows, before: Forms, after: Forms,
  phones: readonly number[], consonants: readonly number[], from: number, to: number,
  insertionCost: number, repeats = true): { marginal: number; phones: number[]; fromFrame: number; toFrame: number } | null {
  if (!before.length || !after.length || !phones.length || !consonants.length ||
    !Number.isFinite(insertionCost) || !Number.isInteger(from) || !Number.isInteger(to) ||
    from < frames.startFrame || to > frames.endFrame || to <= from ||
    phones.some(p => !Number.isInteger(p) || p <= 0) || consonants.some(p => !phones.includes(p))) return null;
  const graph = lattice(before, null, after, insertionCost, repeats,
    { phones: [...new Set(phones)], consonants: new Set(consonants) });
  return score(frames, graph, from, to, true);
}

function score(frames: FrameRows, graph: ReturnType<typeof lattice>, from: number, to: number, viterbi = false) {
  if (!graph) return null;
  const { labels, arcs, finals, loopStates } = graph;
  const S = labels.length, T = to - from;
  const alpha: Float64Array[] = [new Float64Array(S).fill(-Infinity)];
  alpha[0][0] = 0;
  let best = new Float64Array(2*S).fill(-Infinity);
  best[0] = 0;
  const back: Int32Array[] = [];
  for (let t = 0; t < T; t++) {
    const prev = alpha[t], next = new Float64Array(S).fill(-Infinity), row = frames.row(from + t);
    if (labels.some(p => p >= row.length)) return null;
    const winners = viterbi ? new Int32Array(2*S).fill(-1) : null;
    const nextBest = viterbi ? new Float64Array(2*S).fill(-Infinity) : null;
    for (const a of arcs) {
      const emission = -a.cost + row[labels[a.dst]];
      next[a.dst] = add(next[a.dst], prev[a.src] + emission);
      if (nextBest && winners) for (let used=0; used<=1; used++) {
        const src=used*S+a.src, dst=(used || a.insertion ? S : 0)+a.dst;
        if (best[src]+emission > nextBest[dst]) { nextBest[dst]=best[src]+emission; winners[dst]=src; }
      }
    }
    alpha.push(next);
    if (winners && nextBest) { back.push(winners); best = nextBest; }
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
  const marginal = Math.max(0, Math.min(1, Math.exp(used - z)));
  const phones: number[] = [];
  let fromFrame = to, toFrame = from;
  if (viterbi) {
    // The posterior decision concerns all insertion paths. Extract the best
    // such path even if a single no-insertion timing wins the global argmax.
    let at = finals.map(s=>S+s).reduce((a, b) => best[a] >= best[b] ? a : b);
    if (!Number.isFinite(best[at])) return { marginal, phones, fromFrame, toFrame };
    const path = new Int32Array(T);
    for (let t = T - 1; t >= 0; t--) { path[t] = at%S; at = back[t][at]; }
    for (let t = 0; t < T; t++) if (loopStates.has(path[t]) && labels[path[t]]) {
      fromFrame = Math.min(fromFrame, from + t); toFrame = from + t + 1;
      if (!t || path[t] !== path[t - 1]) phones.push(labels[path[t]]);
    }
  }
  return { marginal, phones, fromFrame, toFrame };
}
