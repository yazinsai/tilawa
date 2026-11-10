import type { PosteriorFrames, SequenceEvidenceCandidate } from "@tilawa/core";

const add = (a: number, b: number): number => a === -Infinity ? b : b === -Infinity ? a :
  Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));
const sum = (xs: ArrayLike<number>) => Array.from(xs).reduce(add, -Infinity);
const normalized = (xs: Float64Array) => {
  const z = sum(xs);
  return Number.isFinite(z) ? xs.map(x => x - z) : null;
};

/** Ordinary CTC chain, optionally preceded by an already-active label. */
function chain(labels: number[], incoming?: number) {
  const states = [0, ...labels.flatMap(p => [p, 0])];
  if (incoming !== undefined) states.unshift(incoming);
  const offset = incoming === undefined ? 0 : 1;
  const parents = states.map((p, s) => {
    const ps = [s];
    if (s > 0) ps.push(s - 1);
    if (s > offset + 1 && (s - offset) % 2 === 1 && p !== states[s - 2]) ps.push(s - 2);
    // An active preceding label can enter a different first label without blank.
    if (incoming !== undefined && s === 2 && p !== incoming) ps.push(0);
    return ps;
  });
  return { states, parents };
}

function forward(frames: PosteriorFrames, labels: number[], from: number, to: number) {
  const { states, parents } = chain(labels);
  let prev = new Float64Array(states.length).fill(-Infinity); prev[0] = 0;
  for (let f = from; f < to; f++) {
    const row = frames.row(f);
    prev = Float64Array.from(states, (p, s) => parents[s].map(k => prev[k]).reduce(add, -Infinity) + row[p]);
  }
  return prev;
}

/**
 * A common boundary potential for every possible active class, plus blank. The
 * suffix is fixed, but a label already active at the boundary may continue until
 * the suffix begins. Equal labels still require blank before a new articulation.
 */
function outgoing(frames: PosteriorFrames, labels: number[], from: number, to: number, size: number) {
  if (!labels.length) return new Float64Array(size + 1); // no exterior observation
  const weights = new Float64Array(size + 1);
  for (let p = 0; p <= size; p++) {
    const { states, parents } = chain(labels, p || undefined);
    let beta = new Float64Array(states.length).fill(-Infinity);
    beta[states.length - 1] = beta[states.length - 2] = 0;
    for (let f = to - 1; f >= from; f--) {
      const row = frames.row(f), next = new Float64Array(states.length).fill(-Infinity);
      for (let s = 0; s < states.length; s++)
        for (const k of parents[s]) next[k] = add(next[k], row[states[s]] + beta[s]);
      beta = next;
    }
    weights[p] = beta[0];
  }
  return normalized(weights);
}

export interface ConditionedComparison {
  writtenNll: number;
  accusedNll: number;
  margin: number;
  incomingBlank: number;
  outgoingBlank: number;
  reverses: boolean;
}

/** Score both COMPLETE pronunciations against the SAME exterior-state potentials.
 * The existing interval is retained; no target phone can be emitted before it.
 * Exterior phones remain exactly the primary surrounding pronunciation.
 */
export function conditionedComparison(frames: PosteriorFrames, c: SequenceEvidenceCandidate,
  tokens: ReadonlyMap<string, number>): ConditionedComparison | null {
  const ids = (s: string) => [...s].map(p => tokens.get(p) ?? 0);
  const written = ids(c.written), accused = ids(c.accused), size = tokens.size;
  if (!written.length || !accused.length || [...written, ...accused].some(p => !p) ||
    c.fromFrame < frames.startFrame || c.toFrame > frames.endFrame || c.toFrame <= c.fromFrame) return null;
  const prefix = c.prefix, suffix = c.suffix;
  if (!prefix.length && !suffix.length) return null;
  if (prefix.some(p => p.start < frames.startFrame || p.end >= c.fromFrame) ||
    suffix.some(p => p.start < c.toFrame || p.end >= frames.endFrame)) return null;
  const incoming = new Float64Array(size + 1).fill(-Infinity);
  if (prefix.length) {
    const labels = ids(prefix.map(p => p.p).join(""));
    if (labels.some(p => !p)) return null;
    const dist = forward(frames, labels, prefix[0].start, c.fromFrame);
    incoming[0] = dist[dist.length - 1];
    incoming[labels.at(-1)!] = dist[dist.length - 2];
  } else incoming[0] = 0;
  const alpha = normalized(incoming);
  const right = ids(suffix.map(p => p.p).join(""));
  if (right.some(p => !p)) return null;
  const beta = outgoing(frames, right, c.toFrame, suffix.length ? suffix.at(-1)!.end + 1 : c.toFrame, size);
  if (!alpha || !beta) return null;
  const score = (labels: number[]) => {
    const { states, parents } = chain(labels);
    let prev = new Float64Array(states.length).fill(-Infinity); prev[0] = alpha[0];
    let active = Float64Array.from(alpha.slice(1));
    for (let f = c.fromFrame; f < c.toFrame; f++) {
      const row = frames.row(f);
      const next = Float64Array.from(states, (p, s) => {
        let a = parents[s].map(k => prev[k]).reduce(add, -Infinity);
        if (s === 0) a = add(a, sum(active));
        if (s === 1) for (let i = 0; i < active.length; i++) if (i + 1 !== p) a = add(a, active[i]);
        return a + row[p];
      });
      active = active.map((v, i) => v + row[i + 1]);
      prev = next;
    }
    return -add(prev.at(-1)! + beta[0], prev.at(-2)! + beta[labels.at(-1)!]);
  };
  const writtenNll = score(written), accusedNll = score(accused), margin = writtenNll - accusedNll;
  if (!Number.isFinite(margin)) return null;
  return { writtenNll, accusedNll, margin, incomingBlank: Math.exp(alpha[0]), outgoingBlank: Math.exp(beta[0]),
    reverses: c.margin > 0 && margin < 0 };
}
