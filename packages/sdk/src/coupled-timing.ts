import { PosteriorFrames } from "@tilawa/core";

type Match = readonly [number, number];
type Interval = readonly [number, number];

export interface CoupledTimingResult {
  /** Positive means the complete written pronunciation wins. */
  independent: number;
  normalizedIndependent: number;
  /** Same joint-posterior statistic with the boundary equality constraints removed. */
  uncoupled: number;
  coupled: number;
  correspondences: number;
  sharedBoundariesDiffer: boolean;
  writtenBoundaries: Interval[];
  rivalBoundaries: Interval[];
}

/**
 * Joint timing posterior on a paired CTC graph. All maximum monotone exact-phone
 * correspondences participate with equal prior weight. A matched phone's entry,
 * occupancy and exit must coincide on the two tracks; other phones keep ordinary
 * CTC transitions, including blank separation of repeated articulations.
 *
 * Both complete tracks have a uniform normalized prior over their legal timings.
 * For each correspondence the paired timing prior is also uniform and normalized.
 * Joint acoustic weight is the geometric mean of the two path likelihoods: shared
 * emissions are counted once, and differing emissions symmetrically. The coupled
 * preference is the posterior expectation of the complete-path log likelihood
 * ratio (with the same individual timing normalization on both tracks). This is
 * the derivative of the paired log partition at equal hypothesis weight. Thus
 * neither track can choose a different timing posterior for its shared sounds.
 * It is a recheck statistic, not a replacement for the checker's CTC margin.
 *
 * Null is abstention: missing rows, impossible complete paths, no shared phones,
 * or an excessive correspondence graph. The resource guard never truncates the
 * correspondence marginal or removes phones from either pronunciation.
 */
export function coupledTiming(frames: PosteriorFrames, written: readonly number[], rival: readonly number[],
  from: number, to: number): CoupledTimingResult | null {
  if (!written.length || !rival.length || from < frames.startFrame || to > frames.endFrame || to <= from ||
    written.some(x => !Number.isInteger(x) || x <= 0) || rival.some(x => !Number.isInteger(x) || x <= 0)) return null;
  const matches = correspondences(written, rival);
  if (!matches?.length || !matches[0].length) return null;
  const w = chain(frames, written, from, to), r = chain(frames, rival, from, to);
  if (!w || !r) return null;
  const normalization = r.count - w.count;
  let total = -Infinity, delta = 0;
  for (const match of matches) {
    const paired = pair(frames, written, rival, match, from, to);
    if (!paired) return null;
    // Each correspondence has a normalized timing prior, then a uniform prior
    // across correspondences. The common 1/N cancels in this expectation.
    [total, delta] = mix(total, delta, paired.mass - paired.count, paired.delta + normalization);
  }
  return { independent: r.nll - w.nll, normalizedIndependent: r.nll - w.nll + normalization,
    uncoupled: w.expected - r.expected + normalization,
    coupled: delta, correspondences: matches.length,
    sharedBoundariesDiffer: matches.some(m => m.some(([i,j]) =>
      w.bounds[i][0] !== r.bounds[j][0] || w.bounds[i][1] !== r.bounds[j][1])),
    writtenBoundaries: w.bounds, rivalBoundaries: r.bounds };
}

/** Every maximum exact-phone correspondence, deduplicated without edit-order multiplicity. */
function correspondences(w: readonly number[], r: readonly number[]): Match[][] | null {
  const width = r.length + 1, dp = new Int32Array((w.length + 1) * width);
  for (let i = w.length - 1; i >= 0; i--) for (let j = r.length - 1; j >= 0; j--)
    dp[i*width+j] = Math.max(dp[(i+1)*width+j], dp[i*width+j+1],
      w[i] === r[j] ? 1 + dp[(i+1)*width+j+1] : 0);
  if (!dp[0]) return [];
  const out: Match[][] = [], path: Match[] = [];
  let excessive = false;
  const visit = (i: number, j: number, need: number) => {
    if (excessive) return;
    if (!need) {
      out.push(path.slice());
      if (out.length > 1024) excessive = true;
      return;
    }
    for (let a = i; a < w.length; a++) for (let b = j; b < r.length; b++) {
      if (w[a] !== r[b] || dp[a*width+b] !== need || dp[(a+1)*width+b+1] !== need - 1) continue;
      path.push([a,b]); visit(a+1,b+1,need-1); path.pop();
      if (excessive) return;
    }
  };
  visit(0,0,dp[0]);
  return excessive ? null : out;
}

function label(xs: readonly number[], s: number) { return s % 2 ? xs[(s-1)/2] : 0; }
function predecessors(xs: readonly number[], s: number): number[] {
  const out = [s];
  if (s) out.push(s-1);
  if (s > 1 && s % 2 && label(xs,s) !== label(xs,s-2)) out.push(s-2);
  return out;
}
function logAdd(a: number, b: number) {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  return Math.max(a,b) + Math.log1p(Math.exp(-Math.abs(a-b)));
}
/** Log partition and its first derivative; stable even for long low-probability paths. */
function mix(a: number, da: number, b: number, db: number): [number,number] {
  if (a === -Infinity) return [b,db];
  if (b === -Infinity) return [a,da];
  const z = logAdd(a,b);
  return [z, Math.exp(a-z)*da + Math.exp(b-z)*db];
}

function chain(frames: PosteriorFrames, xs: readonly number[], from: number, to: number) {
  const S = 2*xs.length+1, T = to-from, incoming = Array.from({length:S},(_,s) => predecessors(xs,s));
  let count = new Float64Array(S).fill(-Infinity), score = count.slice();
  let mass = count.slice(), expected = new Float64Array(S);
  count[0] = count[1] = 0;
  score[0] = frames.row(from)[0]; score[1] = frames.row(from)[xs[0]];
  for (const s of [0,1]) { mass[s] = score[s]/2; expected[s] = score[s]; }
  const trace = new Int32Array(T*S).fill(-1);
  for (let t = 1; t < T; t++) {
    const nextCount = new Float64Array(S).fill(-Infinity), nextScore = nextCount.slice(), row = frames.row(from+t);
    const nextMass = nextCount.slice(), nextExpected = new Float64Array(S);
    for (let s = 0; s < S; s++) {
      for (const p of incoming[s]) {
        nextCount[s] = logAdd(nextCount[s],count[p]);
        [nextMass[s],nextExpected[s]] = mix(nextMass[s],nextExpected[s],mass[p],expected[p]);
        if (score[p] > nextScore[s]) { nextScore[s] = score[p]; trace[t*S+s] = p; }
      }
      nextScore[s] += row[label(xs,s)];
      nextMass[s] += row[label(xs,s)]/2;
      nextExpected[s] += row[label(xs,s)];
    }
    count = nextCount; score = nextScore;
    mass = nextMass; expected = nextExpected;
  }
  const end = score[S-1] > score[S-2] ? S-1 : S-2;
  if (!Number.isFinite(score[end])) return null;
  const bounds: [number,number][] = Array.from({length:xs.length},() => [to,from]);
  let s = end;
  for (let t = T-1; t >= 0; t--) {
    if (s % 2) { const b = bounds[(s-1)/2]; b[0] = from+t; b[1] = Math.max(b[1],from+t+1); }
    if (t) s = trace[t*S+s];
  }
  const [,mean] = mix(mass[S-1],expected[S-1],mass[S-2],expected[S-2]);
  return { bounds, expected:mean, count: logAdd(count[S-1],count[S-2]), nll: frames.nll(xs,from,to) };
}

function pair(frames: PosteriorFrames, w: readonly number[], r: readonly number[], match: Match[], from: number, to: number) {
  const W = 2*w.length+1, R = 2*r.length+1;
  // Phase 2k: both tracks between shared phones; phase 2k+1: both in shared phone k.
  const phase = (size: number, which: 0 | 1) => {
    const out = new Int32Array(size);
    let k = 0;
    for (let s = 0; s < size; s++) {
      while (k < match.length && 2*match[k][which]+1 < s) k++;
      out[s] = 2*k + (k < match.length && 2*match[k][which]+1 === s ? 1 : 0);
    }
    return out;
  };
  const wp = phase(W,0), rp = phase(R,1), ids = new Int32Array(W*R).fill(-1);
  const states: [number,number][] = [];
  for (let a = 0; a < W; a++) for (let b = 0; b < R; b++) if (wp[a] === rp[b]) {
    ids[a*R+b] = states.length; states.push([a,b]);
  }
  const incoming = states.map(([a,b]) => predecessors(w,a).flatMap(x =>
    predecessors(r,b).map(y => ids[x*R+y]).filter(i => i >= 0)));
  const S = states.length;
  let mass = new Float64Array(S).fill(-Infinity), count = mass.slice(), delta = new Float64Array(S);
  const initial = frames.row(from);
  for (let s = 0; s < S; s++) {
    const [a,b] = states[s];
    if (a > 1 || b > 1) continue;
    const x = initial[label(w,a)], y = initial[label(r,b)];
    mass[s] = (x+y)/2; delta[s] = x-y; count[s] = 0;
  }
  for (let f = from+1; f < to; f++) {
    const nextMass = new Float64Array(S).fill(-Infinity), nextCount = nextMass.slice(), nextDelta = new Float64Array(S);
    const row = frames.row(f);
    for (let s = 0; s < S; s++) {
      let z = -Infinity, d = 0, n = -Infinity;
      for (const p of incoming[s]) { [z,d] = mix(z,d,mass[p],delta[p]); n = logAdd(n,count[p]); }
      const [a,b] = states[s], x = row[label(w,a)], y = row[label(r,b)];
      nextMass[s] = z + (x+y)/2; nextDelta[s] = d + x-y; nextCount[s] = n;
    }
    mass = nextMass; delta = nextDelta; count = nextCount;
  }
  let z = -Infinity, d = 0, n = -Infinity;
  for (let s = 0; s < S; s++) {
    const [a,b] = states[s];
    if (a < W-2 || b < R-2) continue;
    [z,d] = mix(z,d,mass[s],delta[s]); n = logAdd(n,count[s]);
  }
  return Number.isFinite(z) && Number.isFinite(n) ? { mass:z, delta:d, count:n } : null;
}
