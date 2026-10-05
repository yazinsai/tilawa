import type { AlignmentOp } from "./align.js";

/** Integer cost units; one plain edit costs `UNIT`. */
export const UNIT = 10;
const MADD = 2;
const CONFUSABLE = 5;
const SHORT_VOWEL = 5;

const MADD_LETTERS = new Set(["ا", "ۦ", "ۥ"]);
const SHORT_VOWELS = new Set(["َ", "ُ", "ِ"]);
const CONFUSABLE_PAIRS = ["هح", "عء", "صس", "ضد", "طت", "قك", "ذز", "ثس"];
const confusable = new Set(CONFUSABLE_PAIRS.flatMap(([a, b]) => [a + b, b + a]));

export function subCost(ref: string, obs: string): number {
  if (ref === obs) return 0;
  if (confusable.has(ref + obs)) return CONFUSABLE;
  if (SHORT_VOWELS.has(ref) && SHORT_VOWELS.has(obs)) return SHORT_VOWEL;
  return UNIT;
}

/** Inserting or dropping `p` next to the same madd letter only changes the elongation length. */
function indelCost(p: string, neighbours: string[]): number {
  if (MADD_LETTERS.has(p) && neighbours.includes(p)) return MADD;
  if (SHORT_VOWELS.has(p)) return SHORT_VOWEL;
  return UNIT;
}

export interface SpanAlignment {
  /** Half-open phoneme range in the searched reference. */
  start: number;
  end: number;
  /** Weighted cost in `UNIT`s, for ranking only. */
  cost: number;
  /** Unweighted phoneme edits along `path` divided by the matched reference length. */
  per: number;
  /** `refAt` is relative to `start`. */
  path: AlignmentOp[];
}

const INF = 0x3fffffff;
/** Starting inside a word: the clip was cut mid-word. */
const MID_WORD_START = 15;

export interface AlignOptions {
  /** Per-phoneme decoder confidence. Leading observed phonemes can be dropped at `UNIT * conf` each. */
  conf?: ArrayLike<number>;
  /** Only allow starts where this holds. Other starts cost `MID_WORD_START`. */
  strictStart?: boolean;
}

/**
 * Semi-global weighted alignment of all of `obs` against a substring of `ref[lo, hi)`.
 * Word starts (`isStart`) are free; its end is free so an in-progress word can match.
 */
export function alignSubstring(
  ref: string,
  lo: number,
  hi: number,
  obs: string,
  isStart: (offset: number) => boolean,
  options: AlignOptions = {},
): SpanAlignment | null {
  const n = hi - lo;
  const m = obs.length;
  const w = m + 1;
  const d = new Int32Array((n + 1) * w);
  const at = (i: number, j: number) => i * w + j;
  const r = (i: number) => ref[lo + i - 1];
  const insCost = (i: number, j: number) => indelCost(obs[j - 1], [i > 0 ? r(i) : "", i < n ? ref[lo + i] : "", j > 1 ? obs[j - 2] : ""]);
  const delCost = (i: number, j: number) => indelCost(r(i), [i > 1 ? r(i - 1) : "", i < n ? ref[lo + i] : "", j > 0 ? obs[j - 1] : ""]);
  const startCost = (i: number) => (isStart(lo + i) ? 0 : options.strictStart ? INF : MID_WORD_START);
  const skip = new Int32Array(m + 1).fill(INF);
  skip[0] = 0;
  if (options.conf) for (let j = 1; j <= m; j++) skip[j] = skip[j - 1] + Math.round(UNIT * options.conf[j - 1]);

  for (let i = 0; i <= n; i++) {
    const start = startCost(i);
    d[at(i, 0)] = start;
    for (let j = 1; j <= m; j++) {
      let best = Math.min(d[at(i, j - 1)] + insCost(i, j), start + skip[j]);
      if (i > 0) {
        best = Math.min(best, d[at(i - 1, j - 1)] + subCost(r(i), obs[j - 1]), d[at(i - 1, j)] + delCost(i, j));
      }
      d[at(i, j)] = Math.min(best, INF);
    }
  }

  let endI = -1;
  for (let i = 0; i <= n; i++) if (endI === -1 || d[at(i, m)] < d[at(endI, m)]) endI = i;
  if (endI === -1 || d[at(endI, m)] >= INF) return null;

  const ops: AlignmentOp[] = [];
  let i = endI;
  let j = m;
  while (j > 0) {
    const here = d[at(i, j)];
    if (here === startCost(i) + skip[j]) {
      // Dropped leading phonemes stay in the path as insertions so the PER still counts them.
      for (let k = j; k > 0; k--) ops.push({ op: "ins", obs: obs[k - 1], obsAt: k - 1 });
      break;
    }
    if (i > 0 && j > 0 && here === d[at(i - 1, j - 1)] + subCost(r(i), obs[j - 1])) {
      ops.push(r(i) === obs[j - 1] ? { op: "match", ref: r(i), refAt: i - 1, obsAt: j - 1 } : { op: "sub", ref: r(i), obs: obs[j - 1], refAt: i - 1, obsAt: j - 1 });
      i--;
      j--;
    } else if (i > 0 && here === d[at(i - 1, j)] + delCost(i, j)) {
      ops.push({ op: "del", ref: r(i), refAt: i - 1 });
      i--;
    } else {
      ops.push({ op: "ins", obs: obs[j - 1], obsAt: j - 1 });
      j--;
    }
  }
  ops.reverse();
  for (const o of ops) if (o.op !== "ins") o.refAt -= i;

  const length = endI - i;
  const edits = ops.filter((o) => o.op !== "match").length;
  return { start: lo + i, end: lo + endI, cost: d[at(endI, m)], per: length ? edits / length : edits ? Infinity : 0, path: ops };
}
