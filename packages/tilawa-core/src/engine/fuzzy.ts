import type { AlignmentOp } from "./align.js";

/** Integer cost units; one plain edit costs `UNIT`. */
export const UNIT = 10;
const LENGTH_CHANGE = 2;
const CONFUSABLE = 5;
const SHORT_VOWEL = 5;

const MADD_LETTERS = new Set(["ا", "ۦ", "ۥ"]);
const SHORT_VOWELS = new Set(["َ", "ُ", "ِ"]);
const CONFUSABLE_PAIRS = ["هح", "عء", "صس", "ضد", "طت", "قك", "ذز", "ثس"];
const confusable = new Set(CONFUSABLE_PAIRS.flatMap(([a, b]) => [a + b, b + a]));
/** The plain consonants: no vowel, madd, ghunna or qalqalah mark. */
export const CONSONANTS: ReadonlySet<string> = new Set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي");

/** Consonant `obs` heard for written `ref` is the engine's near-equal: one of `CONFUSABLE_PAIRS`. */
export const isConfusable = (ref: string, obs: string) => confusable.has(ref + obs);

export function subCost(ref: string, obs: string): number {
  if (ref === obs) return 0;
  if (confusable.has(ref + obs)) return CONFUSABLE;
  if (SHORT_VOWELS.has(ref) && SHORT_VOWELS.has(obs)) return SHORT_VOWEL;
  return UNIT;
}

/** Ghunna is written as a run: `ں` for ikhfaa, `۾` for ikhfaa shafawi, `نننن`/`مممم` for a mushaddad noon or meem. */
const GHUNNA_RUNS = new Set(["ں", "۾"]);
const GHUNNA_DOUBLED = new Set(["ن", "م"]);

/**
 * Whether a run of `p` that is `run` long before an edit and `run ± 1` after it is still the same sound, held
 * longer or shorter. A doubled noon or meem shorter than 3 is a shadda, so shortening it is a real error.
 */
function isLengthening(p: string, run: number): boolean {
  if (MADD_LETTERS.has(p) || GHUNNA_RUNS.has(p)) return run >= 2;
  if (GHUNNA_DOUBLED.has(p)) return run >= 3;
  return false;
}

/** Sounds whose held length varies between reciters; runs of them collapse before posterior scoring. */
const HELD = new Set([...MADD_LETTERS, ...GHUNNA_RUNS]);
export const isHeld = (p: string) => HELD.has(p);

/**
 * Collapse held-sound runs so CTC scoring doesn't charge length: a madd or ghunna run becomes one token, and a noon or
 * meem run of 3 or more (ghunna on a shadda) becomes two. Length stays the decoded alignment's job.
 */
export function collapseHeld(tokens: string[]): string[] {
  const out: string[] = [];
  for (let i = 0; i < tokens.length; ) {
    let j = i;
    while (j < tokens.length && tokens[j] === tokens[i]) j++;
    const run = j - i;
    const keep = HELD.has(tokens[i]) ? 1 : GHUNNA_DOUBLED.has(tokens[i]) && run >= 3 ? 2 : run;
    for (let k = 0; k < keep; k++) out.push(tokens[i]);
    i = j;
  }
  return out;
}

function indelCost(p: string, lengthening: boolean): number {
  if (lengthening) return LENGTH_CHANGE;
  if (SHORT_VOWELS.has(p)) return SHORT_VOWEL;
  return UNIT;
}

/** Length of the run of `s[k]` that contains `k`. */
function runAt(s: string, k: number): number {
  let a = k;
  let b = k;
  while (a > 0 && s[a - 1] === s[k]) a--;
  while (b + 1 < s.length && s[b + 1] === s[k]) b++;
  return b - a + 1;
}

/**
 * Inserting `p` before `ref[at]`: lengthening when it sits beside a reference run of `p`. Only `ref` before `end` counts
 * as a neighbour.
 */
export function insertionCost(ref: string, at: number, p: string, end = ref.length): number {
  const before = at > 0 && ref[at - 1] === p ? runAt(ref, at - 1) : 0;
  const after = at < end && ref[at] === p ? runAt(ref, at) : 0;
  return indelCost(p, isLengthening(p, Math.max(before, after) + 1));
}

/** Dropping `ref[at]`: shortening when it's part of a held run. */
export function deletionCost(ref: string, at: number): number {
  return indelCost(ref[at], isLengthening(ref[at], runAt(ref, at)));
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

const INF = Infinity;

const codes = new Map<string, number>();
const tokenCode = (p: string) => {
  let c = codes.get(p);
  if (c === undefined) codes.set(p, (c = codes.size));
  return c;
};
/** `subCost(ref, obs) + EDIT` by observed token code, or 0 on a match; rows grow as new tokens appear. */
const subRows: { row: Float64Array; filled: number }[] = [];
function subTable(refCode: number): Float64Array {
  const have = subRows[refCode];
  if (have && have.filled === codes.size) return have.row;
  const ref = [...codes.keys()][refCode];
  const row = new Float64Array(codes.size);
  for (const [obs, c] of codes) row[c] = ref === obs ? 0 : subCost(ref, obs) + EDIT;
  subRows[refCode] = { row, filled: codes.size };
  return row;
}
/**
 * Each plain edit adds this to the DP key, so equal weighted costs resolve to the path with fewer edits and the PER
 * isn't inflated by an arbitrary choice. A power of two keeps the sums exact; it stays below 1 for up to 2^20 edits.
 */
const EDIT = 2 ** -20;
/** Starting inside a word: the clip was cut mid-word. */
export const MID_WORD_START = 10;

export interface AlignOptions {
  /** Per-phoneme decoder confidence. Leading observed phonemes can be dropped at `UNIT * conf` each. */
  conf?: ArrayLike<number>;
  /** Only allow starts where this holds. Other starts cost `MID_WORD_START`. */
  strictStart?: boolean;
}

/**
 * The cost `alignSubstring(ref, lo, hi, obs, (o) => o === lo, { strictStart: true })` finds for `obs`, kept as `obs`
 * grows a phoneme at a time: one DP column per phoneme instead of the whole table per call. `cost()` is null where
 * that alignment is null.
 */
export function growingFit(ref: string, lo: number, hi: number) {
  const n = hi - lo;
  const del = new Float64Array(n + 1);
  for (let i = 1; i <= n; i++) del[i] = deletionCost(ref, lo + i - 1) + EDIT;
  let col = new Float64Array(n + 1).fill(INF);
  col[0] = 0;
  let cost = 0;
  let heard = 0;
  return {
    /** Phonemes pushed so far. */
    get heard() {
      return heard;
    },
    push(p: string) {
      const next = new Float64Array(n + 1);
      let min = (next[0] = col[0] + (insertionCost(ref, lo, p, hi) + EDIT));
      for (let i = 1; i <= n; i++) {
        const r = ref[lo + i - 1];
        let v = col[i] + (insertionCost(ref, lo + i, p, hi) + EDIT);
        const sub = col[i - 1] + (r === p ? 0 : subCost(r, p) + EDIT);
        if (sub < v) v = sub;
        const gap = next[i - 1] + del[i];
        if (gap < v) v = gap;
        next[i] = v;
        if (v < min) min = v;
      }
      col = next;
      cost = min;
      heard++;
    },
    // As `alignSubstring` reports it: the weighted cost, without the edit count that only breaks ties.
    cost: (): number | null => (cost < INF ? Math.floor(cost) : null),
  };
}

/** The least that inserting or dropping `p` costs anywhere: a held sound's lengthening, where `p` can be one. */
const indelFloor = (p: string) => indelCost(p, isLengthening(p, Infinity));

/**
 * The fits `growingFit(ref, start, end(start))` may find for `obs` below `bar`, for many starts at once: each start
 * whose fit may cost less than `bar`, with the least a fit of `obs` to the part of its reference walked may cost (a fit
 * that reads further may cost less), cheapest first. A start left out costs `bar` or more. `starts` are sorted by the
 * reference text from each start to its `end`, so starts that share their first phonemes sit together and are walked as
 * one branch of a trie: they share its DP rows, and since a row's cheapest cell bounds every row below it, a branch
 * whose row reaches `bar` is dropped whole. While a branch holds several starts, an insertion or deletion is priced at
 * the least it costs beside any neighbour (`indelFloor`); once it holds one, the reference prices them as `growingFit`
 * does.
 */
export function fitBounds(
  ref: string,
  starts: ArrayLike<number>,
  end: (start: number) => number,
  obs: string,
  bar: number,
): { start: number; bound: number }[] {
  const m = obs.length;
  const found: { start: number; bound: number }[] = [];
  // Per heard phoneme: the least an insertion of it costs anywhere, and what it costs beside no run of it.
  const floor = new Float64Array(m);
  const plain = new Float64Array(m);
  for (let j = 0; j < m; j++) [floor[j], plain[j]] = [indelFloor(obs[j]), indelCost(obs[j], false)];
  // Per reference phoneme, what substituting it for each heard phoneme costs.
  const subs = new Map<string, Float64Array>();
  const subRow = (r: string) => {
    let row = subs.get(r);
    if (!row) {
      row = new Float64Array(m);
      for (let j = 0; j < m; j++) row[j] = subCost(r, obs[j]);
      subs.set(r, row);
    }
    return row;
  };
  const rows: Float64Array[] = [new Float64Array(m + 1)];
  for (let j = 1; j <= m; j++) rows[0][j] = rows[0][j - 1] + floor[j - 1];
  // Starts [a, b) share their first `depth` phonemes, which `rows[depth]` aligns against each prefix of `obs`; `best`
  // is the cheapest fit of all of `obs` to some prefix of them (a fit's reference end is free).
  const walk = (a: number, b: number, depth: number, best: number) => {
    if (best < bar) {
      for (let k = a; k < b; k++) found.push({ start: starts[k], bound: best });
      return;
    }
    const row = rows[depth];
    const next = (rows[depth + 1] ??= new Float64Array(m + 1));
    for (let k = a; k < b; ) {
      const s = starts[k];
      const e = end(s);
      if (s + depth >= e) {
        k++;
        continue;
      }
      const r = ref[s + depth];
      let l = k + 1;
      while (l < b && starts[l] + depth < end(starts[l]) && ref[starts[l] + depth] === r) l++;
      const one = l - k === 1;
      const del = one ? deletionCost(ref, s + depth) : indelFloor(r);
      const sub = subRow(r);
      // With one start, an insertion beside a run of its sound (before or after it) is priced as `insertionCost` does.
      const at = s + depth + 1;
      const [before, after] = [ref[at - 1], at < e ? ref[at] : ""];
      let min = (next[0] = row[0] + del);
      for (let j = 1; j <= m; j++) {
        const p = obs[j - 1];
        let v = row[j] + del;
        const d = row[j - 1] + sub[j - 1];
        if (d < v) v = d;
        const ins = next[j - 1] + (!one ? floor[j - 1] : p === before || p === after ? insertionCost(ref, at, p, e) : plain[j - 1]);
        if (ins < v) v = ins;
        next[j] = v;
        if (v < min) min = v;
      }
      if (min < bar) walk(k, l, depth + 1, Math.min(best, next[m]));
      k = l;
    }
  };
  walk(0, starts.length, 0, rows[0][m]);
  return found.sort((x, y) => x.bound - y.bound);
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
  const d = new Float64Array((n + 1) * w);
  const at = (i: number, j: number) => i * w + j;
  const r = (i: number) => ref[lo + i - 1];
  const insCost = (i: number, j: number) => insertionCost(ref, lo + i, obs[j - 1], hi) + EDIT;
  const delCost = (i: number) => deletionCost(ref, lo + i - 1) + EDIT;
  const sub = (i: number, j: number) => (r(i) === obs[j - 1] ? 0 : subCost(r(i), obs[j - 1]) + EDIT);
  const startCost = (i: number) => (isStart(lo + i) ? 0 : options.strictStart ? INF : MID_WORD_START);
  const skip = new Float64Array(m + 1).fill(INF);
  skip[0] = 0;
  if (options.conf) for (let j = 1; j <= m; j++) skip[j] = skip[j - 1] + Math.round(UNIT * options.conf[j - 1]) + EDIT;

  // The fill below is the hot loop of global search; it reads costs from small tables instead of the closures above.
  const obsCode = new Int32Array(m + 1);
  const insBase = new Float64Array(m + 1);
  for (let j = 1; j <= m; j++) {
    obsCode[j] = tokenCode(obs[j - 1]);
    insBase[j] = indelCost(obs[j - 1], false) + EDIT;
  }
  for (let i = 0; i <= n; i++) {
    const start = startCost(i);
    const row = i * w;
    d[row] = start;
    // Insertion before ref[lo + i] only differs from its base cost beside a reference run of the same sound.
    const prevCode = lo + i > 0 ? tokenCode(ref[lo + i - 1]) : -1;
    const nextCode = lo + i < hi ? tokenCode(ref[lo + i]) : -1;
    const del = i > 0 ? delCost(i) : INF;
    const subRow = i > 0 ? subTable(prevCode) : null;
    for (let j = 1; j <= m; j++) {
      const c = obsCode[j];
      const ins = c === prevCode || c === nextCode ? insCost(i, j) : insBase[j];
      let best = Math.min(d[row + j - 1] + ins, start + skip[j]);
      if (subRow) best = Math.min(best, d[row - w + j - 1] + subRow[c], d[row - w + j] + del);
      d[row + j] = best;
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
    if (i > 0 && j > 0 && here === d[at(i - 1, j - 1)] + sub(i, j)) {
      ops.push(r(i) === obs[j - 1] ? { op: "match", ref: r(i), refAt: i - 1, obsAt: j - 1 } : { op: "sub", ref: r(i), obs: obs[j - 1], refAt: i - 1, obsAt: j - 1 });
      i--;
      j--;
    } else if (i > 0 && here === d[at(i - 1, j)] + delCost(i)) {
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
  return { start: lo + i, end: lo + endI, cost: Math.floor(d[at(endI, m)]), per: length ? edits / length : edits ? Infinity : 0, path: ops };
}
