import { BLANK_ID, TOKENS } from "./tokens.js";

/** Ring of recent per-frame CTC log-probabilities, indexed by decoder frame. */
export class FramePosteriors {
  readonly classes: number;
  readonly capacity: number;
  private readonly data: Float32Array;
  private readonly best: Float32Array;
  private readonly bestId: Int32Array;
  private start = 0;
  private end = 0;

  constructor(classes: number, capacity = 1500) {
    this.classes = classes;
    this.capacity = capacity;
    this.data = new Float32Array(classes * capacity);
    this.best = new Float32Array(capacity);
    this.bestId = new Int32Array(capacity);
  }

  /** Append `frames` rows (row-major [frames, classes]) starting at decoder frame `firstFrame`. */
  push(logProbs: ArrayLike<number>, frames: number, firstFrame: number): void {
    if (firstFrame !== this.end) this.clear(firstFrame);
    const c = this.classes;
    for (let t = 0; t < frames; t++) {
      const slot = this.end % this.capacity;
      let mx = -Infinity;
      let arg = 0;
      for (let k = 0; k < c; k++) {
        const v = logProbs[t * c + k]!;
        this.data[slot * c + k] = v;
        if (v > mx) {
          mx = v;
          arg = k;
        }
      }
      this.best[slot] = mx;
      this.bestId[slot] = arg;
      this.end++;
      if (this.end - this.start > this.capacity) this.start = this.end - this.capacity;
    }
  }

  clear(at = 0): void {
    this.start = at;
    this.end = at;
  }

  /** Frames [from, to) are all still held. */
  has(from: number, to: number): boolean {
    return from >= this.start && to <= this.end && from <= to;
  }

  get firstFrame(): number {
    return this.start;
  }

  get endFrame(): number {
    return this.end;
  }

  logp(frame: number, id: number): number {
    return this.data[(frame % this.capacity) * this.classes + id]!;
  }

  bestLogp(frame: number): number {
    return this.best[frame % this.capacity]!;
  }

  argmax(frame: number): number {
    return this.bestId[frame % this.capacity]!;
  }
}

const TOKEN_ID = new Map(TOKENS.map((t, i) => [t, i] as const));
const MAX_TOKEN = Math.max(...TOKENS.filter((t) => t !== "<blank>").map((t) => [...t].length));

/** Greedy longest-match phoneme -> token ids (same as the lab PhonemeTokenizer). Null on OOV. */
export function encodePhonemes(text: string): number[] | null {
  return encodePhonemeSpan(text, 0, text.length);
}

/** Tokenize `text` as a whole, keep the tokens that overlap chars [from, to).
 * Words are tokenized in context because a token can straddle a word boundary. */
export function encodePhonemeSpan(text: string, from: number, to: number): number[] | null {
  const chars = [...text];
  const ids: number[] = [];
  let i = 0;
  while (i < chars.length) {
    let hit = -1;
    let len = 0;
    for (let L = Math.min(MAX_TOKEN, chars.length - i); L > 0; L--) {
      const id = TOKEN_ID.get(chars.slice(i, i + L).join(""));
      if (id !== undefined) {
        hit = id;
        len = L;
        break;
      }
    }
    if (hit < 0) return null;
    if (i < to && i + len > from) ids.push(hit);
    i += len;
  }
  return ids;
}

/**
 * Viterbi log-likelihood of the best CTC path that emits exactly `ids` over
 * frames [from, to), with optional blanks around it. -Infinity if `ids` cannot
 * fit (too few frames).
 */
export function forcedLogLik(
  post: FramePosteriors,
  ids: readonly number[],
  from: number,
  to: number,
  blank = BLANK_ID,
): number {
  const L = ids.length;
  const S = 2 * L + 1;
  if (to <= from) return L === 0 ? 0 : -Infinity;
  const ext = new Int32Array(S);
  for (let s = 0; s < S; s++) ext[s] = s % 2 === 1 ? ids[(s - 1) >> 1]! : blank;
  let prev = new Float64Array(S).fill(-Infinity);
  let next = new Float64Array(S);
  prev[0] = post.logp(from, ext[0]!);
  if (S > 1) prev[1] = post.logp(from, ext[1]!);
  for (let t = from + 1; t < to; t++) {
    for (let s = 0; s < S; s++) {
      let m = prev[s]!;
      if (s >= 1 && prev[s - 1]! > m) m = prev[s - 1]!;
      if (s >= 2 && ext[s] !== blank && ext[s] !== ext[s - 2] && prev[s - 2]! > m) m = prev[s - 2]!;
      next[s] = m === -Infinity ? -Infinity : m + post.logp(t, ext[s]!);
    }
    [prev, next] = [next, prev];
  }
  return L === 0 ? prev[0]! : Math.max(prev[S - 1]!, prev[S - 2]!);
}

/** Sum of per-frame best log-probs over [from, to): the unconstrained competitor. */
export function freeLogLik(post: FramePosteriors, from: number, to: number): number {
  let s = 0;
  for (let t = from; t < to; t++) s += post.bestLogp(t);
  return s;
}

export interface WordGop {
  /** (forced - free) / tokens over the word's acoustic window; <= 0, ~0 when the expected word fits as well as anything. Floored at {@link GOP_FLOOR}. */
  gop: number;
  /** Same, forcing the expected word twice in a row (repetition hypothesis). */
  gopTwice: number;
  /** Same, forcing nothing but blank (omission hypothesis). */
  gopNone: number;
  /** {@link pairScores} through the next word, when available. */
  repGain?: number;
  pairGop?: number;
}

export const GOP_FLOOR = -20;

const floor = (x: number): number => (Number.isFinite(x) ? Math.max(GOP_FLOOR, x) : GOP_FLOOR);

/** Evidence over a window that runs through the next word, from token ids of
 * the word + next word (`once`) and the word twice + next word (`twice`),
 * each tokenized jointly. Per token of the word:
 * - `repGain`: forcing a second copy vs one (> 0 when a second copy is there),
 * - `pairGop`: forcing the pair vs the free decode (~0 when the two words fit
 *   together, i.e. a low single-word GOP was only a boundary misallocation). */
export function pairScores(
  post: FramePosteriors,
  wordTokens: number,
  once: readonly number[],
  twice: readonly number[],
  from: number,
  to: number,
): { repGain: number; pairGop: number } | null {
  if (!wordTokens || !once.length || !post.has(from, to)) return null;
  const free = freeLogLik(post, from, to);
  const one = forcedLogLik(post, once, from, to);
  const two = forcedLogLik(post, twice, from, to);
  const clamp = (x: number) => Math.max(GOP_FLOOR, Math.min(-GOP_FLOOR, x));
  const repGain = !Number.isFinite(two) ? GOP_FLOOR : !Number.isFinite(one) ? -GOP_FLOOR : clamp((two - one) / wordTokens);
  return { repGain, pairGop: floor((one - free) / wordTokens) };
}

/** Goodness-of-pronunciation style scores for `ids` over frames [from, to). Null if frames are gone. */
export function wordGop(post: FramePosteriors, ids: readonly number[], from: number, to: number): WordGop | null {
  if (!ids.length || !post.has(from, to)) return null;
  const free = freeLogLik(post, from, to);
  const n = ids.length;
  return {
    gop: floor((forcedLogLik(post, ids, from, to) - free) / n),
    gopTwice: floor((forcedLogLik(post, [...ids, ...ids], from, to) - free) / n),
    gopNone: floor((forcedLogLik(post, [], from, to) - free) / n),
  };
}
