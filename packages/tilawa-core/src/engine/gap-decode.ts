import { align } from "./align.js";

/**
 * Context-free gap decode. The streaming model carries a long left context, so speech in a pause between two
 * matched words can come out blank or as the verse word the pause should not contain. A second decode of just
 * that gap has no verse context. Phonemes it recovers, and that are not the anchored neighbours, are the ones
 * the checker's added-word rule can confirm.
 *
 * `GAP_PAD_S` is both the audio kept either side of the gap (so the cropped model still sees the consonants at
 * the edges) and the shortest gap worth re-decoding: anything shorter is neighbour audio, not a pause.
 * `GAP_MIN_CONF` is the checker's default confidence for treating a decoded phone as sure.
 */
export const GAP_PAD_S = 0.3;
export const GAP_MIN_CONF = 0.6;

/** Same consonant inventory the checker uses to decide an insertion is a word rather than a held vowel. */
const CONSONANTS = new Set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي");
/** The checker's `MIN_ADDED`: fewer phones than this never become an added word. */
const MIN_ADDED = 3;

export interface GapPhoneme {
  p: string;
  /** Inclusive frame index. */
  start: number;
  end: number;
  conf: number;
}

export interface WordGap {
  /** Half-open frame range between the left word's last aligned phone and the right word's first. */
  fromFrame: number;
  toFrame: number;
  /** Reference phonemes of the anchored words. */
  left: string;
  right: string;
}

export function minGapFrames(frameMs: number): number {
  return Math.max(1, Math.round((GAP_PAD_S * 1000) / frameMs));
}

/** Phones already in the gap are replaced by the cropped decode; everything outside the gap stays. */
export function placeGap(phonemes: GapPhoneme[], gap: WordGap, kept: GapPhoneme[]): GapPhoneme[] {
  const placed = kept.map((p) => ({
    ...p,
    start: Math.max(gap.fromFrame, Math.min(gap.toFrame - 1, p.start)),
    end: Math.max(gap.fromFrame, Math.min(gap.toFrame - 1, Math.max(p.start, p.end))),
  }));
  return [
    ...phonemes.filter((p) => (p.start + p.end) / 2 < gap.fromFrame || (p.start + p.end) / 2 >= gap.toFrame),
    ...placed,
  ].sort((a, b) => a.start - b.start || a.end - b.end);
}

/**
 * An insertion the checker can call an added word: at least `MIN_ADDED` phones and two consonants.
 * Shorter recoveries are breath, a held waqf vowel, or one noisy edge, and are not spliced.
 */
export function gapCarriesAddition(phones: { p: string }[]): boolean {
  if (phones.length < MIN_ADDED) return false;
  let consonants = 0;
  for (const ph of phones) if (CONSONANTS.has(ph.p)) consonants++;
  return consonants >= 2;
}

/**
 * Phones from a context-free decode of the padded gap whose centres lie in `[gapStart, gapEnd)`,
 * dropping a prefix that repeats the left word and a suffix that repeats the right word.
 */
export function recoverGapPhonemes(decoded: GapPhoneme[], gapStart: number, gapEnd: number, left: string, right: string): GapPhoneme[] {
  const mid = decoded.filter((p) => {
    const centre = (p.start + p.end) / 2;
    return centre >= gapStart && centre < gapEnd;
  });
  let a = 0;
  const leftChars = [...left];
  for (let k = Math.min(mid.length, leftChars.length); k > 0; k--) {
    if (left.endsWith(mid.slice(0, k).map((p) => p.p).join(""))) {
      a = k;
      break;
    }
  }
  const last = leftChars.at(-1);
  while (a < mid.length && last !== undefined && mid[a].p === last) a++;
  let b = mid.length;
  const rightChars = [...right];
  for (let k = Math.min(b - a, rightChars.length); k > 0; k--) {
    if (right.startsWith(mid.slice(b - k, b).map((p) => p.p).join(""))) {
      b -= k;
      break;
    }
  }
  const first = rightChars[0];
  while (b > a && first !== undefined && mid[b - 1].p === first) b--;
  return mid.slice(a, b);
}

/**
 * Frame gaps between consecutive reference words in one monotonic recitation of `obs` against `phonemes[from, to)`.
 * A word anchors a side only when some phone aligned to it is at least `minConf`. The boundary itself is that
 * word's first and last aligned phone, so a low-confidence tail of the word stays with the word.
 */
export function interWordGaps(args: {
  phonemes: string;
  from: number;
  to: number;
  wordAt: (offset: number) => number;
  wordText: (wi: number) => string;
  obs: GapPhoneme[];
  minFrames: number;
  minConf?: number;
}): WordGap[] {
  const { phonemes, from, to, wordAt, wordText, obs, minFrames } = args;
  const minConf = args.minConf ?? GAP_MIN_CONF;
  if (!obs.length || to <= from) return [];
  const path = align(phonemes.slice(from, to), obs.map((p) => p.p).join(""));
  const segs: { wi: number; first: number; last: number; confident: boolean }[] = [];
  for (const op of path) {
    if (op.op !== "match" && op.op !== "sub") continue;
    const wi = wordAt(from + op.refAt);
    const ph = obs[op.obsAt];
    const tail = segs.at(-1);
    if (!tail || tail.wi !== wi) segs.push({ wi, first: ph.start, last: ph.end, confident: ph.conf >= minConf });
    else {
      tail.last = ph.end;
      if (ph.conf >= minConf) tail.confident = true;
    }
  }
  const gaps: WordGap[] = [];
  for (let i = 0; i < segs.length - 1; i++) {
    const left = segs[i];
    const right = segs[i + 1];
    if (right.wi !== left.wi + 1 || !left.confident || !right.confident) continue;
    const fromFrame = left.last + 1;
    const toFrame = right.first;
    if (toFrame - fromFrame < minFrames) continue;
    const leftText = wordText(left.wi);
    const rightText = wordText(right.wi);
    if (!leftText || !rightText) continue;
    gaps.push({ fromFrame, toFrame, left: leftText, right: rightText });
  }
  return gaps;
}
