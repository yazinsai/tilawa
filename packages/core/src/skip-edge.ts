import type { PosteriorFrames } from "./ctc.js";

function logAdd(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  return a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b));
}

/**
 * CTC forward sum of `labels` over frames `[from, to)`, in nats. From the end of each word an arc goes back to
 * the start of every word up to and including it: the checker's contiguous repeat, a jump back onto text just
 * read, not a filler and not a skip. No arc cost. `wordLast[i]` is the label index of word i's last phone.
 */
export function sequenceNll(frames: PosteriorFrames, labels: ArrayLike<number>, wordLast: readonly number[], from: number, to: number): number {
  const L = labels.length;
  if (to <= from) return L ? Infinity : 0;
  const S = 2 * L + 1;
  const label = (s: number) => (s % 2 === 0 ? 0 : labels[(s - 1) >> 1]);
  const into: number[][] = Array.from({ length: S }, () => []);
  const starts: number[] = [];
  let prevEnd = -1;
  for (const end of wordLast) {
    starts.push(prevEnd + 1);
    prevEnd = end;
  }
  // From the end of word i, back to the start of word j (j <= i): re-read a contiguous span just finished.
  for (let i = 0; i < wordLast.length; i++) {
    const last = wordLast[i];
    if (last < starts[i] || last >= L) continue;
    const srcs = [2 * last + 1, 2 * last + 2].filter((src) => src < S);
    for (let j = 0; j <= i; j++) {
      const dest = 2 * starts[j] + 1;
      for (const src of srcs) if (src !== dest) into[dest].push(src);
    }
  }
  let prev = new Float64Array(S).fill(-Infinity);
  let next = new Float64Array(S);
  const r0 = frames.row(from);
  prev[0] = r0[0];
  if (L) prev[1] = r0[label(1)];
  for (let f = from + 1; f < to; f++) {
    const row = frames.row(f);
    for (let s = 0; s < S; s++) {
      let a = prev[s];
      if (s > 0) a = logAdd(a, prev[s - 1]);
      if (s > 1 && s % 2 === 1 && label(s) !== label(s - 2)) a = logAdd(a, prev[s - 2]);
      for (const src of into[s]) a = logAdd(a, prev[src]);
      next[s] = a + row[label(s)];
    }
    const swap = prev;
    prev = next;
    next = swap;
  }
  const end = L ? logAdd(prev[S - 1], prev[S - 2]) : prev[0];
  return -end;
}
