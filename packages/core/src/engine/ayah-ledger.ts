import type { PosteriorFrames } from "./ctc.js";

/**
 * One closed hypothesis over stored posteriors: ayahs in mushaf order, each word a contiguous
 * repeat arc and each ayah one too. The same forward sum the parallel phrase graph uses, plus
 * those repeat arcs. There is no phone-loop filler.
 */
export function hypothesis(ayahs: readonly (readonly string[])[], idOf: (phone: string) => number): { ids: number[]; repeats: [number, number][] } {
  const ids: number[] = [];
  const repeats: [number, number][] = [];
  const span = (start: number) => {
    const end = ids.length - 1;
    if (end >= start) repeats.push([start, end]);
  };
  for (const words of ayahs) {
    const ayahStart = ids.length;
    for (const word of words) {
      if (!word) continue;
      const start = ids.length;
      for (const phone of word) ids.push(idOf(phone));
      if (ids.length > start) span(start);
    }
    if (ids.length > ayahStart) span(ayahStart);
  }
  return { ids, repeats };
}

/**
 * How many nats the in-order text with `ayahs[drop]` removed beats the text that keeps it,
 * on frames `[from, to)`. Positive means the ayah is absent. Null when either sum is impossible.
 */
export function absenceMargin(
  frames: PosteriorFrames,
  idOf: (phone: string) => number,
  ayahs: readonly (readonly string[])[],
  drop: number,
  from: number,
  to: number,
): number | null {
  if (drop < 0 || drop >= ayahs.length || to <= from) return null;
  const read = hypothesis(ayahs, idOf);
  const skip = hypothesis(ayahs.filter((_, i) => i !== drop), idOf);
  if (!read.ids.length || !skip.ids.length) return null;
  const readNll = frames.nll(read.ids, from, to, read.repeats);
  const skipNll = frames.nll(skip.ids, from, to, skip.repeats);
  // A read path the frames cannot carry at all still loses to a finite skip. Both impossible is no evidence.
  if (!Number.isFinite(skipNll)) return Number.isFinite(readNll) ? -Infinity : null;
  if (!Number.isFinite(readNll)) return Infinity;
  return readNll - skipNll;
}
