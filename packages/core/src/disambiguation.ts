/** Disambiguation points (GLOSSARY): how far a reciter starting at a word must get before the text is unique. */
import type { ReferenceIndex } from "./reference.js";

/** Prefix-doubling suffix array of `s`. */
export function suffixArray(s: string): Int32Array {
  const n = s.length;
  const sa = Int32Array.from({ length: n }, (_, i) => i);
  let rank = Int32Array.from(s, (c) => c.charCodeAt(0));
  let next = new Int32Array(n);
  for (let k = 1; ; k <<= 1) {
    const key = (i: number) => (i + k < n ? rank[i + k] + 1 : 0);
    sa.sort((a, b) => rank[a] - rank[b] || key(a) - key(b));
    next[sa[0]] = 0;
    for (let r = 1; r < n; r++) next[sa[r]] = next[sa[r - 1]] + (rank[sa[r]] !== rank[sa[r - 1]] || key(sa[r]) !== key(sa[r - 1]) ? 1 : 0);
    [rank, next] = [next, rank];
    if (rank[sa[n - 1]] === n - 1) return sa;
  }
}

/** Kasai: `lcp[r]` is the common prefix length of the suffixes at `sa[r - 1]` and `sa[r]` (`lcp[0] = 0`). */
export function lcpArray(s: string, sa: Int32Array): { lcp: Int32Array; rank: Int32Array } {
  const n = s.length, rank = new Int32Array(n), lcp = new Int32Array(n);
  for (let r = 0; r < n; r++) rank[sa[r]] = r;
  for (let i = 0, h = 0; i < n; i++) {
    if (rank[i] === 0) { h = 0; continue; }
    const j = sa[rank[i] - 1];
    while (i + h < n && j + h < n && s[i + h] === s[j + h]) h++;
    lcp[rank[i]] = h;
    if (h) h--;
  }
  return { lcp, rank };
}

/**
 * For every word, the number of further words a reciter starting there must recite before the phonemes so far occur
 * nowhere else in the Quran (0: the first word alone is unique), or -1 when the text never becomes unique. Exact
 * phoneme matches only, so near-identical passages disambiguate earlier here than they can in noisy evidence, and a
 * case ending that differs only because one copy ends its ayah can make a point one word early for a reciter who
 * pauses there.
 */
export function disambiguationWords(index: ReferenceIndex): Int32Array {
  const s = index.phonemes, n = s.length;
  const { lcp, rank } = lcpArray(s, suffixArray(s));
  const words = index.words, out = new Int32Array(words.length);
  words.forEach((w, i) => {
    const r = rank[w.start];
    const need = w.start + 1 + Math.max(lcp[r], r + 1 < n ? lcp[r + 1] : 0);
    let lo = i, hi = words.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (words[mid].end >= need) hi = mid; else lo = mid + 1; }
    out[i] = need > n || lo === words.length ? -1 : lo - i;
  });
  return out;
}
