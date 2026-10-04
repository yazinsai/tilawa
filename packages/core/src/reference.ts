import type { WordRef } from "./events.js";

export interface ReferenceAsset {
  version: string;
  text: string;
  profile: Record<string, unknown>;
  inventory: { id: string; tokens: string };
  ayahs: { s: number; a: number; w: string[]; p: string; o: number[]; iso: string[] }[];
}

export interface ReferenceWord extends WordRef {
  /** Half-open range in `ReferenceIndex.phonemes`. Empty when fully assimilated into a neighbour. */
  start: number;
  end: number;
}

export interface ReferenceIndex {
  version: string;
  inventoryId: string;
  tokens: ReadonlySet<string>;
  /** Every ayah's continuous phonemes, concatenated in mushaf order. */
  phonemes: string;
  words: ReferenceWord[];
  gramSize: number;
  grams: ReadonlyMap<string, readonly number[]>;
  /** Index into `words` of the word covering a phoneme offset. */
  wordAt(offset: number): number;
}

const GRAM_SIZE = 4;

export function loadReferenceIndex(asset: ReferenceAsset): ReferenceIndex {
  const words: ReferenceWord[] = [];
  let phonemes = "";
  for (const ayah of asset.ayahs) {
    const base = phonemes.length;
    ayah.w.forEach((_, i) => {
      const end = i + 1 < ayah.o.length ? ayah.o[i + 1] : ayah.p.length;
      words.push({ surah: ayah.s, ayah: ayah.a, word: i + 1, start: base + ayah.o[i], end: base + end });
    });
    phonemes += ayah.p;
  }

  const grams = new Map<string, number[]>();
  for (let i = 0; i + GRAM_SIZE <= phonemes.length; i++) {
    const g = phonemes.slice(i, i + GRAM_SIZE);
    const list = grams.get(g);
    if (list) list.push(i);
    else grams.set(g, [i]);
  }

  const starts = words.map((w) => w.start);
  const wordAt = (offset: number) => {
    let lo = 0;
    let hi = starts.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (starts[mid] <= offset) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  };

  return {
    version: asset.version,
    inventoryId: asset.inventory.id,
    tokens: new Set(asset.inventory.tokens),
    phonemes,
    words,
    gramSize: GRAM_SIZE,
    grams,
    wordAt,
  };
}
