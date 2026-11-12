import type { WordRef } from "./events.js";

export interface ReferenceAsset {
  version: string;
  text: string;
  profile: Record<string, unknown>;
  inventory: { id: string; tokens: string };
  ayahs: { s: number; a: number; w: string[]; p: string; o: number[]; iso: string[] }[];
}

/** Non-Quran phrases said around a recitation (isti'adha, takbir, amin...), phonemized with the reference profile. */
export interface PhraseAsset {
  version: string;
  phrases: { name: string; text: string; p: string }[];
}

export interface AyahRef {
  surah: number;
  ayah: number;
}

/** Madani (Hafs, 604-page) mushaf page boundaries: `pages[n - 1]` is `[fromSurah, fromAyah, toSurah, toAyah]` of page n. */
export interface PageAsset {
  version: string;
  source: string;
  sourceSha256: string;
  pages: [number, number, number, number][];
}

export function pageRange(asset: PageAsset, page: number): { from: AyahRef; to: AyahRef } {
  const p = asset.pages[page - 1];
  if (!Number.isInteger(page) || !p) throw new Error(`No Madani page ${page}`);
  return { from: { surah: p[0], ayah: p[1] }, to: { surah: p[2], ayah: p[3] } };
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
  /** Supported connected and isolated pronunciations, in word order. */
  wordForms?: readonly (readonly string[])[];
  /** Context-free pause pronunciations, kept separate from connected forms. */
  isolatedWordForms?: readonly string[];
  /** Mushaf word text, in word order. Letter identity for parallel phrases; absent on hand-built indexes. */
  wordText?: readonly string[];
  gramSize: number;
  grams: ReadonlyMap<string, readonly number[]>;
  /** Index into `words` of the word covering a phoneme offset. */
  wordAt(offset: number): number;
}

const GRAM_SIZE = 4;

/**
 * The connected form opened the way the word is said after a pause. The previous word's assimilated ending doubles
 * this word's first consonant in connected speech (وجوهٌ يومئذ: يييَومَءِذِن); a reader who stops between the two
 * starts it plainly, with the rest still joined to what follows (يَومَءِذِن). The isolated form keeps the plain onset.
 */
function plainOnset(connected: string, iso: string | undefined): string | undefined {
  if (!connected || !iso || connected[0] !== iso[0]) return undefined;
  const run = (s: string) => { let n = 1; while (s[n] === s[0]) n++; return n; };
  const extra = run(connected) - run(iso);
  return extra > 0 ? connected.slice(extra) : undefined;
}

export function loadReferenceIndex(asset: ReferenceAsset): ReferenceIndex {
  const words: ReferenceWord[] = [];
  const wordForms: string[][] = [];
  const isolatedWordForms: string[] = [];
  const wordText: string[] = [];
  let phonemes = "";
  for (const ayah of asset.ayahs) {
    const base = phonemes.length;
    ayah.w.forEach((text, i) => {
      const end = i + 1 < ayah.o.length ? ayah.o[i + 1] : ayah.p.length;
      words.push({ surah: ayah.s, ayah: ayah.a, word: i + 1, start: base + ayah.o[i], end: base + end });
      const connected = ayah.p.slice(ayah.o[i], end);
      wordForms.push([...new Set([connected, ayah.iso[i], plainOnset(connected, ayah.iso[i])].filter((p): p is string => !!p))]);
      isolatedWordForms.push(ayah.iso[i]);
      wordText.push(text);
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
    wordForms,
    isolatedWordForms,
    wordText,
    gramSize: GRAM_SIZE,
    grams,
    wordAt,
  };
}
