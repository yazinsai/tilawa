import { describe, it, expect } from "vitest";
import { QuranDB } from "../src/quran-db";
import type { QuranVerse } from "../src/types";

/**
 * The packed n-gram cache (`_packNgrams` / `_intersectionSizePacked`) exists purely
 * as a memory/speed substitute for the `Set<string>` pair (`_jointNgrams` /
 * `_intersectionSize`). It is only a legitimate substitute if the overlap scores it
 * produces are bit-identical, so that candidate ordering never moves.
 *
 * These tests pin that equivalence, including on the fallback path taken when a
 * corpus has too many distinct characters to intern.
 */

// Deterministic pseudo-corpus: enough verses that `_jointCandidateVerses` gets past
// its "fewer than 80 overlapping verses -> return everything" safety valve.
const PHONEME_ALPHABET = "abdfghijklmnqrstuwxyzHSTZDE^<*$";

function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeVerse(surah: number, ayah: number, phonemes: string): QuranVerse {
  return {
    surah,
    ayah,
    text_uthmani: `${surah}:${ayah}`,
    surah_name: `s${surah}`,
    surah_name_en: `s${surah}`,
    phonemes,
    phonemes_joined: phonemes,
    phoneme_words: phonemes.split(/\s+/).filter(Boolean),
  } as QuranVerse;
}

function buildCorpus(alphabet = PHONEME_ALPHABET, verseCount = 400): QuranVerse[] {
  const rand = mulberry32(0x7ada);
  const verses: QuranVerse[] = [];
  for (let i = 0; i < verseCount; i++) {
    const wordCount = 3 + Math.floor(rand() * 8);
    const words: string[] = [];
    for (let w = 0; w < wordCount; w++) {
      const len = 2 + Math.floor(rand() * 5);
      let word = "";
      for (let c = 0; c < len; c++) {
        word += alphabet[Math.floor(rand() * alphabet.length)];
      }
      words.push(word);
    }
    verses.push(makeVerse(Math.floor(i / 20) + 1, (i % 20) + 1, words.join(" ")));
  }
  return verses;
}

/** Original `Set`-based overlap, reproduced here as the reference implementation. */
function referenceOverlap(query: string, ref: string): number {
  const ngrams = (s: string, n: number) => {
    const out = new Set<string>();
    for (let i = 0; i + n <= s.length; i++) out.add(s.slice(i, i + n));
    return out;
  };
  const inter = (a: Set<string>, b: Set<string>) => {
    let count = 0;
    for (const item of a) if (b.has(item)) count++;
    return count;
  };
  return (
    inter(ngrams(query, 2), ngrams(ref, 2)) +
    0.48 * inter(ngrams(query, 3), ngrams(ref, 3))
  );
}

describe("packed n-gram cache", () => {
  const verses = buildCorpus();
  const db = new QuranDB(verses) as unknown as {
    _ngramCharTable(): Int32Array | null;
    _jointCandidateVerses(noSpaceText: string, maxCandidates?: number): QuranVerse[];
  };

  const packNgrams = (QuranDB as unknown as {
    _packNgrams(s: string, n: number, charIds: Int32Array): Int32Array;
  })._packNgrams;
  const intersectPacked = (QuranDB as unknown as {
    _intersectionSizePacked(a: Int32Array, b: Int32Array): number;
  })._intersectionSizePacked;

  it("interns the whole corpus alphabet", () => {
    const charIds = db._ngramCharTable();
    expect(charIds).not.toBeNull();
    for (const ch of PHONEME_ALPHABET) {
      expect(charIds![ch.charCodeAt(0)]).toBeGreaterThan(0);
    }
  });

  it("produces sorted, deduplicated keys", () => {
    const charIds = db._ngramCharTable()!;
    for (const n of [2, 3]) {
      const packed = packNgrams("abababcabc", n, charIds);
      for (let i = 1; i < packed.length; i++) {
        expect(packed[i]).toBeGreaterThan(packed[i - 1]);
      }
      const distinct = new Set<string>();
      for (let i = 0; i + n <= "abababcabc".length; i++) {
        distinct.add("abababcabc".slice(i, i + n));
      }
      expect(packed.length).toBe(distinct.size);
    }
  });

  it("scores overlap identically to the Set implementation", () => {
    const charIds = db._ngramCharTable()!;
    const rand = mulberry32(0x1234);
    // Queries drawn from real verse text, corrupted the way ASR corrupts it.
    for (let trial = 0; trial < 300; trial++) {
      const source = verses[Math.floor(rand() * verses.length)].phonemes_joined;
      let query = source.replace(/ /g, "");
      const cut = Math.floor(rand() * query.length);
      query = query.slice(cut) + PHONEME_ALPHABET[Math.floor(rand() * PHONEME_ALPHABET.length)];

      for (let k = 0; k < 6; k++) {
        const ref = verses[Math.floor(rand() * verses.length)].phonemes_joined.replace(/ /g, "");
        const packedOv =
          intersectPacked(packNgrams(query, 2, charIds), packNgrams(ref, 2, charIds)) +
          0.48 * intersectPacked(packNgrams(query, 3, charIds), packNgrams(ref, 3, charIds));
        expect(packedOv).toBe(referenceOverlap(query, ref));
      }
    }
  });

  it("never matches a query character the corpus has never seen", () => {
    const charIds = db._ngramCharTable()!;
    // Two distinct out-of-alphabet characters both intern to id 0, so their n-grams
    // may collide with each other — but neither may collide with corpus text.
    for (const query of ["المab", "ھab", "??ab", "\u0000ab"]) {
      for (const verse of verses.slice(0, 40)) {
        const ref = verse.phonemes_joined.replace(/ /g, "");
        const packedOv =
          intersectPacked(packNgrams(query, 2, charIds), packNgrams(ref, 2, charIds)) +
          0.48 * intersectPacked(packNgrams(query, 3, charIds), packNgrams(ref, 3, charIds));
        expect(packedOv).toBe(referenceOverlap(query, ref));
      }
    }
  });

  it("shortlists identically on the packed and Set fallback paths", () => {
    const rand = mulberry32(0xbeef);
    const queries: string[] = [];
    for (let i = 0; i < 40; i++) {
      const source = verses[Math.floor(rand() * verses.length)].phonemes_joined.replace(/ /g, "");
      queries.push(source.slice(0, 4 + Math.floor(rand() * source.length)));
    }
    queries.push("", "ab", "abc", "abcd");

    const packedResults = queries.map((q) =>
      db._jointCandidateVerses(q).map((v) => `${v.surah}:${v.ayah}`),
    );

    // Force the fallback branch the same way an over-large alphabet would.
    const fallbackDb = new QuranDB(verses) as unknown as {
      _ngramCharIds: Int32Array | null;
      _ngramCharIdsBuilt: boolean;
      _jointCandidateVerses(noSpaceText: string, maxCandidates?: number): QuranVerse[];
    };
    fallbackDb._ngramCharIds = null;
    fallbackDb._ngramCharIdsBuilt = true;

    const fallbackResults = queries.map((q) =>
      fallbackDb._jointCandidateVerses(q).map((v) => `${v.surah}:${v.ayah}`),
    );

    expect(packedResults).toEqual(fallbackResults);
    // Guard against the comparison being vacuous: at least one query must have been
    // narrowed rather than falling through to the whole corpus.
    expect(packedResults.some((r) => r.length < verses.length)).toBe(true);
  });

  it("falls back rather than colliding when the alphabet is too large", () => {
    // 1200 distinct characters exceeds the 1023 ids a 10-bit field can hold.
    let alphabet = "";
    for (let i = 0; i < 1200; i++) alphabet += String.fromCharCode(0x3000 + i);
    const wideDb = new QuranDB(buildCorpus(alphabet, 120)) as unknown as {
      _ngramCharTable(): Int32Array | null;
    };
    expect(wideDb._ngramCharTable()).toBeNull();
  });
});
