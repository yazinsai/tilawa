/**
 * Display text for the engine's word positions. The engine counts words in its reference (ref-v2); the page shows
 * Uthmani text from public/quran.json. They agree once the display text is cleaned up, except for three ayahs where
 * quran.json writes بَعْدَمَا as one word and the reference splits it in two.
 */

export interface QuranVerse {
  surah: number;
  ayah: number;
  text_uthmani: string;
  surah_name: string;
  surah_name_en: string;
}

export interface DisplayAyah {
  surah: number;
  ayah: number;
  surahName: string;
  surahNameEn: string;
  words: string[];
}

// Pause marks (ۖ … ۜ) and the sajda mark (۩) attach to the word before them.
const ATTACHED = /^[\u06D6-\u06DC\u06E9]{1,2}$/;
const RUB_EL_HIZB = "\u06DE";
const BISMILLAH_WORDS = 4;

/** Uthmani words as the reference counts them for this ayah. */
export function splitDisplayWords(text: string, surah: number, ayah: number): string[] {
  const out: string[] = [];
  for (const token of text.replace(/\uFEFF/g, "").split(/\s+/)) {
    if (!token || token === RUB_EL_HIZB) continue;
    if (ATTACHED.test(token) && out.length) out[out.length - 1] += ` ${token}`;
    else out.push(token);
  }
  // quran.json prefixes the basmala to the first ayah of every surah but 1 and 9; the reference doesn't.
  if (ayah === 1 && surah !== 1 && surah !== 9 && out.length > BISMILLAH_WORDS) return out.slice(BISMILLAH_WORDS);
  return out;
}

/** "surah:ayah" → the display word that holds two reference words (word numbers are 1-based). */
const MERGED: Record<string, number> = { "2:181": 3, "8:6": 4, "13:37": 8 };

/** The 0-based display word for a 1-based reference word. */
export function displayIndex(surah: number, ayah: number, word: number): number {
  const merged = MERGED[`${surah}:${ayah}`];
  return merged !== undefined && word > merged ? word - 2 : word - 1;
}

export type QuranText = Map<string, DisplayAyah>;

export const ayahKey = (surah: number, ayah: number) => `${surah}:${ayah}`;

export function indexQuran(verses: QuranVerse[]): QuranText {
  const map: QuranText = new Map();
  for (const v of verses)
    map.set(ayahKey(v.surah, v.ayah), {
      surah: v.surah,
      ayah: v.ayah,
      surahName: v.surah_name,
      surahNameEn: v.surah_name_en,
      words: splitDisplayWords(v.text_uthmani, v.surah, v.ayah),
    });
  return map;
}

export interface SurahInfo {
  surah: number;
  nameEn: string;
  name: string;
  ayahs: number;
}

export function surahList(text: QuranText): SurahInfo[] {
  const out: SurahInfo[] = [];
  for (const a of text.values()) {
    const s = out[a.surah - 1] ?? (out[a.surah - 1] = { surah: a.surah, nameEn: a.surahNameEn, name: a.surahName, ayahs: 0 });
    s.ayahs = Math.max(s.ayahs, a.ayah);
  }
  return out;
}
