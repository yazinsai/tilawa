import madaniData from "./madani-pages.json" with { type: "json" };
import { ayahKey, type DisplayAyah, type QuranText } from "./quran-words.js";

export type PageBoundary = [fromSurah: number, fromAyah: number, toSurah: number, toAyah: number];

export const TOTAL_PAGES = 604;
export const pages: PageBoundary[] = (madaniData as unknown as { pages: PageBoundary[] }).pages;

export const JUZ_START_PAGES = [
  1, 22, 42, 62, 82, 102, 122, 142, 162, 182, 202, 222, 242, 262, 282,
  302, 322, 342, 362, 382, 402, 422, 442, 462, 482, 502, 522, 542, 562, 582,
];

export const JUZ_NAMES_AR = [
  "الجزء الأول", "الجزء الثاني", "الجزء الثالث", "الجزء الرابع", "الجزء الخامس",
  "الجزء السادس", "الجزء السابع", "الجزء الثامن", "الجزء التاسع", "الجزء العاشر",
  "الجزء الحادي عشر", "الجزء الثاني عشر", "الجزء الثالث عشر", "الجزء الرابع عشر", "الجزء الخامس عشر",
  "الجزء السادس عشر", "الجزء السابع عشر", "الجزء الثامن عشر", "الجزء التاسع عشر", "الجزء العشرون",
  "الجزء الحادي والعشرون", "الجزء الثاني والعشرون", "الجزء الثالث والعشرون", "الجزء الرابع والعشرون", "الجزء الخامس والعشرون",
  "الجزء السادس والعشرون", "الجزء السابع والعشرون", "الجزء الثامن والعشرون", "الجزء التاسع والعشرون", "الجزء الثلاثون",
];

export function getPageForVerse(surah: number, ayah: number): number {
  const idx = pages.findIndex(
    ([fs, fa, ts, ta]) =>
      (surah > fs || (surah === fs && ayah >= fa)) &&
      (surah < ts || (surah === ts && ayah <= ta))
  );
  return idx >= 0 ? idx + 1 : 1;
}

export function getJuzForPage(page: number): { number: number; nameAr: string; nameEn: string } {
  let juzNum = 1;
  for (let i = 0; i < JUZ_START_PAGES.length; i++) {
    if (page >= JUZ_START_PAGES[i]) {
      juzNum = i + 1;
    } else {
      break;
    }
  }
  return {
    number: juzNum,
    nameAr: JUZ_NAMES_AR[juzNum - 1] ?? `الجزء ${juzNum}`,
    nameEn: `Juz ${juzNum}`,
  };
}

export function getAyahsForPage(page: number, quran: QuranText): DisplayAyah[] {
  if (page < 1 || page > TOTAL_PAGES) return [];
  const [fs, fa, ts, ta] = pages[page - 1];
  const list: DisplayAyah[] = [];

  for (let s = fs; s <= ts; s++) {
    const startA = s === fs ? fa : 1;
    for (let a = startA; ; a++) {
      if (s === ts && a > ta) break;
      const v = quran.get(ayahKey(s, a));
      if (!v) break;
      list.push(v);
    }
  }
  return list;
}

export const toArabicDigits = (n: number): string =>
  String(n).replace(/\d/g, (d) => "٠١٢٣٤٥٦٧٨٩"[Number(d)]);
