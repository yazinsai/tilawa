import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { getAyahsForPage, getJuzForPage, getPageForVerse, TOTAL_PAGES } from "../src/lib/mushaf";
import { indexQuran, type QuranVerse } from "../src/lib/quran-words";

const verses = JSON.parse(readFileSync(new URL("../public/quran.json", import.meta.url), "utf8")) as QuranVerse[];
const quran = indexQuran(verses);

it("maps verses to their Madani mushaf page", () => {
  expect(getPageForVerse(1, 1)).toBe(1);
  expect(getPageForVerse(1, 7)).toBe(1);
  expect(getPageForVerse(2, 1)).toBe(2);
  expect(getPageForVerse(2, 5)).toBe(2);
  expect(getPageForVerse(2, 6)).toBe(3);
  expect(getPageForVerse(112, 1)).toBe(604);
  expect(getPageForVerse(113, 1)).toBe(604);
  expect(getPageForVerse(114, 6)).toBe(604);
});

it("returns the exact ayahs for a page", () => {
  const p1 = getAyahsForPage(1, quran);
  expect(p1.map((a) => `${a.surah}:${a.ayah}`)).toEqual(["1:1", "1:2", "1:3", "1:4", "1:5", "1:6", "1:7"]);

  const p2 = getAyahsForPage(2, quran);
  expect(p2.map((a) => `${a.surah}:${a.ayah}`)).toEqual(["2:1", "2:2", "2:3", "2:4", "2:5"]);

  const p604 = getAyahsForPage(604, quran);
  expect(p604.map((a) => `${a.surah}:${a.ayah}`)).toEqual([
    "112:1", "112:2", "112:3", "112:4",
    "113:1", "113:2", "113:3", "113:4", "113:5",
    "114:1", "114:2", "114:3", "114:4", "114:5", "114:6",
  ]);

  expect(getAyahsForPage(0, quran)).toEqual([]);
  expect(getAyahsForPage(TOTAL_PAGES + 1, quran)).toEqual([]);
});

it("covers every ayah across all 604 pages without gaps or duplicates", () => {
  const seen = new Set<string>();
  for (let page = 1; page <= TOTAL_PAGES; page++) {
    const ayahs = getAyahsForPage(page, quran);
    expect(ayahs.length).toBeGreaterThan(0);
    for (const a of ayahs) {
      const key = `${a.surah}:${a.ayah}`;
      expect(seen.has(key)).toBe(false);
      seen.add(key);
      expect(getPageForVerse(a.surah, a.ayah)).toBe(page);
    }
  }
  expect(seen.size).toBe(verses.length);
});

it("maps pages to their Juz", () => {
  expect(getJuzForPage(1)).toMatchObject({ number: 1, nameEn: "Juz 1", nameAr: "الجزء الأول" });
  expect(getJuzForPage(21)).toMatchObject({ number: 1, nameEn: "Juz 1" });
  expect(getJuzForPage(22)).toMatchObject({ number: 2, nameEn: "Juz 2" });
  expect(getJuzForPage(581)).toMatchObject({ number: 29, nameEn: "Juz 29" });
  expect(getJuzForPage(582)).toMatchObject({ number: 30, nameEn: "Juz 30", nameAr: "الجزء الثلاثون" });
  expect(getJuzForPage(604)).toMatchObject({ number: 30, nameEn: "Juz 30" });
});
