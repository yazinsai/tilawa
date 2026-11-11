import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { displayIndex, indexQuran, splitDisplayWords, surahList, type QuranVerse } from "../src/lib/quran-words";

const verses = JSON.parse(readFileSync(new URL("../public/quran.json", import.meta.url), "utf8")) as QuranVerse[];
const quran = indexQuran(verses);

it("attaches pause marks to the word before them", () => {
  expect(splitDisplayWords("ذَٰلِكَ ٱلْكِتَٰبُ لَا رَيْبَ ۛ فِيهِ ۛ", 2, 2)).toEqual(["ذَٰلِكَ", "ٱلْكِتَٰبُ", "لَا", "رَيْبَ ۛ", "فِيهِ ۛ"]);
});

it("drops the basmala quran.json prefixes to first ayahs, except in Al-Fatiha", () => {
  expect(quran.get("2:1")!.words).toEqual(["الٓمٓ"]);
  expect(quran.get("1:1")!.words).toHaveLength(4);
  const raw = (s: number, a: number) => verses.find((v) => v.surah === s && v.ayah === a)!.text_uthmani.split(" ");
  expect(quran.get("9:1")!.words[0]).toBe(raw(9, 1)[0]);
  expect(quran.get("36:1")!.words).toEqual([raw(36, 1)[4]]);
});

it("drops the rub el hizb sign", () => {
  const raw = verses.find((v) => v.surah === 2 && v.ayah === 26)!.text_uthmani.split(" ");
  expect(raw[0]).toBe("\u06DE");
  expect(quran.get("2:26")!.words[0]).toBe(raw[1]);
});

it("maps reference words onto display words", () => {
  expect(displayIndex(112, 1, 1)).toBe(0);
  // 2:181 word 3+4 (بَعْدَ مَا) are one display word.
  expect(displayIndex(2, 181, 3)).toBe(2);
  expect(displayIndex(2, 181, 4)).toBe(2);
  expect(displayIndex(2, 181, 5)).toBe(3);
  expect(quran.get("2:181")!.words[displayIndex(2, 181, 14)]).toBe("عَلِيمٌۭ");
});

it("lists 114 surahs with their ayah counts", () => {
  const list = surahList(quran);
  expect(list).toHaveLength(114);
  expect(list[0]).toMatchObject({ surah: 1, ayahs: 7 });
  expect(list[1].ayahs).toBe(286);
});
