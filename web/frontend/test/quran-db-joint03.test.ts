import { describe, expect, it } from "vitest";
import { QuranDB } from "../src/lib/quran-db";
import type { QuranVerse } from "../src/lib/types";

function verse(surah: number, ayah: number, phonemes: string): QuranVerse {
  return {
    surah,
    ayah,
    text_uthmani: `${surah}:${ayah}`,
    surah_name: `s${surah}`,
    surah_name_en: `s${surah}`,
    phonemes,
    phonemes_joined: phonemes,
    phoneme_words: phonemes.split(/\s+/),
  };
}

describe("QuranDB joint03 champion matcher", () => {
  it("preserves precomputed CTC token ids from the text adapter", () => {
    const db = new QuranDB([
      {
        ...verse(2, 1, "الم"),
        phoneme_tokens: ["▁الم"],
        phoneme_token_ids: [101, 102, 103],
        word_token_ends: [3],
      },
    ]);

    expect(db.getVerse(2, 1)?.phoneme_token_ids).toEqual([101, 102, 103]);
    expect(db.getVerse(2, 1)?.word_token_ends).toEqual([3]);
  });

  it("keeps single-ayah joint02 matches shaped like the Python champion", () => {
    const db = new QuranDB([
      verse(1, 1, "bismi allahi arraHmaani arraHiimi"),
      verse(2, 1, "qaala rabbuka lil malaaikati innii jaailun fii alardi"),
    ]);

    const [match] = db.matchPhonemeTextJoint03("qaala rabbuka lil malaaikati innii");

    expect(match).toMatchObject({ surah: 2, ayah: 1, ayah_end: null });
  });

  it("collapses a late same-surah span to the surah prefix rescue", () => {
    const db = new QuranDB([
      verse(10, 1, "alif laam raa tilka ayaatu alkitaabi alhakiimi"),
      verse(10, 2, "akaana lilnnaasi ajaban an awHaynaa ilaa rajulin minhum"),
      verse(10, 3, "an andhir alnnasa wabashshiri alladhina aamanuu"),
    ]);
    const lateSpan =
      "akaana lilnnaasi ajaban an awHaynaa ilaa rajulin minhum " +
      "an andhir alnnasa wabashshiri alladhina aamanuu";

    const match = db.bestJoint03Match(lateSpan);

    expect(match).toMatchObject({
      surah: 10,
      ayah: 1,
      ayah_end: 3,
      _prefix_rescue: true,
    });
  });

  it("keeps a short opening ayah when the transcript continues into the next ayah", () => {
    const db = new QuranDB([
      verse(2, 1, "بسم الله الرحمن الرحيم الم"),
      verse(2, 2, "ذلك الكتاب لا ريب فيه هدى للمتقين"),
      verse(114, 2, "ملك الناس"),
      verse(114, 3, "اله الناس"),
    ]);

    const match = db.bestJoint03Match("الم ذلك الكتاب لا");

    expect(match).toMatchObject({
      surah: 2,
      ayah: 1,
      ayah_end: 2,
    });
  });
});
