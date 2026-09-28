import { describe, expect, it } from "vitest";
import { QuranDB } from "../src/quran-db";
import { RecitationTracker, type TranscribeResult } from "../src/tracker";
import type { NextVerseEmitMode, QuranVerse, VerseMatchMessage, WorkerOutbound } from "../src/types";

const words = [
  "alpha bravo charlie delta",
  "echo foxtrot golf hotel",
  "india juliet kilo lima",
];

function verse(ayah: number): QuranVerse {
  const text = words[ayah - 1];
  return {
    surah: 1,
    ayah,
    text_uthmani: text,
    surah_name: "Test",
    surah_name_en: "Test",
    phonemes: text,
    phonemes_joined: text,
    phoneme_words: text.split(" "),
    phoneme_token_ids: [ayah],
  };
}

function setup(results: Array<string | TranscribeResult>, nextVerseEmitMode: NextVerseEmitMode = "deferred_confirm") {
  const db = new QuranDB([verse(1), verse(2), verse(3)]);
  let read = 0;
  const transcribe = async (): Promise<TranscribeResult> => {
    const result = results[read++] ?? "";
    return typeof result === "string" ? { text: result, rawPhonemes: result } : result;
  };
  const tracker = new RecitationTracker(db, transcribe, {
    config: {
      discoveryTriggerSec: 0.5,
      discoveryRepeatCycles: 1,
      firstMatchThreshold: 0.5,
      nextVerseEmitMode,
    },
  });
  return tracker;
}

const speech = new Float32Array(32000).fill(0.2);
const silence = new Float32Array(32000);
const nextVerseAcoustic = {
  logprobs: new Float32Array([
    0, -5, -1,
    -2, -5, 0,
    0, -5, -1,
  ]),
  timeSteps: 3,
  vocabSize: 3,
  blankId: 0,
};

function matches(events: WorkerOutbound[]): number[] {
  return events
    .filter((event): event is VerseMatchMessage => event.type === "verse_match")
    .map((event) => event.ayah);
}

describe("FastConformer tracker state", () => {
  it("emits an advanced verse once after fresh audio confirms it", async () => {
    const tracker = setup([words[0], words[0], words[1]]);
    const first = await tracker.feed(speech);
    const advanced = await tracker.feed(speech);
    const confirmed = await tracker.feed(speech);

    expect(matches(first)).toEqual([1]);
    expect(matches(advanced)).toEqual([]);
    expect(matches(confirmed)).toEqual([2]);
  });

  it("does not emit an unconfirmed verse on final silence", async () => {
    const tracker = setup([words[0], words[0], ""]);
    await tracker.feed(speech);
    const advanced = await tracker.feed(speech);
    const flushed = await tracker.feed(silence);

    expect(matches(advanced)).toEqual([]);
    expect(matches(flushed)).toEqual([]);
  });

  it("emits an advanced verse on final silence when acoustic evidence is strong", async () => {
    const tracker = setup([
      words[0],
      { text: words[0], rawPhonemes: words[0], acoustic: nextVerseAcoustic },
      "",
    ]);
    await tracker.feed(speech);
    const advanced = await tracker.feed(speech);
    const flushed = await tracker.feed(silence);

    expect(matches(advanced)).toEqual([]);
    expect(matches(flushed)).toEqual([2]);
  });

  it("keeps immediate emission for the immediate mode", async () => {
    const tracker = setup([words[0], words[0]], "immediate_on_completion");
    await tracker.feed(speech);
    const advanced = await tracker.feed(speech);

    expect(matches(advanced)).toEqual([2]);
  });

  it("shows a candidate before confirmation in candidate mode", async () => {
    const tracker = setup([words[0], words[0], words[1]], "candidate_until_confirmed");
    await tracker.feed(speech);
    const advanced = await tracker.feed(speech);
    const confirmed = await tracker.feed(speech);

    expect(matches(advanced)).toEqual([]);
    expect(advanced.some((event) => event.type === "verse_candidate" &&
      event.candidates[0]?.ayah === 2)).toBe(true);
    expect(matches(confirmed)).toEqual([2]);
  });
});
