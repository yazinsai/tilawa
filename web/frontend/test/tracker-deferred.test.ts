/**
 * Deterministic unit tests for deferred emission in RecitationTracker.
 *
 * These tests mock the transcribe function and QuranDB, and inject
 * internal state to isolate the deferred emission logic.
 * No ONNX model dependency.
 */

import { describe, it, expect, vi } from "vitest";
import { RecitationTracker } from "../src/lib/tracker";
import type { TranscribeResult } from "../src/lib/tracker";
import type { QuranVerse, WorkerOutbound } from "../src/lib/types";
import { SAMPLE_RATE, TRACKING_TRIGGER_SAMPLES } from "../src/lib/types";

// ---------------------------------------------------------------------------
// Mock verse data
// ---------------------------------------------------------------------------
function makeVerse(surah: number, ayah: number, words: string[]): QuranVerse {
  const phonemes_joined = words.join(" ");
  return {
    surah,
    ayah,
    text_uthmani: `Verse ${surah}:${ayah}`,
    surah_name: `Surah ${surah}`,
    surah_name_en: `Surah ${surah}`,
    phonemes: phonemes_joined,
    phonemes_joined,
    phonemes_joined_ns: phonemes_joined.replace(/ /g, ""),
    phoneme_words: words,
    phoneme_tokens: words,
    phoneme_token_ids: words.map((_, i) => i + 1),
    word_token_ends: words.map((_, i) => i + 1),
    phonemes_joined_no_bsm: null,
    phonemes_joined_no_bsm_ns: null,
    phoneme_token_ids_no_bsm: null,
    phoneme_tokens_no_bsm: null,
  };
}

const VERSE_1 = makeVerse(2, 1, ["alif", "laam", "miim"]);
// Use long distinct words to avoid accidental fuzzy matches
const VERSE_2 = makeVerse(2, 2, [
  "dhaalika", "alkitaabu", "laaa", "rayba", "fiihi",
  "hudan", "lilmuttaqiina", "wabilaakhirati", "hum", "yuuqinuuna",
]);
const VERSE_3 = makeVerse(2, 3, [
  "alladhiina", "yu'minuuna", "bilghaybi", "wayuqiimuuna", "aSSalaata",
]);

// ---------------------------------------------------------------------------
// Mock QuranDB
// ---------------------------------------------------------------------------
function createMockDB() {
  const verses = [VERSE_1, VERSE_2, VERSE_3];
  const verseMap = new Map(verses.map((v) => [`${v.surah}:${v.ayah}`, v]));

  return {
    totalVerses: verses.length,
    surahCount: 1,
    getVerse: (surah: number, ayah: number) => verseMap.get(`${surah}:${ayah}`) ?? null,
    getNextVerse: (surah: number, ayah: number) => verseMap.get(`${surah}:${ayah + 1}`) ?? null,
    getSurah: (surah: number) => verses.filter((v) => v.surah === surah),
    matchVerse: vi.fn().mockReturnValue(null),
    retrieveCandidates: vi.fn().mockReturnValue({ combined: [] }),
    getShortVerseCandidates: vi.fn().mockReturnValue([]),
  } as any;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * 0.3); // 300ms = 4800 samples

function makeSpeechChunk(length = CHUNK_SAMPLES): Float32Array {
  const audio = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    audio[i] = 0.05 * Math.sin((2 * Math.PI * 440 * i) / SAMPLE_RATE);
  }
  return audio;
}

function makeSilenceChunk(length = CHUNK_SAMPLES): Float32Array {
  return new Float32Array(length);
}

function createTranscribeFn(results: TranscribeResult[]) {
  let callIndex = 0;
  return vi.fn(async (_audio: Float32Array): Promise<TranscribeResult> => {
    const idx = Math.min(callIndex, results.length - 1);
    callIndex++;
    return results[idx];
  });
}

function makeResult(text: string): TranscribeResult {
  return {
    text,
    rawPhonemes: text,
    tokenIds: text
      .split(" ")
      .filter(Boolean)
      .map((_, i) => i + 1),
  };
}

function collectVerseMatches(messages: WorkerOutbound[]) {
  return messages
    .filter((m) => m.type === "verse_match")
    .map((m) => {
      if (m.type !== "verse_match") throw new Error("unreachable");
      return `${m.surah}:${m.ayah}`;
    });
}

// Words that won't fuzzy-match any VERSE_2 word (all ratios < 0.7)
const UNRELATED_TEXT = "xyzzy qwerty plugh";

/**
 * Inject tracker into tracking mode for a verse with a strong commit.
 * Bypasses discovery to isolate the deferred emission logic.
 */
function injectTrackingState(tracker: RecitationTracker, verse: QuranVerse): void {
  const t = tracker as any;
  t.trackingVerse = verse;
  t.trackingVerseWords = verse.phoneme_words;
  t.trackingLastWordIdx = -1;
  t.trackingProgressEstablished = false;
  t.staleCycles = 0;
  t.lastEmittedRef = [verse.surah, verse.ayah];
  t.lastEmittedText = verse.phonemes_joined;
  t.lastCommitEvidence = { confidence: 0.85, acousticMargin: 0.3, strong: true };
  t.utteranceHasSpeech = true;
  t.consecutiveAutoAdvances = 0;
  t.cyclesSinceCommit = 0;

  const tokenIds = verse.phoneme_token_ids ?? [];
  const wordEnds = verse.word_token_ends ?? [];
  t.trackingPrefixes = wordEnds
    .map((end: number, idx: number) => ({
      wordIndex: idx,
      ids: tokenIds.slice(0, end),
    }))
    .filter((prefix: any) => prefix.ids.length > 0);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe("Deferred emission", () => {
  it("stale pending verse drops silently (no verse_match emitted)", async () => {
    const transcribeFn = createTranscribeFn([
      makeResult("alif laam miim"), // VERSE_1 complete → auto-advance
      makeResult(UNRELATED_TEXT), // VERSE_2 tracking: no match → stale
    ]);

    const db = createMockDB();
    const tracker = new RecitationTracker(db, transcribeFn);
    injectTrackingState(tracker, VERSE_1);
    const allMessages: WorkerOutbound[] = [];

    // Complete VERSE_1 → auto-advance to VERSE_2 (deferred)
    for (let i = 0; i < 5; i++) {
      const msgs = await tracker.feed(makeSpeechChunk());
      allMessages.push(...msgs);
    }

    // Feed more speech — stale cycles accumulate, exit tracking, rollback
    for (let i = 0; i < 15; i++) {
      const msgs = await tracker.feed(makeSpeechChunk());
      allMessages.push(...msgs);
    }

    const verseMatches = collectVerseMatches(allMessages);
    expect(verseMatches).not.toContain("2:2");
  });

  it("valid multi-verse continuation emits after primary word alignment confirms", async () => {
    const transcribeFn = createTranscribeFn([
      makeResult("alif laam miim"), // VERSE_1 complete → auto-advance
      makeResult("dhaalika alkitaabu laaa"), // VERSE_2: primary word alignment matches
    ]);

    const db = createMockDB();
    const tracker = new RecitationTracker(db, transcribeFn);
    injectTrackingState(tracker, VERSE_1);
    const allMessages: WorkerOutbound[] = [];

    // Complete VERSE_1 → auto-advance to VERSE_2 (deferred)
    for (let i = 0; i < 5; i++) {
      const msgs = await tracker.feed(makeSpeechChunk());
      allMessages.push(...msgs);
    }

    // Feed fresh audio → VERSE_2 words align → pending emission confirmed
    for (let i = 0; i < 5; i++) {
      const msgs = await tracker.feed(makeSpeechChunk());
      allMessages.push(...msgs);
    }

    const verseMatches = collectVerseMatches(allMessages);
    expect(verseMatches).toContain("2:2");
  });

  it("end-of-stream with pending emission does not leak a verse", async () => {
    const transcribeFn = createTranscribeFn([
      makeResult("alif laam miim"), // VERSE_1 complete → auto-advance
      makeResult(UNRELATED_TEXT), // silence period: unrelated
    ]);

    const db = createMockDB();
    const tracker = new RecitationTracker(db, transcribeFn);
    injectTrackingState(tracker, VERSE_1);
    const allMessages: WorkerOutbound[] = [];

    // Complete VERSE_1 → auto-advance (deferred)
    for (let i = 0; i < 5; i++) {
      const msgs = await tracker.feed(makeSpeechChunk());
      allMessages.push(...msgs);
    }

    // Extended silence → tracking timeout → rollback
    for (let i = 0; i < 30; i++) {
      const msgs = await tracker.feed(makeSilenceChunk());
      allMessages.push(...msgs);
    }

    const verseMatches = collectVerseMatches(allMessages);
    expect(verseMatches).not.toContain("2:2");
  });

  it("state rollback on drop restores all fields correctly", async () => {
    const transcribeFn = createTranscribeFn([
      makeResult("alif laam miim"), // VERSE_1 complete → auto-advance
      makeResult(UNRELATED_TEXT), // stale
    ]);

    const db = createMockDB();
    const tracker = new RecitationTracker(db, transcribeFn);
    injectTrackingState(tracker, VERSE_1);
    const allMessages: WorkerOutbound[] = [];

    // Complete VERSE_1 → auto-advance to VERSE_2 (deferred)
    for (let i = 0; i < 5; i++) {
      const msgs = await tracker.feed(makeSpeechChunk());
      allMessages.push(...msgs);
    }

    // Stale tracking → exit + rollback
    for (let i = 0; i < 15; i++) {
      const msgs = await tracker.feed(makeSpeechChunk());
      allMessages.push(...msgs);
    }

    // Verify rollback restored state
    const t = tracker as any;
    expect(t.lastEmittedRef).toEqual([2, 1]); // VERSE_1, not VERSE_2
    expect(t.trackingPendingEmission).toBe(false);
    expect(t.pendingEmissionMessage).toBeNull();
    expect(t.preAdvanceSnapshot).toBeNull();
    expect(t.consecutiveAutoAdvances).toBe(0);

    const verseMatches = collectVerseMatches(allMessages);
    expect(verseMatches).not.toContain("2:2");
  });

  it("acoustic/char-level fallback do NOT trigger pending emission", async () => {
    const transcribeFn = createTranscribeFn([
      makeResult("alif laam miim"), // VERSE_1 complete → auto-advance
      // Single short word — won't match any VERSE_2 word via primary alignment
      // (ratio < 0.7 for all). Could potentially match via char-level but
      // VERSE_2 has < 10 words so char-level is disabled.
      makeResult("xyz"),
    ]);

    const db = createMockDB();
    const tracker = new RecitationTracker(db, transcribeFn);
    injectTrackingState(tracker, VERSE_1);
    const allMessages: WorkerOutbound[] = [];

    // Complete VERSE_1 → auto-advance (deferred)
    for (let i = 0; i < 5; i++) {
      const msgs = await tracker.feed(makeSpeechChunk());
      allMessages.push(...msgs);
    }

    // Feed with non-matching text — no primary word alignment
    for (let i = 0; i < 15; i++) {
      const msgs = await tracker.feed(makeSpeechChunk());
      allMessages.push(...msgs);
    }

    const verseMatches = collectVerseMatches(allMessages);
    expect(verseMatches).not.toContain("2:2");
  });

  it("audio buffer retains 0.5s on auto-advance, not full 2s", async () => {
    const transcribeFn = createTranscribeFn([
      makeResult("alif laam miim"),
      makeResult(UNRELATED_TEXT),
    ]);

    const db = createMockDB();
    const tracker = new RecitationTracker(db, transcribeFn);

    // Pre-fill with 5 seconds of audio
    const t = tracker as any;
    t.utteranceAudio = makeSpeechChunk(SAMPLE_RATE * 5);
    t.utteranceHasSpeech = true;
    injectTrackingState(tracker, VERSE_1);

    // Complete VERSE_1 → auto-advance → _retainTailAfterCommit with short retention
    for (let i = 0; i < 5; i++) {
      await tracker.feed(makeSpeechChunk());
    }

    // After auto-advance with pending emission, buffer should be ~0.5s
    expect(t.utteranceAudio.length).toBeLessThanOrEqual(
      TRACKING_TRIGGER_SAMPLES + CHUNK_SAMPLES * 5,
    );
  });
});
