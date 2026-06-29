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
import type { QuranCandidate } from "../src/lib/quran-db";
import {
  SAMPLE_RATE,
  TRACKING_TRIGGER_SAMPLES,
} from "../src/lib/types";

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
const VERSE_4 = makeVerse(2, 4, [
  "waalladhiina", "yu'minuuna", "bimaa", "unzila", "ilayka",
]);
const VERSE_5 = makeVerse(2, 5, [
  "ulaaika", "alaa", "hudan", "min", "rabbihim",
]);
const RANDOM_VERSE = makeVerse(75, 19, ["thumma", "inna", "alaynaa", "bayanah"]);

// ---------------------------------------------------------------------------
// Mock QuranDB
// ---------------------------------------------------------------------------
function createMockDB() {
  const verses = [VERSE_1, VERSE_2, VERSE_3, VERSE_4, VERSE_5, RANDOM_VERSE];
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

function makeCandidate(
  verse: QuranVerse,
  score = 0.9,
  ayahEnd: number | null = null,
): QuranCandidate {
  const end = ayahEnd ?? verse.ayah;
  return {
    surah: verse.surah,
    ayah: verse.ayah,
    ayah_end: ayahEnd,
    text: verse.text_uthmani,
    phonemes_joined: verse.phonemes_joined,
    phoneme_token_ids: verse.phoneme_token_ids ?? [],
    stage_a_score: score,
    raw_score: score,
    bonus: 0,
    kind: end > verse.ayah ? "span" : "single",
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

function collectVerseCandidates(messages: WorkerOutbound[]) {
  return messages
    .filter((m) => m.type === "verse_candidate")
    .flatMap((m) => {
      if (m.type !== "verse_candidate") throw new Error("unreachable");
      return m.candidates.map((candidate) => `${candidate.surah}:${candidate.ayah}` +
        (candidate.ayah_end ? `-${candidate.ayah_end}` : ""));
    });
}

function collectWordProgress(messages: WorkerOutbound[]) {
  return messages
    .filter((m) => m.type === "word_progress")
    .map((m) => {
      if (m.type !== "word_progress") throw new Error("unreachable");
      return `${m.surah}:${m.ayah}:${m.word_index}/${m.total_words}`;
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
  it("uses championMatch as the authoritative discovery match", async () => {
    const db = createMockDB();
    db.matchVerse.mockReturnValue({
      surah: VERSE_1.surah,
      ayah: VERSE_1.ayah,
      ayah_end: null,
      text: VERSE_1.text_uthmani,
      phonemes_joined: VERSE_1.phonemes_joined,
      score: 0.99,
      raw_score: 0.99,
      bonus: 0,
    });

    const championMatch = {
      surah: VERSE_3.surah,
      ayah: VERSE_3.ayah,
      ayah_end: null,
      text: VERSE_3.text_uthmani,
      phonemes_joined: VERSE_3.phonemes_joined,
      score: 0.9,
      raw_score: 0.9,
      bonus: 0,
    };
    const transcribeFn = createTranscribeFn([
      {
        ...makeResult(VERSE_3.phonemes_joined),
        championMatch,
        championTranscript: VERSE_3.phonemes_joined,
      },
    ]);

    const tracker = new RecitationTracker(db, transcribeFn);
    const t = tracker as any;
    t.utteranceHasSpeech = true;
    t.utteranceAudio = makeSpeechChunk();

    const messages = await t._handleDiscovery(true);

    expect(db.matchVerse).not.toHaveBeenCalled();
    expect(collectVerseMatches(messages)).toEqual(["2:3"]);
  });

  it("live discovery span covering last emitted ayah rebases to next ayah", async () => {
    const db = createMockDB();
    const span = makeCandidate(VERSE_1, 0.91, VERSE_4.ayah);
    db.matchVerse.mockReturnValue({
      ...span,
      score: span.stage_a_score,
    });
    db.retrieveCandidates.mockReturnValue({ combined: [span] });

    const tracker = new RecitationTracker(
      db,
      createTranscribeFn([makeResult(`${VERSE_1.phonemes_joined} ${VERSE_2.phonemes_joined}`)]),
      { config: { discoveryRepeatCycles: 1 } },
    );
    const t = tracker as any;
    t.lastEmittedRef = [VERSE_1.surah, VERSE_1.ayah];
    t.utteranceHasSpeech = true;
    t.utteranceAudio = makeSpeechChunk(SAMPLE_RATE * 8);
    t.newAudioCount = SAMPLE_RATE * 3;

    const messages = await t._handleDiscovery(false);

    expect(collectVerseMatches(messages)).toEqual(["2:2"]);
    expect(collectVerseCandidates(messages)).toContain("2:2");
    expect(t.trackingVerse).toEqual(VERSE_2);
    expect(t.lastEmittedRef).toEqual([2, 2]);
  });

  it("live discovery forward spans do not rebase backward to the immediate next ayah", async () => {
    const db = createMockDB();
    const span = makeCandidate(VERSE_3, 0.91, VERSE_4.ayah);
    db.matchVerse.mockReturnValue({
      ...span,
      score: span.stage_a_score,
    });
    db.retrieveCandidates.mockReturnValue({ combined: [span] });

    const tracker = new RecitationTracker(
      db,
      createTranscribeFn([makeResult(`${VERSE_1.phonemes_joined} ${VERSE_2.phonemes_joined}`)]),
      { config: { discoveryRepeatCycles: 1 } },
    );
    const t = tracker as any;
    t.lastEmittedRef = [VERSE_1.surah, VERSE_1.ayah];
    t.utteranceHasSpeech = true;
    t.utteranceAudio = makeSpeechChunk(SAMPLE_RATE * 8);
    t.newAudioCount = SAMPLE_RATE * 3;

    const messages = await t._handleDiscovery(false);

    expect(collectVerseMatches(messages)).toEqual(["2:3"]);
    expect(collectVerseMatches(messages)).not.toContain("2:2");
    expect(t.trackingVerse).toEqual(VERSE_3);
    expect(t.lastEmittedRef).toEqual([2, 3]);
  });

  it("far-ahead same-surah spans are not committed as the next ayah", async () => {
    const db = createMockDB();
    const span = makeCandidate(VERSE_5, 0.95, null);
    db.matchVerse.mockReturnValue({
      ...span,
      score: span.stage_a_score,
    });
    db.retrieveCandidates.mockReturnValue({ combined: [span] });

    const tracker = new RecitationTracker(
      db,
      createTranscribeFn([makeResult(VERSE_5.phonemes_joined)]),
      { config: { discoveryRepeatCycles: 1 } },
    );
    const t = tracker as any;
    t.lastEmittedRef = [VERSE_1.surah, VERSE_1.ayah];
    t.utteranceHasSpeech = true;
    t.utteranceAudio = makeSpeechChunk(SAMPLE_RATE * 8);
    t.newAudioCount = SAMPLE_RATE * 3;

    const messages = await t._handleDiscovery(false);

    expect(collectVerseMatches(messages)).toEqual([]);
    expect(collectVerseMatches(messages)).not.toContain("2:2");
    expect(collectVerseMatches(messages)).not.toContain("2:5");
    expect(t.lastEmittedRef).toEqual([2, 1]);
  });

  it("broad stale champion span yields to a nearby stronger forward candidate", async () => {
    const db = createMockDB();
    const broadChampion = {
      ...makeCandidate(VERSE_1, 0.86, VERSE_5.ayah),
      score: 0.86,
      raw_score: 0.86,
    };
    const forward = makeCandidate(VERSE_4, 1, VERSE_5.ayah);
    db.retrieveCandidates.mockReturnValue({ combined: [forward] });

    const tracker = new RecitationTracker(
      db,
      createTranscribeFn([{
        ...makeResult(`${VERSE_4.phonemes_joined} ${VERSE_5.phonemes_joined}`),
        championMatch: broadChampion,
      }]),
      { config: { discoveryRepeatCycles: 1 } },
    );
    const t = tracker as any;
    t.lastEmittedRef = [VERSE_2.surah, VERSE_2.ayah];
    t.utteranceHasSpeech = true;
    t.utteranceAudio = makeSpeechChunk(SAMPLE_RATE * 8);
    t.newAudioCount = SAMPLE_RATE * 3;

    const messages = await t._handleDiscovery(false);

    expect(collectVerseMatches(messages)).toEqual(["2:4"]);
    expect(collectVerseMatches(messages)).not.toContain("2:3");
    expect(t.trackingVerse).toEqual(VERSE_4);
    expect(t.lastEmittedRef).toEqual([2, 4]);
  });

  it("blocks random live discovery jumps to another surah after context is established", async () => {
    const db = createMockDB();
    const random = makeCandidate(RANDOM_VERSE, 0.95, null);
    db.matchVerse.mockReturnValue({
      ...random,
      score: random.stage_a_score,
    });
    db.retrieveCandidates.mockReturnValue({ combined: [random] });

    const tracker = new RecitationTracker(
      db,
      createTranscribeFn([makeResult(RANDOM_VERSE.phonemes_joined)]),
      { config: { discoveryRepeatCycles: 1 } },
    );
    const t = tracker as any;
    t.lastEmittedRef = [VERSE_2.surah, VERSE_2.ayah];
    t.utteranceHasSpeech = true;
    t.utteranceAudio = makeSpeechChunk(SAMPLE_RATE * 4);
    t.newAudioCount = SAMPLE_RATE * 3;

    const messages = await t._handleDiscovery(false);

    expect(collectVerseMatches(messages)).toEqual([]);
    expect(t.lastEmittedRef).toEqual([2, 2]);
    expect(t.trackingVerse).toBeNull();
  });

  it("does not residual-skip a long transcript that continues past the last emitted verse", async () => {
    const db = createMockDB();
    const next = makeCandidate(VERSE_3, 0.9, null);
    db.matchVerse.mockReturnValue({
      ...next,
      score: next.stage_a_score,
    });
    db.retrieveCandidates.mockReturnValue({ combined: [next] });

    const tracker = new RecitationTracker(
      db,
      createTranscribeFn([
        makeResult(`${VERSE_2.phonemes_joined} ${VERSE_3.phonemes_joined}`),
      ]),
      { config: { discoveryRepeatCycles: 1 } },
    );
    const t = tracker as any;
    t.lastEmittedRef = [VERSE_2.surah, VERSE_2.ayah];
    t.lastEmittedText = VERSE_2.phonemes_joined;
    t.lastCommitEvidence = { confidence: 0.99, acousticMargin: 1, strong: true };
    t.utteranceHasSpeech = true;
    t.utteranceAudio = makeSpeechChunk(SAMPLE_RATE * 8);
    t.newAudioCount = SAMPLE_RATE * 3;

    const messages = await t._handleDiscovery(false);

    expect(collectVerseMatches(messages)).toEqual(["2:3"]);
    expect(t.trackingVerse).toEqual(VERSE_3);
    expect(t.lastEmittedRef).toEqual([2, 3]);
  });

  it("word progress can advance from a weak repeat-leader commit", async () => {
    const transcribeFn = createTranscribeFn([
      makeResult(VERSE_2.phoneme_words.slice(0, -1).join(" ")),
    ]);

    const db = createMockDB();
    const tracker = new RecitationTracker(db, transcribeFn);
    injectTrackingState(tracker, VERSE_2);
    const t = tracker as any;
    t.lastCommitEvidence = { confidence: 0.88, acousticMargin: 0.02, strong: false };

    for (let i = 0; i < 2; i++) {
      await tracker.feed(makeSpeechChunk());
    }

    expect(t.trackingVerse).toEqual(VERSE_3);
    expect(t.trackingPendingEmission).toBe(true);
    expect(t.pendingEmissionMessage?.surah).toBe(VERSE_3.surah);
    expect(t.pendingEmissionMessage?.ayah).toBe(VERSE_3.ayah);
  });

  it("completion coverage arms the next verse as pending without committing it", async () => {
    const almostComplete = VERSE_2.phoneme_words.slice(0, -1).join(" ");
    const transcribeFn = createTranscribeFn([
      makeResult(almostComplete),
    ]);

    const db = createMockDB();
    const tracker = new RecitationTracker(db, transcribeFn);
    injectTrackingState(tracker, VERSE_2);

    const messages = await tracker.feed(makeSpeechChunk());

    const t = tracker as any;
    expect(collectVerseMatches(messages)).toEqual([]);
    expect(t.trackingVerse).toEqual(VERSE_3);
    expect(t.trackingPendingEmission).toBe(true);
    expect(t.pendingEmissionMessage?.surah).toBe(VERSE_3.surah);
    expect(t.pendingEmissionMessage?.ayah).toBe(VERSE_3.ayah);
  });

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

  it("does not confirm pending verse from a single non-prefix word match", async () => {
    const transcribeFn = createTranscribeFn([
      makeResult("alif laam miim"), // VERSE_1 complete → auto-advance
      makeResult("alkitaabu"), // VERSE_2 word 2 only; not enough prefix evidence
    ]);

    const db = createMockDB();
    const tracker = new RecitationTracker(db, transcribeFn);
    injectTrackingState(tracker, VERSE_1);

    await tracker.feed(makeSpeechChunk());
    const messages = await tracker.feed(makeSpeechChunk());

    expect(collectVerseMatches(messages)).not.toContain("2:2");
    expect((tracker as any).trackingVerse).toEqual(VERSE_2);
    expect((tracker as any).trackingPendingEmission).toBe(true);
  });

  it("does not emit word progress for an unconfirmed pending verse", async () => {
    const transcribeFn = createTranscribeFn([
      makeResult("alif laam miim"), // VERSE_1 complete → auto-advance
      makeResult("dhaalika"), // VERSE_2 starts, but not enough evidence to confirm
    ]);

    const db = createMockDB();
    const tracker = new RecitationTracker(db, transcribeFn);
    injectTrackingState(tracker, VERSE_1);

    await tracker.feed(makeSpeechChunk());
    const messages = await tracker.feed(makeSpeechChunk());

    expect(collectVerseMatches(messages)).not.toContain("2:2");
    expect(collectWordProgress(messages)).toEqual([]);
    expect((tracker as any).trackingVerse).toEqual(VERSE_2);
    expect((tracker as any).trackingPendingEmission).toBe(true);
  });

  it("confirmed pending verse can arm the next verse in the same tracking cycle", async () => {
    const transcribeFn = createTranscribeFn([
      makeResult("alif laam miim"), // VERSE_1 complete → auto-advance
      makeResult(VERSE_2.phoneme_words.join(" ")), // VERSE_2 complete in fresh audio
    ]);

    const db = createMockDB();
    const tracker = new RecitationTracker(db, transcribeFn);
    injectTrackingState(tracker, VERSE_1);

    const first = await tracker.feed(makeSpeechChunk());
    expect(collectVerseMatches(first)).toEqual([]);
    expect((tracker as any).trackingVerse).toEqual(VERSE_2);
    expect((tracker as any).trackingPendingEmission).toBe(true);

    const second = await tracker.feed(makeSpeechChunk());

    expect(collectVerseMatches(second)).toContain("2:2");
    expect((tracker as any).trackingVerse).toEqual(VERSE_3);
    expect((tracker as any).trackingPendingEmission).toBe(true);
    expect((tracker as any).pendingEmissionMessage?.surah).toBe(VERSE_3.surah);
    expect((tracker as any).pendingEmissionMessage?.ayah).toBe(VERSE_3.ayah);
  });

  it("confirmed pending verse does not cascade before its final word", async () => {
    const transcribeFn = createTranscribeFn([
      makeResult("alif laam miim"), // VERSE_1 complete → auto-advance
      makeResult(VERSE_2.phoneme_words.slice(0, -1).join(" ")), // complete enough, not final
    ]);

    const db = createMockDB();
    const tracker = new RecitationTracker(db, transcribeFn);
    injectTrackingState(tracker, VERSE_1);

    await tracker.feed(makeSpeechChunk());
    const messages = await tracker.feed(makeSpeechChunk());

    expect(collectVerseMatches(messages)).toContain("2:2");
    expect((tracker as any).trackingVerse).toEqual(VERSE_2);
    expect((tracker as any).trackingPendingEmission).toBe(false);
    expect((tracker as any).pendingEmissionMessage).toBeNull();
  });

  it("completion coverage confirms pending current verse without arming next from late-word matches", async () => {
    const transcribeFn = createTranscribeFn([
      makeResult("alif laam miim"), // VERSE_1 complete → auto-advance
      makeResult("hum"), // VERSE_2 late word only; enough to reach completion coverage
    ]);

    const db = createMockDB();
    const tracker = new RecitationTracker(db, transcribeFn);
    injectTrackingState(tracker, VERSE_1);

    await tracker.feed(makeSpeechChunk());
    const t = tracker as any;
    expect(t.trackingVerse).toEqual(VERSE_2);
    expect(t.trackingPendingEmission).toBe(true);

    // Simulate prior progress through the pending verse. The next decode only
    // matches a late word, so it should confirm VERSE_2 but not cascade to VERSE_3.
    t.trackingLastWordIdx = 7;
    t.trackingProgressEstablished = true;

    const messages = await tracker.feed(makeSpeechChunk());

    expect(collectVerseMatches(messages)).toContain("2:2");
    expect(t.trackingVerse).toEqual(VERSE_2);
    expect(t.trackingPendingEmission).toBe(false);
    expect(t.pendingEmissionMessage).toBeNull();
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

    // Complete VERSE_1 → auto-advance (deferred), then stop before fresh
    // next-verse audio can confirm or reject the pending emission.
    for (let i = 0; i < 5; i++) {
      const msgs = await tracker.feed(makeSpeechChunk());
      allMessages.push(...msgs);
      if ((tracker as any).trackingPendingEmission) break;
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
      if ((tracker as any).trackingPendingEmission) break;
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

    // Complete VERSE_1 → auto-advance (deferred), then stop before fresh
    // next-verse audio can confirm or reject the pending emission.
    for (let i = 0; i < 5; i++) {
      const msgs = await tracker.feed(makeSpeechChunk());
      allMessages.push(...msgs);
      if ((tracker as any).trackingPendingEmission) break;
    }

    // Feed with non-matching text — no primary word alignment
    for (let i = 0; i < 15; i++) {
      const msgs = await tracker.feed(makeSpeechChunk());
      allMessages.push(...msgs);
    }

    const verseMatches = collectVerseMatches(allMessages);
    expect(verseMatches).not.toContain("2:2");
  });

  it("final flush emits pending next-verse when advance margin is strict", async () => {
    // Reproduces the multi_114 / user_ikhlas_2_3 "last verse dropped" pattern.
    // After auto-advance with strong acoustic evidence, simulate the stream
    // ending (final flush) before any fresh audio can confirm the pending
    // emission. With the silence-flush fix, the next verse should still emit.
    const transcribeFn = createTranscribeFn([
      makeResult("alif laam miim"), // VERSE_1 complete → auto-advance
      makeResult(UNRELATED_TEXT), // no match on fresh audio
    ]);

    const db = createMockDB();
    const tracker = new RecitationTracker(db, transcribeFn);
    injectTrackingState(tracker, VERSE_1);
    const allMessages: WorkerOutbound[] = [];

    // Complete VERSE_1 → auto-advance (deferred)
    for (let i = 0; i < 5; i++) {
      const msgs = await tracker.feed(makeSpeechChunk());
      allMessages.push(...msgs);
      if ((tracker as any).trackingPendingEmission) break;
    }

    // Simulate strong advance margin (as if the acoustic gate had seen
    // next-verse content strongly in the tail audio).
    const t = tracker as any;
    expect(t.trackingPendingEmission).toBe(true);
    t.pendingEmissionMargin = -1.0; // well below ADVANCE_FLUSH_STRICT_MARGIN (0.5)

    // Trigger final flush via extended silence.
    for (let i = 0; i < 30; i++) {
      const msgs = await tracker.feed(makeSilenceChunk());
      allMessages.push(...msgs);
    }

    const verseMatches = collectVerseMatches(allMessages);
    expect(verseMatches).toContain("2:2"); // pending emission DID fire on flush
  });

  it("final flush does NOT emit pending when advance margin is loose", async () => {
    const transcribeFn = createTranscribeFn([
      makeResult("alif laam miim"),
      makeResult(UNRELATED_TEXT),
    ]);

    const db = createMockDB();
    const tracker = new RecitationTracker(db, transcribeFn);
    injectTrackingState(tracker, VERSE_1);
    const allMessages: WorkerOutbound[] = [];

    for (let i = 0; i < 5; i++) {
      const msgs = await tracker.feed(makeSpeechChunk());
      allMessages.push(...msgs);
    }

    const t = tracker as any;
    // Loose margin: just barely passed the normal advance gate (3.0) but far
    // from the strict flush gate (0.5).
    t.pendingEmissionMargin = 2.0;

    for (let i = 0; i < 30; i++) {
      const msgs = await tracker.feed(makeSilenceChunk());
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
