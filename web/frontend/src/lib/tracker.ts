import { ratio as levRatio } from "./levenshtein";
import { QuranDB, partialRatio } from "./quran-db";
import { computeCorrection } from "./correction";
import type { QuranVerse, WorkerOutbound, SurroundingVerse } from "./types";
import {
  SAMPLE_RATE,
  TRIGGER_SAMPLES,
  MAX_WINDOW_SAMPLES,
  SILENCE_RMS_THRESHOLD,
  VERSE_MATCH_THRESHOLD,
  FIRST_MATCH_THRESHOLD,
  RAW_TRANSCRIPT_THRESHOLD,
  SURROUNDING_CONTEXT,
  TRACKING_TRIGGER_SAMPLES,
  TRACKING_SILENCE_SAMPLES,
  TRACKING_MAX_WINDOW_SAMPLES,
  STALE_CYCLE_LIMIT,
  LOOKAHEAD,
} from "./types";

export interface TranscribeResult {
  text: string;
  rawPhonemes: string;
}

type TranscribeFn = (audio: Float32Array) => Promise<TranscribeResult>;

function concatFloat32(a: Float32Array, b: Float32Array): Float32Array {
  const result = new Float32Array(a.length + b.length);
  result.set(a);
  result.set(b, a.length);
  return result;
}

function isSilence(audio: Float32Array): boolean {
  let sumSq = 0;
  for (let i = 0; i < audio.length; i++) {
    sumSq += audio[i] * audio[i];
  }
  const rms = Math.sqrt(sumSq / audio.length);
  return rms < SILENCE_RMS_THRESHOLD;
}

function wordsMatch(w1: string, w2: string, threshold = 0.7): boolean {
  if (w1 === w2) return true;
  if (w1.length <= 2 || w2.length <= 2) return w1 === w2;
  return levRatio(w1, w2) >= threshold;
}

function alignPosition(
  recognizedWords: string[],
  verseWords: string[],
  startFrom = 0,
): { position: number; matchedIndices: number[] } {
  if (!recognizedWords.length || !verseWords.length) {
    return { position: 0, matchedIndices: [] };
  }

  const matchedIndices: number[] = [];
  let versePtr = startFrom;

  for (const rec of recognizedWords) {
    if (versePtr >= verseWords.length) break;
    const limit = Math.min(versePtr + LOOKAHEAD, verseWords.length);
    for (let j = versePtr; j < limit; j++) {
      if (wordsMatch(rec, verseWords[j])) {
        matchedIndices.push(j);
        versePtr = j + 1;
        break;
      }
    }
  }

  if (matchedIndices.length) {
    return {
      position: matchedIndices[matchedIndices.length - 1] + 1,
      matchedIndices,
    };
  }
  return { position: startFrom, matchedIndices: [] };
}

function getSurroundingVerses(
  db: QuranDB,
  surah: number,
  ayah: number,
): SurroundingVerse[] {
  const verses = db.getSurah(surah);
  const result: SurroundingVerse[] = [];
  for (const v of verses) {
    if (Math.abs(v.ayah - ayah) <= SURROUNDING_CONTEXT) {
      result.push({
        surah: v.surah,
        ayah: v.ayah,
        text: v.text_uthmani,
        is_current: v.ayah === ayah,
      });
    }
  }
  return result;
}

export class RecitationTracker {
  private fullAudio = new Float32Array(0);
  private newAudioCount = 0;
  private lastEmittedRef: [number, number] | null = null;
  private lastEmittedText = "";
  private prevEmittedRef: [number, number] | null = null;
  private prevEmittedText = "";
  private hasEverMatched = false;
  private cyclesSinceEmit = Infinity; // anti-cascade: counts discovery cycles since last emit

  // Tracking mode state
  private trackingVerse: QuranVerse | null = null;
  private trackingVerseWords: string[] = [];
  private trackingLastWordIdx = -1;
  private silenceSamples = 0;
  private staleCycles = 0;

  constructor(
    private db: QuranDB,
    private transcribe: TranscribeFn,
  ) {}

  async feed(samples: Float32Array): Promise<WorkerOutbound[]> {
    const messages: WorkerOutbound[] = [];

    // Append audio
    this.fullAudio = concatFloat32(this.fullAudio, samples);
    this.newAudioCount += samples.length;

    // Trim to max window
    const maxSamples =
      this.trackingVerse !== null
        ? TRACKING_MAX_WINDOW_SAMPLES
        : MAX_WINDOW_SAMPLES;
    if (this.fullAudio.length > maxSamples) {
      this.fullAudio = this.fullAudio.slice(-maxSamples);
    }

    // TRACKING MODE
    if (this.trackingVerse !== null) {
      const trackMsgs = await this._handleTracking(samples);
      messages.push(...trackMsgs);
      return messages;
    }

    // DISCOVERY MODE
    const discMsgs = await this._handleDiscovery();
    messages.push(...discMsgs);
    return messages;
  }

  private async _handleTracking(
    samples: Float32Array,
  ): Promise<WorkerOutbound[]> {
    const messages: WorkerOutbound[] = [];

    // Check silence accumulation
    let sumSq = 0;
    for (let i = 0; i < samples.length; i++) {
      sumSq += samples[i] * samples[i];
    }
    const chunkRms = Math.sqrt(sumSq / samples.length);

    if (chunkRms < SILENCE_RMS_THRESHOLD) {
      this.silenceSamples += samples.length;
      if (this.silenceSamples >= TRACKING_SILENCE_SAMPLES) {
        this._exitTracking("extended silence");
        this.newAudioCount = 0;
        return messages;
      }
    } else {
      this.silenceSamples = 0;
    }

    // Faster trigger in tracking mode
    if (this.newAudioCount < TRACKING_TRIGGER_SAMPLES) {
      return messages;
    }
    this.newAudioCount = 0;

    // Transcribe
    const { text, rawPhonemes } = await this.transcribe(this.fullAudio.slice());
    if (!text || text.trim().length < 3) return messages;

    const recognizedWords = text.split(" ");

    // Align against known verse (using joined phoneme words)
    const resumeFrom = Math.max(this.trackingLastWordIdx, 0);
    let { matchedIndices } = alignPosition(
      recognizedWords,
      this.trackingVerseWords,
      resumeFrom,
    );

    // Fallback: character-level progress when word alignment fails
    // (model often outputs spaceless phoneme strings)
    // Only for verses with 10+ words where word-level alignment is unreliable
    if (matchedIndices.length === 0 && text.length >= 5 && this.trackingVerseWords.length >= 10) {
      const charWordIdx = this._charLevelProgress(text);
      if (charWordIdx > this.trackingLastWordIdx) {
        matchedIndices = [charWordIdx];
      }
    }

    // Check for stale tracking
    const advanced =
      matchedIndices.length > 0 &&
      matchedIndices[matchedIndices.length - 1] > this.trackingLastWordIdx;

    if (!advanced) {
      this.staleCycles++;
      if (this.staleCycles >= STALE_CYCLE_LIMIT) {
        this._exitTracking(
          `stale (${this.staleCycles} cycles, no progress)`,
        );
        // Let audio buffer accumulate naturally before re-attempting discovery
        this.newAudioCount = 0;
        return messages;
      }
    } else {
      this.staleCycles = 0;
    }

    // Send word_progress if advanced
    if (advanced) {
      this.trackingLastWordIdx =
        matchedIndices[matchedIndices.length - 1];
      const wordPos = this.trackingLastWordIdx + 1;
      messages.push({
        type: "word_progress",
        surah: this.trackingVerse!.surah,
        ayah: this.trackingVerse!.ayah,
        word_index: wordPos,
        total_words: this.trackingVerseWords.length,
        matched_indices: matchedIndices,
      });

      // Compute and emit word corrections for the recited portion
      const corrections = computeCorrection(
        rawPhonemes,
        this.trackingVerse!.phonemes,
        wordPos,
      );
      if (corrections.length > 0) {
        messages.push({
          type: "word_correction",
          surah: this.trackingVerse!.surah,
          ayah: this.trackingVerse!.ayah,
          corrections,
        });
      }
    }

    // Check if verse is complete
    if (matchedIndices.length > 0) {
      const cumulativeCoverage =
        (this.trackingLastWordIdx + 1) / this.trackingVerseWords.length;
      const nearEnd =
        this.trackingLastWordIdx >=
        this.trackingVerseWords.length - 2;

      if (cumulativeCoverage >= 0.8 && nearEnd) {
        // Advance to next verse
        const curRef: [number, number] = [
          this.trackingVerse!.surah,
          this.trackingVerse!.ayah,
        ];
        this.lastEmittedRef = curRef;
        this.lastEmittedText = this.trackingVerse!.phonemes_joined;
        this.cyclesSinceEmit = 0;
        const nextV = this.db.getNextVerse(curRef[0], curRef[1]);
        this._exitTracking("verse complete");

        if (nextV) {
          const nextRef: [number, number] = [nextV.surah, nextV.ayah];
          const surrounding = getSurroundingVerses(
            this.db,
            nextV.surah,
            nextV.ayah,
          );
          messages.push({
            type: "verse_match",
            surah: nextV.surah,
            ayah: nextV.ayah,
            verse_text: nextV.text_uthmani,
            surah_name: nextV.surah_name,
            confidence: 0.99,
            surrounding_verses: surrounding,
          });
          // Save completed verse state for recovery if next-verse tracking fails
          this.prevEmittedRef = this.lastEmittedRef;
          this.prevEmittedText = this.lastEmittedText;
          this.lastEmittedRef = nextRef;
          this.lastEmittedText = nextV.phonemes_joined;
          this._enterTracking(nextV, nextRef);
        }

        // Reset audio window — keep last 2s
        const keepSamples = Math.min(
          this.fullAudio.length,
          TRIGGER_SAMPLES,
        );
        this.fullAudio = this.fullAudio.slice(-keepSamples);
      }
    }

    return messages;
  }

  private async _handleDiscovery(): Promise<WorkerOutbound[]> {
    const messages: WorkerOutbound[] = [];

    if (this.newAudioCount < TRIGGER_SAMPLES) return messages;
    this.newAudioCount = 0;
    this.cyclesSinceEmit++;

    // Skip silent chunks
    const tail = this.fullAudio.slice(-TRIGGER_SAMPLES);
    if (isSilence(tail)) return messages;

    // Transcribe
    const { text } = await this.transcribe(this.fullAudio.slice());
    if (!text || text.trim().length < 5) return messages;

    // Skip if transcription is mostly residual from last emitted verse
    if (this.lastEmittedText) {
      const residual = partialRatio(text, this.lastEmittedText);
      if (residual > 0.7) return messages;
    }

    // Match against QuranDB
    const match = this.db.matchVerse(
      text,
      RAW_TRANSCRIPT_THRESHOLD,
      4,
      this.lastEmittedRef,
      5,
    );

    // Anti-cascade: shortly after an emit, require higher threshold for
    // non-continuation jumps to prevent false positives from cascading
    let effectiveThreshold = this.hasEverMatched
      ? VERSE_MATCH_THRESHOLD
      : FIRST_MATCH_THRESHOLD;

    if (match && this.hasEverMatched && this.cyclesSinceEmit <= 2 && this.lastEmittedRef) {
      const isContinuation =
        match.surah === this.lastEmittedRef[0] &&
        match.ayah >= this.lastEmittedRef[1] + 1 &&
        match.ayah <= this.lastEmittedRef[1] + 3;
      if (!isContinuation) {
        effectiveThreshold = Math.max(effectiveThreshold, 0.65);
      }
    }

    if (match && match.score >= effectiveThreshold) {
      const ref: [number, number] = [match.surah, match.ayah];

      // Ambiguity guard: only suppress when scores are nearly identical
      // and the transcript hasn't clearly differentiated the verses.
      const runnersUp: Record<string, any>[] = match.runners_up ?? [];
      if (runnersUp.length >= 2) {
        const matchVerse = this.db.getVerse(match.surah, match.ayah);
        let altRunner: Record<string, any> | null = null;
        for (const ru of runnersUp) {
          if (ru.surah !== match.surah || ru.ayah !== match.ayah) {
            altRunner = ru;
            break;
          }
        }
        // Only guard when alt is within 3% of top score
        if (altRunner && altRunner.score >= runnersUp[0].score * 0.97) {
          const altVerse = this.db.getVerse(altRunner.surah, altRunner.ayah);
          if (matchVerse && altVerse) {
            const w1 = matchVerse.phonemes_joined.split(" ");
            const w2 = altVerse.phonemes_joined.split(" ");
            let sharedPrefix = 0;
            for (
              let i = 0;
              i < Math.min(w1.length, w2.length);
              i++
            ) {
              if (w1[i] === w2[i]) sharedPrefix++;
              else break;
            }
            // Require longer shared prefix (8+ words) and very short text
            if (sharedPrefix >= 8) {
              const textWords = text.split(" ").length;
              if (textWords <= sharedPrefix + 2) {
                messages.push({
                  type: "raw_transcript",
                  text,
                  confidence:
                    Math.round(match.score * 100) / 100,
                });
                return messages;
              }
            }
          }
        }
      }

      // Dedup: skip if same verse was just sent
      if (
        this.lastEmittedRef &&
        this.lastEmittedRef[0] === ref[0] &&
        this.lastEmittedRef[1] === ref[1]
      ) {
        return messages;
      }

      const verse = this.db.getVerse(match.surah, match.ayah);
      const surrounding = getSurroundingVerses(
        this.db,
        match.surah,
        match.ayah,
      );

      messages.push({
        type: "verse_match",
        surah: match.surah,
        ayah: match.ayah,
        verse_text: verse?.text_uthmani ?? match.text ?? "",
        surah_name: verse?.surah_name ?? "",
        confidence: Math.round(match.score * 100) / 100,
        surrounding_verses: surrounding,
      });

      this.hasEverMatched = true;
      this.cyclesSinceEmit = 0;

      // For multi-verse spans, advance hint to the last verse
      const ayahEnd = match.ayah_end;
      const effectiveRef: [number, number] = ayahEnd
        ? [match.surah, ayahEnd]
        : ref;
      // Save pre-match state for recovery if tracking determines misidentification
      this.prevEmittedRef = this.lastEmittedRef;
      this.prevEmittedText = this.lastEmittedText;
      this.lastEmittedRef = effectiveRef;
      this.lastEmittedText =
        match.phonemes_joined ?? verse?.phonemes_joined ?? "";

      // Enter tracking mode
      if (verse) {
        this._enterTracking(verse, ref);
      } else {
        // No tracking — reset window
        this.fullAudio = tail.slice();
      }
    } else {
      // Send raw transcript
      const score = match ? Math.round(match.score * 100) / 100 : 0;
      messages.push({
        type: "raw_transcript",
        text,
        confidence: score,
      });
    }

    return messages;
  }

  private _charLevelProgress(text: string): number {
    if (!this.trackingVerse) return -1;
    const joined = this.trackingVerse.phonemes_joined;
    const words = this.trackingVerseWords;
    if (!joined || words.length === 0) return -1;

    // Compare no-space text against no-space verse for spaceless model output
    const noSpaceText = text.replace(/ /g, "");
    const noSpaceJoined = joined.replace(/ /g, "");
    const tLen = noSpaceText.length;
    if (tLen < 3 || tLen >= noSpaceJoined.length) return -1;

    // Slide transcript-sized window across verse, find best match position
    let bestScore = 0;
    let bestEnd = 0;
    // Step by ~10 chars for speed, then refine
    const step = Math.max(1, Math.floor(tLen / 5));
    for (let i = 0; i <= noSpaceJoined.length - tLen; i += step) {
      const span = noSpaceJoined.slice(i, i + tLen);
      const s = levRatio(noSpaceText, span);
      if (s > bestScore) {
        bestScore = s;
        bestEnd = i + tLen;
      }
    }
    // Refine around best position
    if (step > 1) {
      const refStart = Math.max(0, bestEnd - tLen - step);
      const refEnd = Math.min(noSpaceJoined.length - tLen, bestEnd - tLen + step);
      for (let i = refStart; i <= refEnd; i++) {
        const span = noSpaceJoined.slice(i, i + tLen);
        const s = levRatio(noSpaceText, span);
        if (s > bestScore) {
          bestScore = s;
          bestEnd = i + tLen;
        }
      }
    }

    if (bestScore < 0.55) return -1;

    // Map bestEnd position in no-space string back to word index
    // Count chars consumed per word (without spaces) to find which word bestEnd falls in
    let charCount = 0;
    for (let w = 0; w < words.length; w++) {
      charCount += words[w].length;
      if (charCount >= bestEnd) return w;
    }
    return words.length - 1;
  }

  private _enterTracking(verse: QuranVerse, _ref: [number, number]): void {
    this.trackingVerse = verse;
    this.trackingVerseWords = verse.phoneme_words;
    this.trackingLastWordIdx = -1;
    this.silenceSamples = 0;
    this.staleCycles = 0;
  }

  private _exitTracking(reason: string): void {
    const verseLen = this.trackingVerseWords.length;
    const progress =
      verseLen > 0 ? (this.trackingLastWordIdx + 1) / verseLen : 0;

    if (reason === "verse complete") {
      // Caller already updated lastEmittedRef/Text
      this.hasEverMatched = true;
    } else if (reason.startsWith("stale") && progress < 0.5) {
      // Low progress + stale = likely misidentification
      this.lastEmittedRef = this.prevEmittedRef;
      this.lastEmittedText = this.prevEmittedText;
    } else if (
      reason.startsWith("stale") &&
      this.trackingVerseWords.length > 0 &&
      this.trackingLastWordIdx >= 0
    ) {
      // Good progress + stale = was tracking correctly but user
      // paused or diverged. Trim residual text to tracked portion.
      this.hasEverMatched = true;
      this.lastEmittedText = this.trackingVerseWords
        .slice(0, this.trackingLastWordIdx + 1)
        .join(" ");
    }

    this.trackingVerse = null;
    this.trackingVerseWords = [];
    this.trackingLastWordIdx = -1;
    this.silenceSamples = 0;
    this.staleCycles = 0;
  }
}
