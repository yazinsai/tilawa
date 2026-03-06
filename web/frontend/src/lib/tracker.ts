import { ratio as levRatio } from "./levenshtein";
import { scoreCtcSequence, scoreCtcCandidates, chooseLongestStablePrefix } from "./ctc-rescore";
import { QuranDB, partialRatio, type QuranCandidate } from "./quran-db";
import { computeCorrection } from "./correction";
import type { AcousticEvidence } from "./ctc-rescore";
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
  DISCOVERY_REPEAT_CYCLES,
  DISCOVERY_TOP_SINGLE_CANDIDATES,
  DISCOVERY_TOP_SURAHS,
  DISCOVERY_MAX_SPAN,
  ACOUSTIC_CLEAR_MARGIN,
  ACOUSTIC_CONTINUATION_MARGIN,
  TRACKING_PREFIX_TOLERANCE,
  TRACKING_WEAK_COMMIT_CONFIDENCE,
  UTTERANCE_FINAL_SILENCE_SAMPLES,
  NON_CONTINUATION_JUMP_THRESHOLD,
  ADVANCE_RELATIVE_MARGIN,
  ADVANCE_PREFIX_TOKENS,
  ACOUSTIC_OVERRIDE_TEXT_THRESHOLD,
  ACOUSTIC_OVERRIDE_MIN_MARGIN,
  DISCOVERY_EXPANDED_CANDIDATES,
} from "./types";

export interface TranscribeResult {
  text: string;
  rawPhonemes: string;
  tokenIds?: number[];
  acoustic?: AcousticEvidence;
}

type TranscribeFn = (audio: Float32Array) => Promise<TranscribeResult>;

interface PendingLeader {
  key: string;
  count: number;
}

interface CommitEvidence {
  confidence: number;
  acousticMargin: number;
  strong: boolean;
}

interface RankedCandidate {
  candidate: QuranCandidate;
  acousticScore: number;
  acousticMargin: number;
  feasible: boolean;
  lengthFit: number;
}

interface TrackingPrefix {
  wordIndex: number;
  ids: number[];
}

export type TrackerDiagnosticEvent =
  | {
      type: "discovery_cycle";
      text: string;
      final_flush: boolean;
      candidates: Array<{
        ref: string;
        kind: "single" | "span";
        stageA: number;
        acoustic: number;
      }>;
    }
  | { type: "silence_skip"; mode: "discovery" | "tracking"; reason: string }
  | { type: "commit"; ref: string; reason: string; confidence: number }
  | { type: "rollback"; reason: string; restored_ref: string | null }
  | { type: "stale_exit"; ref: string; stale_cycles: number }
  | { type: "flush"; mode: "discovery" | "tracking"; duration_sec: number };

export interface RecitationTrackerOptions {
  onDiagnostic?: (event: TrackerDiagnosticEvent) => void;
}

function concatFloat32(a: Float32Array, b: Float32Array): Float32Array {
  const result = new Float32Array(a.length + b.length);
  result.set(a);
  result.set(b, a.length);
  return result;
}

function isSilence(audio: Float32Array): boolean {
  if (audio.length === 0) return true;
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

function refKey(surah: number, ayah: number, ayahEnd?: number | null): string {
  return ayahEnd && ayahEnd !== ayah
    ? `${surah}:${ayah}-${ayahEnd}`
    : `${surah}:${ayah}`;
}

export class RecitationTracker {
  private utteranceAudio = new Float32Array(0);
  private newAudioCount = 0;
  private silenceSamples = 0;
  private utteranceHasSpeech = false;
  private didFinalFlush = false;

  private lastEmittedRef: [number, number] | null = null;
  private lastEmittedText = "";
  private prevEmittedRef: [number, number] | null = null;
  private prevEmittedText = "";
  private pendingLeader: PendingLeader | null = null;
  private lastCommitEvidence: CommitEvidence | null = null;

  private trackingVerse: QuranVerse | null = null;
  private trackingVerseWords: string[] = [];
  private trackingPrefixes: TrackingPrefix[] = [];
  private trackingLastWordIdx = -1;
  private trackingProgressEstablished = false;
  private staleCycles = 0;
  private cyclesSinceCommit = Infinity;
  private lastTrackingResult: TranscribeResult | null = null;
  private consecutiveAutoAdvances = 0;

  constructor(
    private db: QuranDB,
    private transcribe: TranscribeFn,
    private options: RecitationTrackerOptions = {},
  ) {}

  async feed(samples: Float32Array): Promise<WorkerOutbound[]> {
    const messages: WorkerOutbound[] = [];

    this.utteranceAudio = concatFloat32(this.utteranceAudio, samples);
    const maxSamples =
      this.trackingVerse !== null
        ? TRACKING_MAX_WINDOW_SAMPLES
        : MAX_WINDOW_SAMPLES;
    if (this.utteranceAudio.length > maxSamples) {
      this.utteranceAudio = this.utteranceAudio.slice(-maxSamples);
    }

    this.newAudioCount += samples.length;

    if (isSilence(samples)) {
      this.silenceSamples += samples.length;
    } else {
      this.silenceSamples = 0;
      this.utteranceHasSpeech = true;
      this.didFinalFlush = false;
    }

    const finalFlush =
      this.utteranceHasSpeech &&
      !this.didFinalFlush &&
      this.silenceSamples >= UTTERANCE_FINAL_SILENCE_SAMPLES;

    if (this.trackingVerse !== null) {
      messages.push(...(await this._handleTracking(finalFlush)));
    } else {
      messages.push(...(await this._handleDiscovery(finalFlush)));
    }

    if (finalFlush) {
      this.didFinalFlush = true;
      this._emitDiagnostic({
        type: "flush",
        mode: this.trackingVerse ? "tracking" : "discovery",
        duration_sec: this.utteranceAudio.length / SAMPLE_RATE,
      });
      if (this.trackingVerse === null) {
        this._resetUtterance();
      }
    }

    return messages;
  }

  private async _handleTracking(finalFlush: boolean): Promise<WorkerOutbound[]> {
    const messages: WorkerOutbound[] = [];
    if (!this.trackingVerse) return messages;

    if (!finalFlush && this.newAudioCount < TRACKING_TRIGGER_SAMPLES) {
      if (this.silenceSamples >= TRACKING_SILENCE_SAMPLES) {
        this._rollbackWeakCommit("tracking silence timeout");
        this._exitTracking("extended silence");
      }
      return messages;
    }
    this.newAudioCount = 0;

    const result = await this.transcribe(this.utteranceAudio.slice());
    this.lastTrackingResult = result;
    const text = result.text.trim();
    if (!text && !finalFlush) {
      return messages;
    }

    const recognizedWords = text.split(" ").filter(Boolean);
    const resumeFrom = Math.max(this.trackingLastWordIdx, 0);
    let { matchedIndices } = alignPosition(
      recognizedWords,
      this.trackingVerseWords,
      resumeFrom,
    );

    if (matchedIndices.length === 0) {
      const acousticIdx = this._resolveTrackingAcousticWord(result);
      if (acousticIdx > this.trackingLastWordIdx) {
        matchedIndices = [acousticIdx];
      }
    }

    if (
      matchedIndices.length === 0 &&
      text.length >= 5 &&
      this.trackingVerseWords.length >= 10
    ) {
      const charWordIdx = this._charLevelProgress(text);
      if (charWordIdx > this.trackingLastWordIdx) {
        matchedIndices = [charWordIdx];
      }
    }

    const advanced =
      matchedIndices.length > 0 &&
      matchedIndices[matchedIndices.length - 1] > this.trackingLastWordIdx;

    if (!advanced) {
      this.staleCycles++;
      if (this.staleCycles >= STALE_CYCLE_LIMIT || finalFlush) {
        this._emitDiagnostic({
          type: "stale_exit",
          ref: `${this.trackingVerse.surah}:${this.trackingVerse.ayah}`,
          stale_cycles: this.staleCycles,
        });
        this._rollbackWeakCommit(finalFlush ? "final silence flush" : "stale tracking");
        this._exitTracking(finalFlush ? "final silence flush" : "stale tracking");
      }
      return messages;
    }

    this.staleCycles = 0;
    this.trackingProgressEstablished = true;
    this.trackingLastWordIdx = matchedIndices[matchedIndices.length - 1];
    const wordPos = this.trackingLastWordIdx + 1;

    messages.push({
      type: "word_progress",
      surah: this.trackingVerse.surah,
      ayah: this.trackingVerse.ayah,
      word_index: wordPos,
      total_words: this.trackingVerseWords.length,
      matched_indices: matchedIndices,
    });

    const corrections = computeCorrection(
      result.rawPhonemes,
      this.trackingVerse.phonemes,
      wordPos,
    );
    if (corrections.length > 0) {
      messages.push({
        type: "word_correction",
        surah: this.trackingVerse.surah,
        ayah: this.trackingVerse.ayah,
        corrections,
      });
    }

    const cumulativeCoverage = wordPos / this.trackingVerseWords.length;
    const nearEnd = this.trackingLastWordIdx >= this.trackingVerseWords.length - 2;
    if (cumulativeCoverage >= 0.8 && nearEnd) {
      if (!(this.lastCommitEvidence?.strong)) {
        this._exitTracking("weak completion");
        return messages;
      }

      const currentRef: [number, number] = [
        this.trackingVerse.surah,
        this.trackingVerse.ayah,
      ];
      const currentIds = this.trackingVerse.phoneme_token_ids ?? [];
      this.lastEmittedRef = currentRef;
      this.lastEmittedText = this.trackingVerse.phonemes_joined;
      const nextVerse = this.db.getNextVerse(currentRef[0], currentRef[1]);
      this._exitTracking("verse complete");

      if (nextVerse) {
        let advanceOk = true; // default: advance (preserves behavior when no acoustic data)

        const acoustic = this.lastTrackingResult?.acoustic;
        const nextIds = nextVerse.phoneme_token_ids ?? [];

        if (acoustic && currentIds.length > 0 && nextIds.length > 0) {
          // Relative evidence gate: compare current verse suffix vs next verse prefix
          // Both use ~ADVANCE_PREFIX_TOKENS tokens for comparable normalization
          const n = ADVANCE_PREFIX_TOKENS;
          const suffixIds = currentIds.slice(-Math.min(n, currentIds.length));
          const prefixIds = nextIds.slice(0, Math.min(n, nextIds.length));

          const suffixScore = scoreCtcSequence(acoustic, suffixIds);
          const prefixScore = scoreCtcSequence(acoustic, prefixIds);

          // Both must be finite (feasible). If suffix is infeasible, audio is bad — block.
          // If prefix is infeasible, next verse isn't in the audio at all — block.
          if (
            !Number.isFinite(suffixScore) ||
            !Number.isFinite(prefixScore)
          ) {
            advanceOk = false;
          } else {
            advanceOk = (prefixScore - suffixScore) < ADVANCE_RELATIVE_MARGIN;
          }
        }

        if (advanceOk) {
          messages.push({
            type: "verse_match",
            surah: nextVerse.surah,
            ayah: nextVerse.ayah,
            verse_text: nextVerse.text_uthmani,
            surah_name: nextVerse.surah_name,
            confidence: 0.99,
            surrounding_verses: getSurroundingVerses(
              this.db,
              nextVerse.surah,
              nextVerse.ayah,
            ),
          });
          this.prevEmittedRef = currentRef;
          this.prevEmittedText = this.lastEmittedText;
          this.lastEmittedRef = [nextVerse.surah, nextVerse.ayah];
          this.lastEmittedText = nextVerse.phonemes_joined;
          this.lastCommitEvidence = {
            confidence: 0.99,
            acousticMargin: 1,
            strong: true,
          };
          this._enterTracking(nextVerse);
          this.consecutiveAutoAdvances++;
          // After sustained auto-advances, degrade to weak so stale-exit
          // triggers rediscovery instead of persisting
          if (this.consecutiveAutoAdvances >= 5) {
            this.lastCommitEvidence = {
              ...this.lastCommitEvidence!,
              strong: false,
            };
          }
        }
        // If !advanceOk, we already exited tracking — falls through to rediscovery
      }

      this._retainTailAfterCommit();
    }

    return messages;
  }

  private async _handleDiscovery(finalFlush: boolean): Promise<WorkerOutbound[]> {
    const messages: WorkerOutbound[] = [];

    if (!this.utteranceHasSpeech) {
      this._emitDiagnostic({
        type: "silence_skip",
        mode: "discovery",
        reason: "no speech detected",
      });
      return messages;
    }

    if (!finalFlush && this.newAudioCount < TRIGGER_SAMPLES) return messages;
    this.newAudioCount = 0;
    this.cyclesSinceCommit++;

    const result = await this.transcribe(this.utteranceAudio.slice());
    const text = result.text.trim();
    if (!text || text.length < 5) {
      this._emitDiagnostic({
        type: "silence_skip",
        mode: "discovery",
        reason: "transcript too short",
      });
      return messages;
    }

    if (this.lastEmittedText && this.lastCommitEvidence?.strong) {
      const residual = partialRatio(text, this.lastEmittedText);
      if (residual > 0.7 && !finalFlush) {
        this._emitDiagnostic({
          type: "silence_skip",
          mode: "discovery",
          reason: `residual=${residual.toFixed(3)}`,
        });
        return messages;
      }
    }

    const match = this.db.matchVerse(
      text,
      RAW_TRANSCRIPT_THRESHOLD,
      DISCOVERY_MAX_SPAN,
      this.lastEmittedRef,
      5,
    );
    // Expand candidate set when text match is unreliable
    const textConfidenceLow = !match || match.score < ACOUSTIC_OVERRIDE_TEXT_THRESHOLD;
    const singleLimit = textConfidenceLow
      ? DISCOVERY_EXPANDED_CANDIDATES
      : DISCOVERY_TOP_SINGLE_CANDIDATES;
    const retrieved = this.db.retrieveCandidates(text, {
      maxSpan: DISCOVERY_MAX_SPAN,
      hint: this.lastEmittedRef,
      singleLimit,
      topSurahs: textConfidenceLow ? 10 : DISCOVERY_TOP_SURAHS,
      spanLimit: DISCOVERY_TOP_SINGLE_CANDIDATES,
    });
    const ranked = this._rankCandidates(retrieved.combined, result);

    this._emitDiagnostic({
      type: "discovery_cycle",
      text,
      final_flush: finalFlush,
      candidates: ranked.slice(0, 8).map((entry) => ({
        ref: refKey(
          entry.candidate.surah,
          entry.candidate.ayah,
          entry.candidate.ayah_end,
        ),
        kind: entry.candidate.kind,
        stageA: Math.round(entry.candidate.stage_a_score * 1000) / 1000,
        acoustic: Math.round(entry.acousticScore * 1000) / 1000,
      })),
    });

    let acousticMargin = 0;
    let lengthFit = 1;
    let effectiveMatch = match;

    // If acoustic evidence available, check if CTC-best differs from text-best
    const acousticBest = ranked.find((r) => r.feasible);
    if (match) {
      const matchKey = refKey(match.surah, match.ayah, match.ayah_end);
      const rescoredMatch = ranked.find(
        (entry) =>
          refKey(
            entry.candidate.surah,
            entry.candidate.ayah,
            entry.candidate.ayah_end,
          ) === matchKey,
      );
      acousticMargin = rescoredMatch?.acousticMargin ?? 0;
      lengthFit = rescoredMatch?.lengthFit ?? 1;

      // Acoustic override: when text match is unreliable, trust CTC
      if (
        acousticBest &&
        match.score < ACOUSTIC_OVERRIDE_TEXT_THRESHOLD &&
        acousticBest.acousticMargin >= ACOUSTIC_OVERRIDE_MIN_MARGIN
      ) {
        const abKey = refKey(
          acousticBest.candidate.surah,
          acousticBest.candidate.ayah,
          acousticBest.candidate.ayah_end,
        );
        if (abKey !== matchKey) {
          // CTC winner differs from text winner — override
          effectiveMatch = {
            surah: acousticBest.candidate.surah,
            ayah: acousticBest.candidate.ayah,
            ayah_end: acousticBest.candidate.ayah_end,
            text: acousticBest.candidate.text,
            phonemes_joined: acousticBest.candidate.phonemes_joined,
            score: Math.max(match.score, 0.5), // synthesize confidence
            raw_score: acousticBest.candidate.raw_score,
            bonus: acousticBest.candidate.bonus,
          };
          acousticMargin = acousticBest.acousticMargin;
          lengthFit = acousticBest.lengthFit;
        }
      }
    }

    const threshold = this.lastEmittedRef ? VERSE_MATCH_THRESHOLD : FIRST_MATCH_THRESHOLD;

    if (effectiveMatch && effectiveMatch.score >= threshold) {
      const key = refKey(effectiveMatch.surah, effectiveMatch.ayah, effectiveMatch.ayah_end);
      this.pendingLeader =
        this.pendingLeader?.key === key
          ? { key, count: this.pendingLeader.count + 1 }
          : { key, count: 1 };

      const isContinuation = this._isContinuation(effectiveMatch.surah, effectiveMatch.ayah);
      const clearMargin =
        lengthFit >= 0.6 &&
        acousticMargin >=
        (isContinuation ? ACOUSTIC_CONTINUATION_MARGIN : ACOUSTIC_CLEAR_MARGIN);
      const repeatedLeader =
        (this.pendingLeader?.count ?? 0) >= DISCOVERY_REPEAT_CYCLES;

      // Anti-cascade: shortly after a commit, require higher score for
      // non-continuation jumps to prevent false positives
      let effectivelyBlocked = false;
      if (
        !isContinuation &&
        this.lastEmittedRef &&
        this.cyclesSinceCommit <= 2
      ) {
        if (effectiveMatch.score < NON_CONTINUATION_JUMP_THRESHOLD && !repeatedLeader) {
          effectivelyBlocked = true;
        }
      }

      if (!effectivelyBlocked && (clearMargin || repeatedLeader)) {
        const ref: [number, number] = [effectiveMatch.surah, effectiveMatch.ayah];
        if (
          this.lastEmittedRef &&
          this.lastEmittedRef[0] === ref[0] &&
          this.lastEmittedRef[1] === ref[1]
        ) {
          return messages;
        }

        const verse = this.db.getVerse(effectiveMatch.surah, effectiveMatch.ayah);
        const surrounding = getSurroundingVerses(
          this.db,
          effectiveMatch.surah,
          effectiveMatch.ayah,
        );
        const confidence = Math.max(
          effectiveMatch.score,
          Math.min(0.99, 0.45 + acousticMargin + lengthFit * 0.2),
        );

        messages.push({
          type: "verse_match",
          surah: effectiveMatch.surah,
          ayah: effectiveMatch.ayah,
          verse_text: verse?.text_uthmani ?? effectiveMatch.text ?? "",
          surah_name: verse?.surah_name ?? "",
          confidence: Math.round(confidence * 100) / 100,
          surrounding_verses: surrounding,
        });

        this.prevEmittedRef = this.lastEmittedRef;
        this.prevEmittedText = this.lastEmittedText;
        const ayahEnd = effectiveMatch.ayah_end;
        const effectiveRef: [number, number] = ayahEnd
          ? [effectiveMatch.surah, ayahEnd]
          : ref;
        this.lastEmittedRef = effectiveRef;
        this.lastEmittedText =
          effectiveMatch.phonemes_joined ?? verse?.phonemes_joined ?? "";
        this.lastCommitEvidence = {
          confidence,
          acousticMargin,
          strong:
            confidence >= TRACKING_WEAK_COMMIT_CONFIDENCE &&
            lengthFit >= 0.8 &&
            clearMargin,
        };
        this.pendingLeader = null;
        this.cyclesSinceCommit = 0;
        this.consecutiveAutoAdvances = 0;

        this._emitDiagnostic({
          type: "commit",
          ref: key,
          reason: clearMargin ? "acoustic_margin" : "repeat_leader",
          confidence: Math.round(confidence * 1000) / 1000,
        });

        if (verse) {
          this._enterTracking(verse);
        } else {
          this._retainTailAfterCommit();
        }
      } else {
        messages.push({
          type: "raw_transcript",
          text,
          confidence: Math.round(effectiveMatch.score * 100) / 100,
        });
      }
    } else {
      const score = effectiveMatch ? Math.round(effectiveMatch.score * 100) / 100 : 0;
      messages.push({
        type: "raw_transcript",
        text,
        confidence: score,
      });
    }

    return messages;
  }

  private _resolveTrackingAcousticWord(result: TranscribeResult): number {
    if (!result.acoustic || !this.trackingPrefixes.length) {
      return -1;
    }

    const start = Math.max(this.trackingLastWordIdx, 0);
    const prefixes = this.trackingPrefixes.slice(start);
    const scored = scoreCtcCandidates(
      result.acoustic,
      prefixes.map((prefix) => ({
        ids: prefix.ids,
        meta: prefix,
        priorScore: prefix.wordIndex + 1,
      })),
    );
    const stable = chooseLongestStablePrefix(scored, TRACKING_PREFIX_TOLERANCE);
    return stable?.meta.wordIndex ?? -1;
  }

  private _rankCandidates(
    candidates: QuranCandidate[],
    result: TranscribeResult,
  ): RankedCandidate[] {
    if (!result.acoustic || candidates.length === 0) {
      return candidates
        .map((candidate) => ({
          candidate,
          acousticScore: 0,
          acousticMargin: 0,
          feasible: false,
          lengthFit: 1,
        }))
        .sort((a, b) => b.candidate.stage_a_score - a.candidate.stage_a_score);
    }

    const observedLength = Math.max(result.tokenIds?.length ?? 0, 1);
    const scored = scoreCtcCandidates(
      result.acoustic,
      candidates.map((candidate) => ({
        ids: candidate.phoneme_token_ids,
        meta: candidate,
        priorScore: candidate.stage_a_score,
      })),
    );
    const feasibleScores = scored
      .filter((entry) => entry.feasible)
      .map((entry) => entry.acousticScore);
    const minAcoustic = feasibleScores.length ? Math.min(...feasibleScores) : 0;
    const maxAcoustic = feasibleScores.length ? Math.max(...feasibleScores) : 1;
    const acousticRange = Math.max(maxAcoustic - minAcoustic, 1e-6);

    const ranked = scored.map((entry, idx) => {
      const candidateLength = Math.max(entry.meta.phoneme_token_ids.length, 1);
      const lengthFit =
        Math.min(candidateLength, observedLength) /
        Math.max(candidateLength, observedLength);
      const acousticFit = entry.feasible
        ? 1 - (entry.acousticScore - minAcoustic) / acousticRange
        : 0;

      return {
        candidate: entry.meta,
        acousticScore: entry.acousticScore,
        acousticMargin:
          (scored[idx + 1]?.acousticScore ?? entry.acousticScore) - entry.acousticScore,
        feasible: entry.feasible,
        lengthFit,
      };
    });

    ranked.sort((a, b) => {
      if (b.candidate.stage_a_score !== a.candidate.stage_a_score) {
        return b.candidate.stage_a_score - a.candidate.stage_a_score;
      }
      return a.acousticScore - b.acousticScore;
    });
    return ranked;
  }

  private _charLevelProgress(text: string): number {
    if (!this.trackingVerse) return -1;
    const joined = this.trackingVerse.phonemes_joined;
    const words = this.trackingVerseWords;
    if (!joined || words.length === 0) return -1;

    const noSpaceText = text.replace(/ /g, "");
    const noSpaceJoined = joined.replace(/ /g, "");
    const textLen = noSpaceText.length;
    if (textLen < 3 || textLen >= noSpaceJoined.length) return -1;

    let bestScore = 0;
    let bestEnd = 0;
    const step = Math.max(1, Math.floor(textLen / 5));
    for (let i = 0; i <= noSpaceJoined.length - textLen; i += step) {
      const span = noSpaceJoined.slice(i, i + textLen);
      const score = levRatio(noSpaceText, span);
      if (score > bestScore) {
        bestScore = score;
        bestEnd = i + textLen;
      }
    }
    if (step > 1) {
      const refineStart = Math.max(0, bestEnd - textLen - step);
      const refineEnd = Math.min(noSpaceJoined.length - textLen, bestEnd - textLen + step);
      for (let i = refineStart; i <= refineEnd; i++) {
        const span = noSpaceJoined.slice(i, i + textLen);
        const score = levRatio(noSpaceText, span);
        if (score > bestScore) {
          bestScore = score;
          bestEnd = i + textLen;
        }
      }
    }

    if (bestScore < 0.55) return -1;

    let charCount = 0;
    for (let w = 0; w < words.length; w++) {
      charCount += words[w].length;
      if (charCount >= bestEnd) return w;
    }
    return words.length - 1;
  }

  private _enterTracking(verse: QuranVerse): void {
    this.trackingVerse = verse;
    this.trackingVerseWords = verse.phoneme_words;
    this.trackingLastWordIdx = -1;
    this.trackingProgressEstablished = false;
    this.staleCycles = 0;
    const tokenIds = verse.phoneme_token_ids ?? [];
    const wordEnds = verse.word_token_ends ?? [];
    this.trackingPrefixes = wordEnds
      .map((end, idx) => ({
        wordIndex: idx,
        ids: tokenIds.slice(0, end),
      }))
      .filter((prefix) => prefix.ids.length > 0);
    this._retainTailAfterCommit();
  }

  private _exitTracking(_reason: string): void {
    this.trackingVerse = null;
    this.trackingVerseWords = [];
    this.trackingPrefixes = [];
    this.trackingLastWordIdx = -1;
    this.trackingProgressEstablished = false;
    this.staleCycles = 0;
    this.lastTrackingResult = null;
  }

  private _rollbackWeakCommit(reason: string): void {
    if (this.lastCommitEvidence?.strong || this.trackingProgressEstablished) {
      return;
    }

    this.lastEmittedRef = this.prevEmittedRef;
    this.lastEmittedText = this.prevEmittedText;
    this.lastCommitEvidence = null;
    this._emitDiagnostic({
      type: "rollback",
      reason,
      restored_ref: this.prevEmittedRef
        ? `${this.prevEmittedRef[0]}:${this.prevEmittedRef[1]}`
        : null,
    });
  }

  private _retainTailAfterCommit(): void {
    if (this.lastCommitEvidence?.strong) {
      const keepSamples = Math.min(this.utteranceAudio.length, TRIGGER_SAMPLES);
      this.utteranceAudio = this.utteranceAudio.slice(-keepSamples);
    }
    this.newAudioCount = 0;
    this.silenceSamples = 0;
    this.utteranceHasSpeech = this.utteranceAudio.length > 0;
    this.didFinalFlush = false;
  }

  private _resetUtterance(): void {
    this.utteranceAudio = new Float32Array(0);
    this.newAudioCount = 0;
    this.silenceSamples = 0;
    this.utteranceHasSpeech = false;
    this.didFinalFlush = false;
    this.pendingLeader = null;
  }

  private _isContinuation(surah: number, ayah: number): boolean {
    if (!this.lastEmittedRef) return false;
    return (
      surah === this.lastEmittedRef[0] &&
      ayah >= this.lastEmittedRef[1] + 1 &&
      ayah <= this.lastEmittedRef[1] + 3
    );
  }

  private _emitDiagnostic(event: TrackerDiagnosticEvent): void {
    this.options.onDiagnostic?.(event);
  }
}
