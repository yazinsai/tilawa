import { ratio as levRatio } from "./levenshtein.js";
import { scoreCtcSequence, scoreCtcCandidates, chooseLongestStablePrefix } from "./ctc-rescore.js";
import { QuranDB, partialRatio, type QuranCandidate, type QuranChampionMatch } from "./quran-db.js";
import type { AcousticEvidence } from "./ctc-rescore.js";
import type {
  FinalSequenceMessage,
  FinalSequenceVerse,
  QuranVerse,
  StreamingConfig,
  VerseCandidate,
  VerseCandidateMessage,
  VerseMatchMessage,
  WorkerOutbound,
  SurroundingVerse,
} from "./types.js";
import {
  DEFAULT_STREAMING_CONFIG,
  SAMPLE_RATE,
  RAW_TRANSCRIPT_THRESHOLD,
  SURROUNDING_CONTEXT,
  DISCOVERY_TOP_SINGLE_CANDIDATES,
  DISCOVERY_TOP_SURAHS,
  DISCOVERY_MAX_SPAN,
  TRACKING_WEAK_COMMIT_CONFIDENCE,
  DISCOVERY_EXPANDED_CANDIDATES,
  DISCOVERY_LOW_CONFIDENCE_WORDS,
  DISCOVERY_LOW_CONFIDENCE_CHARS,
  DISCOVERY_FUSION_TEXT_WEIGHT,
  DISCOVERY_FUSION_ACOUSTIC_WEIGHT,
  DISCOVERY_FUSION_LENGTH_WEIGHT,
  DISCOVERY_FUSION_LOW_TEXT_WEIGHT,
  DISCOVERY_FUSION_LOW_ACOUSTIC_WEIGHT,
  DISCOVERY_FUSION_LOW_LENGTH_WEIGHT,
  DISCOVERY_FUSION_SELECTION_GAP,
  normalizeStreamingConfig,
} from "./types.js";

const SHORT_PENDING_CONFIRM_MAX_WORDS = 12;
const SHORT_PENDING_CONFIRM_MIN_SAMPLES = SAMPLE_RATE;

export interface BeamVerseMatch {
  /** Index into the quran_phonemes array */
  verseIndex: number;
  /** Number of verses in the span (1 for single, 2-3 for multi) */
  spanLength: number;
  /** Beam log-probability score */
  score: number;
}

export interface TranscribeResult {
  text: string;
  rawPhonemes: string;
  tokenIds?: number[];
  acoustic?: AcousticEvidence;
  /** Verse matches from trie-constrained beam search (if available) */
  beamMatches?: BeamVerseMatch[];
  /** Final match from the champion joint03 decode/matcher path. */
  championMatch?: QuranChampionMatch;
  championTranscript?: string;
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
  fusionScore: number;
}

interface TrackingPrefix {
  wordIndex: number;
  ids: number[];
}

interface HypothesisCycle {
  candidates: VerseCandidate[];
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
        acousticMargin?: number;
        lengthFit?: number;
        fusion?: number;
        feasible?: boolean;
      }>;
      beam?: Array<{
        ref: string;
        spanLength: number;
        score: number;
      }>;
    }
  | { type: "silence_skip"; mode: "discovery" | "tracking"; reason: string }
  | {
      type: "tracking_cycle";
      ref: string;
      text_length: number;
      word_matches: number;
      acoustic_word: number | null;
      char_word: number | null;
      advanced: boolean;
      final_flush: boolean;
      word_position: number;
      total_words: number;
      coverage: number;
      pending: boolean;
    }
  | {
      type: "pending_emission";
      action: "armed" | "confirmed" | "final_flush_emit" | "dropped" | "cascade_blocked";
      ref: string;
      margin: number | null;
      fresh_samples: number;
      matched_indices?: number[];
    }
  | {
      type: "advance_decision";
      from_ref: string;
      to_ref: string | null;
      action: "wait" | "armed" | "blocked";
      reason: string;
      word_position: number;
      total_words: number;
      coverage: number;
      completion_target: number;
      final_word: boolean;
      advance_ok: boolean;
      early_advance_ok: boolean;
      margin: number | null;
      normal_margin: number;
      strict_margin: number;
    }
  | {
      type: "commit";
      ref: string;
      reason: string;
      confidence: number;
      origin?: "discovery" | "short_rescue" | "tracking_auto";
      selected_rank?: number | null;
      selected_feasible?: boolean | null;
      selected_fusion?: number | null;
      top_ref?: string | null;
      top_fusion?: number | null;
      effective_score?: number;
      threshold?: number;
      acoustic_margin?: number;
      length_fit?: number;
      clear_margin?: boolean;
      repeated_leader?: boolean;
      final_flush_commit?: boolean;
      is_continuation?: boolean;
    }
  | { type: "rollback"; reason: string; restored_ref: string | null }
  | { type: "stale_exit"; ref: string; stale_cycles: number }
  | { type: "flush"; mode: "discovery" | "tracking"; duration_sec: number };

export interface RecitationTrackerOptions {
  onDiagnostic?: (event: TrackerDiagnosticEvent) => void;
  config?: Partial<StreamingConfig>;
}

// Decode-stability gate: single-cycle clearMargin commits require the
// current decode to be similar (>= STABILITY_RATIO) to the previous cycle's
// decode. The context-sweep diagnostic showed ~50% of every short-context
// prefix decode gets revised when full audio arrives, so single-cycle
// commits during streaming were riding unstable predictions. Repeated-leader
// and finalFlush commits are not gated (they have their own protection).
//
// Default on. Set DECODE_STABILITY_GATE_OFF=1 to disable (benchmarking).
const DECODE_STABILITY_GATE: boolean = (() => {
  try {
    return (globalThis as { process?: { env?: Record<string, string> } })
      .process?.env?.DECODE_STABILITY_GATE_OFF !== "1";
  } catch {
    return true;
  }
})();

function concatFloat32(
  a: Float32Array,
  b: Float32Array,
): Float32Array<ArrayBuffer> {
  const result = new Float32Array(a.length + b.length);
  result.set(a);
  result.set(b, a.length);
  return result;
}

function isSilence(audio: Float32Array, threshold: number): boolean {
  if (audio.length === 0) return true;
  let sumSq = 0;
  for (let i = 0; i < audio.length; i++) {
    sumSq += audio[i] * audio[i];
  }
  const rms = Math.sqrt(sumSq / audio.length);
  return rms < threshold;
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
  lookahead = DEFAULT_STREAMING_CONFIG.lookaheadWords,
): { position: number; matchedIndices: number[] } {
  if (!recognizedWords.length || !verseWords.length) {
    return { position: 0, matchedIndices: [] };
  }

  const matchedIndices: number[] = [];
  let versePtr = startFrom;

  for (const rec of recognizedWords) {
    if (versePtr >= verseWords.length) break;
    const limit = Math.min(versePtr + lookahead, verseWords.length);
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

function hasStrongPendingPrefixEvidence(
  matchedIndices: number[],
  totalWords: number,
): boolean {
  if (matchedIndices.length === 0) return false;

  const first = matchedIndices[0];
  const last = matchedIndices[matchedIndices.length - 1];
  if (totalWords <= 3) {
    return first === 0;
  }

  return first <= 1 && (matchedIndices.length >= 2 || last >= 2);
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

function expandCandidate(candidate: VerseCandidate): FinalSequenceVerse[] {
  const end = candidate.ayah_end && candidate.ayah_end > candidate.ayah
    ? candidate.ayah_end
    : candidate.ayah;
  const verses: FinalSequenceVerse[] = [];
  for (let ayah = candidate.ayah; ayah <= end; ayah++) {
    verses.push({
      surah: candidate.surah,
      ayah,
      confidence: candidate.confidence,
    });
  }
  return verses;
}

function isSameRef(a: FinalSequenceVerse, b: FinalSequenceVerse): boolean {
  return a.surah === b.surah && a.ayah === b.ayah;
}

class StreamingHypothesis {
  private cycles: HypothesisCycle[] = [];
  private committed: FinalSequenceVerse[] = [];

  observeCandidates(message: VerseCandidateMessage): void {
    if (message.candidates.length === 0) return;
    this.cycles.push({ candidates: message.candidates.slice(0, 5) });
    if (this.cycles.length > 80) {
      this.cycles.shift();
    }
  }

  observeCommit(message: VerseMatchMessage): void {
    const verse = {
      surah: message.surah,
      ayah: message.ayah,
      confidence: message.confidence,
    };
    if (!this.committed.some((entry) => isSameRef(entry, verse))) {
      this.committed.push(verse);
    }
  }

  finalize(): FinalSequenceMessage | null {
    const path = this.bestPath();
    const verses = path.length > 0 ? path : this.committed;
    if (verses.length === 0) return null;

    const deduped: FinalSequenceVerse[] = [];
    for (const verse of verses) {
      if (!deduped.some((entry) => isSameRef(entry, verse))) {
        deduped.push(verse);
      }
    }

    const confidence =
      deduped.reduce((sum, verse) => sum + verse.confidence, 0) / deduped.length;
    return {
      type: "final_sequence",
      verses: deduped,
      confidence: Math.round(confidence * 100) / 100,
    };
  }

  reset(): void {
    this.cycles = [];
    this.committed = [];
  }

  private bestPath(): FinalSequenceVerse[] {
    if (this.cycles.length === 0) return [];

    type State = {
      candidate: VerseCandidate;
      score: number;
      prev: number;
      verses: FinalSequenceVerse[];
    };

    let previous: State[] = [];
    for (const cycle of this.cycles) {
      const current: State[] = [];
      for (const candidate of cycle.candidates) {
        const verses = expandCandidate(candidate);
        if (previous.length === 0) {
          current.push({
            candidate,
            score: candidate.confidence,
            prev: -1,
            verses,
          });
          continue;
        }

        let bestPrev = 0;
        let bestScore = Number.NEGATIVE_INFINITY;
        for (let i = 0; i < previous.length; i++) {
          const score = previous[i].score + candidate.confidence + transitionScore(
            previous[i].candidate,
            candidate,
          );
          if (score > bestScore) {
            bestScore = score;
            bestPrev = i;
          }
        }
        current.push({
          candidate,
          score: bestScore,
          prev: bestPrev,
          verses: previous[bestPrev].verses.concat(verses),
        });
      }

      previous = current;
    }

    const best = previous.reduce((a, b) => (b.score > a.score ? b : a));
    return best.verses;
  }
}

function transitionScore(prev: VerseCandidate, next: VerseCandidate): number {
  const params = streamingHypothesisParams();
  if (prev.surah !== next.surah) {
    return next.confidence >= 0.85
      ? params.surahJumpHighConfidence
      : params.surahJump;
  }

  const prevEnd = prev.ayah_end && prev.ayah_end > prev.ayah ? prev.ayah_end : prev.ayah;
  const delta = next.ayah - prevEnd;
  if (delta === 0) return params.sameAyah;
  if (delta === 1) return params.nextAyah;
  if (delta > 1 && delta <= 3) return params.smallForwardPerAyah * delta;
  if (delta < 0) return params.backward;
  return params.farForward;
}

interface StreamingHypothesisParams {
  surahJumpHighConfidence: number;
  surahJump: number;
  sameAyah: number;
  nextAyah: number;
  smallForwardPerAyah: number;
  backward: number;
  farForward: number;
}

const DEFAULT_STREAMING_HYPOTHESIS_PARAMS: StreamingHypothesisParams = {
  surahJumpHighConfidence: -0.35,
  surahJump: -1.25,
  sameAyah: 0.15,
  nextAyah: 0.35,
  smallForwardPerAyah: -0.15,
  backward: -1.0,
  farForward: -0.65,
};

function envNumber(name: string, fallback: number): number {
  const value = (globalThis as { process?: { env?: Record<string, string> } })
    .process?.env?.[`STREAMING_HYPOTHESIS_${name}`];
  if (value === undefined) return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

let cachedStreamingHypothesisParams: StreamingHypothesisParams | null = null;

function streamingHypothesisParams(): StreamingHypothesisParams {
  if (cachedStreamingHypothesisParams) return cachedStreamingHypothesisParams;
  cachedStreamingHypothesisParams = {
    surahJumpHighConfidence: envNumber(
      "SURAH_JUMP_HIGH_CONFIDENCE",
      DEFAULT_STREAMING_HYPOTHESIS_PARAMS.surahJumpHighConfidence,
    ),
    surahJump: envNumber("SURAH_JUMP", DEFAULT_STREAMING_HYPOTHESIS_PARAMS.surahJump),
    sameAyah: envNumber("SAME_AYAH", DEFAULT_STREAMING_HYPOTHESIS_PARAMS.sameAyah),
    nextAyah: envNumber("NEXT_AYAH", DEFAULT_STREAMING_HYPOTHESIS_PARAMS.nextAyah),
    smallForwardPerAyah: envNumber(
      "SMALL_FORWARD_PER_AYAH",
      DEFAULT_STREAMING_HYPOTHESIS_PARAMS.smallForwardPerAyah,
    ),
    backward: envNumber("BACKWARD", DEFAULT_STREAMING_HYPOTHESIS_PARAMS.backward),
    farForward: envNumber("FAR_FORWARD", DEFAULT_STREAMING_HYPOTHESIS_PARAMS.farForward),
  };
  return cachedStreamingHypothesisParams;
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

  // Decode-stability gate state
  private lastRawPhonemes: string | null = null;

  // Deferred emission state
  private trackingPendingEmission = false;
  private pendingEmissionMessage: VerseMatchMessage | null = null;
  // prefixScore - suffixScore at advance time; smaller/more-negative means
  // stronger evidence that next verse is already in the tail audio.
  private pendingEmissionMargin = Number.POSITIVE_INFINITY;
  private preAdvanceSnapshot: {
    emittedRef: [number, number] | null;
    emittedText: string;
    prevEmittedRef: [number, number] | null;
    prevEmittedText: string;
    commitEvidence: CommitEvidence | null;
  } | null = null;
  private totalSamplesFed = 0;
  private samplesAtAdvance = 0;
  private hypothesis = new StreamingHypothesis();
  private config: StreamingConfig;

  private db: QuranDB;
  private transcribe: TranscribeFn;
  private options: RecitationTrackerOptions;

  constructor(
    db: QuranDB,
    transcribe: TranscribeFn,
    options: RecitationTrackerOptions = {},
  ) {
    this.db = db;
    this.transcribe = transcribe;
    this.options = options;
    this.config = normalizeStreamingConfig(options.config);
  }

  setConfig(config: Partial<StreamingConfig>): void {
    this.config = normalizeStreamingConfig(config);
  }

  async feed(samples: Float32Array): Promise<WorkerOutbound[]> {
    const messages: WorkerOutbound[] = [];

    this.totalSamplesFed += samples.length;
    this.utteranceAudio = concatFloat32(this.utteranceAudio, samples);
    const maxSamples =
      this.trackingVerse !== null
        ? this.samplesForSeconds(this.config.trackingMaxWindowSec)
        : this.samplesForSeconds(this.config.discoveryMaxWindowSec);
    if (this.utteranceAudio.length > maxSamples) {
      this.utteranceAudio = this.utteranceAudio.slice(-maxSamples);
    }

    this.newAudioCount += samples.length;

    if (isSilence(samples, this.config.silenceRmsThreshold)) {
      this.silenceSamples += samples.length;
    } else {
      this.silenceSamples = 0;
      this.utteranceHasSpeech = true;
      this.didFinalFlush = false;
    }

    const finalFlush =
      this.utteranceHasSpeech &&
      !this.didFinalFlush &&
      this.silenceSamples >= this.samplesForSeconds(this.config.finalSilenceSec);

    if (this.trackingVerse !== null) {
      messages.push(...(await this._handleTracking(finalFlush)));
    } else {
      messages.push(...(await this._handleDiscovery(finalFlush)));
    }

    for (const message of messages) {
      if (message.type === "verse_candidate") {
        this.hypothesis.observeCandidates(message);
      } else if (message.type === "verse_match") {
        this.hypothesis.observeCommit(message);
      }
    }

    if (finalFlush) {
      const finalSequence = this.hypothesis.finalize();
      if (finalSequence) {
        messages.push(finalSequence);
      }
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

    if (!finalFlush && this.newAudioCount < this.samplesForSeconds(this.config.trackingTriggerSec)) {
      if (this.silenceSamples >= this.samplesForSeconds(this.config.trackingSilenceTimeoutSec)) {
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

    if (
      finalFlush &&
      this.trackingPendingEmission &&
      this.pendingEmissionMessage !== null &&
      this.pendingEmissionMargin < this.config.advanceFlushStrictMargin
    ) {
      messages.push(this.pendingEmissionMessage);
      this._emitDiagnostic({
        type: "commit",
        ref: `${this.pendingEmissionMessage.surah}:${this.pendingEmissionMessage.ayah}`,
        reason: "final_flush_pending_emit",
        confidence: this.pendingEmissionMessage.confidence,
      });
      this._emitDiagnostic({
        type: "pending_emission",
        action: "final_flush_emit",
        ref: `${this.pendingEmissionMessage.surah}:${this.pendingEmissionMessage.ayah}`,
        margin: Number.isFinite(this.pendingEmissionMargin)
          ? Math.round(this.pendingEmissionMargin * 1000) / 1000
          : null,
        fresh_samples: this.totalSamplesFed - this.samplesAtAdvance,
      });
      this._clearPendingEmission();
      this._exitTracking("final silence flush (pending emitted)");
      return messages;
    }

    const recognizedWords = text.split(" ").filter(Boolean);
    const resumeFrom = Math.max(this.trackingLastWordIdx, 0);
    let confirmedPendingEmission = false;
    let { matchedIndices } = alignPosition(
      recognizedWords,
      this.trackingVerseWords,
      resumeFrom,
      this.config.lookaheadWords,
    );
    const primaryMatchedIndices = matchedIndices.slice();

    const freshSamplesSinceAdvance = this.totalSamplesFed - this.samplesAtAdvance;
    const pendingHasEnoughDwell =
      !this.trackingPendingEmission ||
      this.trackingVerseWords.length > SHORT_PENDING_CONFIRM_MAX_WORDS ||
      freshSamplesSinceAdvance >= SHORT_PENDING_CONFIRM_MIN_SAMPLES;

    // Confirm pending emission only on primary word alignment from fresh audio.
    // Short verses need a small dwell so back-to-back final-word decodes do not
    // visibly skip over the verse the user just reached.
    if (
      this.trackingPendingEmission &&
      pendingHasEnoughDwell &&
      hasStrongPendingPrefixEvidence(matchedIndices, this.trackingVerseWords.length) &&
      this.totalSamplesFed > this.samplesAtAdvance
    ) {
      const pending = this.pendingEmissionMessage!;
      messages.push(pending);
      this._emitDiagnostic({
        type: "pending_emission",
        action: "confirmed",
        ref: `${pending.surah}:${pending.ayah}`,
        margin: Number.isFinite(this.pendingEmissionMargin)
          ? Math.round(this.pendingEmissionMargin * 1000) / 1000
          : null,
        fresh_samples: this.totalSamplesFed - this.samplesAtAdvance,
        matched_indices: matchedIndices,
      });
      this._clearPendingEmission();
      confirmedPendingEmission = true;
    }

    let acousticWord: number | null = null;
    if (matchedIndices.length === 0) {
      const acousticIdx = this._resolveTrackingAcousticWord(result);
      if (acousticIdx > this.trackingLastWordIdx) {
        acousticWord = acousticIdx;
        matchedIndices = [acousticIdx];
      }
    }

    let charWord: number | null = null;
    if (
      matchedIndices.length === 0 &&
      text.length >= 5 &&
      this.trackingVerseWords.length >= 10
    ) {
      const charWordIdx = this._charLevelProgress(text);
      if (charWordIdx > this.trackingLastWordIdx) {
        charWord = charWordIdx;
        matchedIndices = [charWordIdx];
      }
    }

    const advanced =
      matchedIndices.length > 0 &&
      matchedIndices[matchedIndices.length - 1] > this.trackingLastWordIdx;

    this._emitDiagnostic({
      type: "tracking_cycle",
      ref: `${this.trackingVerse.surah}:${this.trackingVerse.ayah}`,
      text_length: text.length,
      word_matches: primaryMatchedIndices.length,
      acoustic_word: acousticWord,
      char_word: charWord,
      advanced,
      final_flush: finalFlush,
      word_position: advanced
        ? matchedIndices[matchedIndices.length - 1] + 1
        : this.trackingLastWordIdx + 1,
      total_words: this.trackingVerseWords.length,
      coverage: Math.round(
        ((advanced
          ? matchedIndices[matchedIndices.length - 1] + 1
          : this.trackingLastWordIdx + 1) / this.trackingVerseWords.length) * 1000,
      ) / 1000,
      pending: this.trackingPendingEmission,
    });

    if (!advanced) {
      this.staleCycles++;
      if (this.staleCycles >= this.config.staleCycleLimit || finalFlush) {
        this._emitDiagnostic({
          type: "stale_exit",
          ref: `${this.trackingVerse.surah}:${this.trackingVerse.ayah}`,
          stale_cycles: this.staleCycles,
        });
        // Final-flush emit: if an advance was queued with strong acoustic
        // evidence (stricter than normal ADVANCE_RELATIVE_MARGIN), emit the
        // pending next-verse match before rolling back. Addresses the
        // multi_114 / user_ikhlas_2_3 "last verse dropped on silence" pattern.
        if (
          finalFlush &&
          this.trackingPendingEmission &&
          this.pendingEmissionMessage !== null &&
          this.pendingEmissionMargin < this.config.advanceFlushStrictMargin
        ) {
          messages.push(this.pendingEmissionMessage);
          this._emitDiagnostic({
            type: "commit",
            ref: `${this.pendingEmissionMessage.surah}:${this.pendingEmissionMessage.ayah}`,
            reason: "final_flush_pending_emit",
            confidence: this.pendingEmissionMessage.confidence,
          });
          this._emitDiagnostic({
            type: "pending_emission",
            action: "final_flush_emit",
            ref: `${this.pendingEmissionMessage.surah}:${this.pendingEmissionMessage.ayah}`,
            margin: Number.isFinite(this.pendingEmissionMargin)
              ? Math.round(this.pendingEmissionMargin * 1000) / 1000
              : null,
            fresh_samples: this.totalSamplesFed - this.samplesAtAdvance,
          });
          this._clearPendingEmission();
          // Do NOT rollback — the pending emission has been confirmed.
          this._exitTracking("final silence flush (pending emitted)");
        } else {
          this._rollbackWeakCommit(finalFlush ? "final silence flush" : "stale tracking");
          this._exitTracking(finalFlush ? "final silence flush" : "stale tracking");
        }
      }
      return messages;
    }

    const observedWordIdx = matchedIndices[matchedIndices.length - 1];
    const observedWordPos = observedWordIdx + 1;
    const totalWords = this.trackingVerseWords.length;
    const observedCoverage = Math.round((observedWordPos / totalWords) * 1000) / 1000;
    const completionWordCount = Math.ceil(
      totalWords * this.config.trackingCompletionCoverage,
    );
    const observedFinalWordReached = observedWordIdx >= totalWords - 1;

    if (
      this.trackingPendingEmission &&
      this.pendingEmissionMessage !== null &&
      !pendingHasEnoughDwell
    ) {
      this._emitDiagnostic({
        type: "advance_decision",
        from_ref: `${this.trackingVerse.surah}:${this.trackingVerse.ayah}`,
        to_ref: null,
        action: "blocked",
        reason: "pending confirmation dwell",
        word_position: observedWordPos,
        total_words: totalWords,
        coverage: observedCoverage,
        completion_target: completionWordCount,
        final_word: observedFinalWordReached,
        advance_ok: false,
        early_advance_ok: false,
        margin: null,
        normal_margin: this.config.advanceRelativeMargin,
        strict_margin: this.config.advanceFlushStrictMargin,
      });
      return messages;
    }

    this.staleCycles = 0;
    this.trackingProgressEstablished = true;
    this.trackingLastWordIdx = observedWordIdx;
    const wordPos = this.trackingLastWordIdx + 1;
    const coverage = Math.round((wordPos / totalWords) * 1000) / 1000;

    const completedEnough = wordPos >= completionWordCount;
    const finalWordReached =
      this.trackingLastWordIdx >= totalWords - 1;

    if (
      completedEnough &&
      this.trackingPendingEmission &&
      this.pendingEmissionMessage !== null &&
      pendingHasEnoughDwell
    ) {
      const pending = this.pendingEmissionMessage;
      messages.push(pending);
      this._emitDiagnostic({
        type: "pending_emission",
        action: "confirmed",
        ref: `${pending.surah}:${pending.ayah}`,
        margin: Number.isFinite(this.pendingEmissionMargin)
          ? Math.round(this.pendingEmissionMargin * 1000) / 1000
          : null,
        fresh_samples: this.totalSamplesFed - this.samplesAtAdvance,
        matched_indices: matchedIndices,
      });
      this._clearPendingEmission();
      confirmedPendingEmission = true;
    }

    if (!this.trackingPendingEmission) {
      messages.push({
        type: "word_progress",
        surah: this.trackingVerse.surah,
        ayah: this.trackingVerse.ayah,
        word_index: wordPos,
        total_words: totalWords,
        matched_indices: matchedIndices,
      });
    }

    if (completedEnough && confirmedPendingEmission && !finalWordReached) {
      this._emitDiagnostic({
        type: "pending_emission",
        action: "cascade_blocked",
        ref: `${this.trackingVerse.surah}:${this.trackingVerse.ayah}`,
        margin: null,
        fresh_samples: this.totalSamplesFed - this.samplesAtAdvance,
        matched_indices: matchedIndices,
      });
      this._emitDiagnostic({
        type: "advance_decision",
        from_ref: `${this.trackingVerse.surah}:${this.trackingVerse.ayah}`,
        to_ref: null,
        action: "blocked",
        reason: "pending confirmed before final word",
        word_position: wordPos,
        total_words: totalWords,
        coverage,
        completion_target: completionWordCount,
        final_word: finalWordReached,
        advance_ok: false,
        early_advance_ok: false,
        margin: null,
        normal_margin: this.config.advanceRelativeMargin,
        strict_margin: this.config.advanceFlushStrictMargin,
      });
    }
    if (completedEnough && (!confirmedPendingEmission || finalWordReached)) {
      if (!(this.lastCommitEvidence?.strong) && !this.trackingProgressEstablished) {
        this._emitDiagnostic({
          type: "advance_decision",
          from_ref: `${this.trackingVerse.surah}:${this.trackingVerse.ayah}`,
          to_ref: null,
          action: "blocked",
          reason: "weak commit evidence",
          word_position: wordPos,
          total_words: totalWords,
          coverage,
          completion_target: completionWordCount,
          final_word: finalWordReached,
          advance_ok: false,
          early_advance_ok: false,
          margin: null,
          normal_margin: this.config.advanceRelativeMargin,
          strict_margin: this.config.advanceFlushStrictMargin,
        });
        this._exitTracking("weak completion");
        return messages;
      }

      const currentRef: [number, number] = [
        this.trackingVerse.surah,
        this.trackingVerse.ayah,
      ];
      const currentIds = this.trackingVerse.phoneme_token_ids ?? [];
      const nextVerse = this.db.getNextVerse(currentRef[0], currentRef[1]);
      let advanceOk = true; // default: advance (preserves behavior when no acoustic data)
      let earlyAdvanceOk = completedEnough;
      // Evidence strength captured for optional final-flush emit. Defaults to
      // +Inf so the default-advance (no acoustic) path never passes the
      // stricter flush gate and still requires fresh-audio confirmation.
      let advanceMargin = Number.POSITIVE_INFINITY;
      const acoustic = this.lastTrackingResult?.acoustic;
      const nextIds = nextVerse?.phoneme_token_ids ?? [];

      if (nextVerse && acoustic && currentIds.length > 0 && nextIds.length > 0) {
        // Relative evidence gate: compare current verse suffix vs next verse prefix.
        const n = this.config.advancePrefixTokens;
        const suffixIds = currentIds.slice(-Math.min(n, currentIds.length));
        const prefixIds = nextIds.slice(0, Math.min(n, nextIds.length));

        const suffixScore = scoreCtcSequence(acoustic, suffixIds);
        const prefixScore = scoreCtcSequence(acoustic, prefixIds);

        if (
          !Number.isFinite(suffixScore) ||
          !Number.isFinite(prefixScore)
        ) {
          advanceOk = false;
        } else {
          advanceMargin = prefixScore - suffixScore;
          advanceOk = advanceMargin < this.config.advanceRelativeMargin;
          earlyAdvanceOk = earlyAdvanceOk ||
            advanceMargin < this.config.advanceFlushStrictMargin;
        }
      }

      if (!finalWordReached && !earlyAdvanceOk) {
        this._emitDiagnostic({
          type: "advance_decision",
          from_ref: `${this.trackingVerse.surah}:${this.trackingVerse.ayah}`,
          to_ref: nextVerse ? `${nextVerse.surah}:${nextVerse.ayah}` : null,
          action: "wait",
          reason: "coverage reached without final word or next-prefix evidence",
          word_position: wordPos,
          total_words: totalWords,
          coverage,
          completion_target: completionWordCount,
          final_word: finalWordReached,
          advance_ok: advanceOk,
          early_advance_ok: earlyAdvanceOk,
          margin: Number.isFinite(advanceMargin)
            ? Math.round(advanceMargin * 1000) / 1000
            : null,
          normal_margin: this.config.advanceRelativeMargin,
          strict_margin: this.config.advanceFlushStrictMargin,
        });
        return messages;
      }

      this.lastEmittedRef = currentRef;
      this.lastEmittedText = this.trackingVerse.phonemes_joined;
      this._exitTracking(finalWordReached ? "verse complete" : "near-complete with next prefix");

      if (nextVerse) {
        if (advanceOk) {
          // Snapshot state before advance for rollback on drop
          this.preAdvanceSnapshot = {
            emittedRef: this.lastEmittedRef ? [...this.lastEmittedRef] as [number, number] : null,
            emittedText: this.lastEmittedText,
            prevEmittedRef: this.prevEmittedRef ? [...this.prevEmittedRef] as [number, number] : null,
            prevEmittedText: this.prevEmittedText,
            commitEvidence: this.lastCommitEvidence ? { ...this.lastCommitEvidence } : null,
          };

          // Build verse_match but defer emission until fresh audio confirms
          this.pendingEmissionMessage = {
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
          };
          this.trackingPendingEmission = true;
          this.samplesAtAdvance = this.totalSamplesFed;
          this.pendingEmissionMargin = advanceMargin;
          this._emitDiagnostic({
            type: "advance_decision",
            from_ref: `${currentRef[0]}:${currentRef[1]}`,
            to_ref: `${nextVerse.surah}:${nextVerse.ayah}`,
            action: "armed",
            reason: finalWordReached
              ? "final word reached"
              : earlyAdvanceOk ? "completion coverage reached" : "next-prefix evidence",
            word_position: wordPos,
            total_words: totalWords,
            coverage,
            completion_target: completionWordCount,
            final_word: finalWordReached,
            advance_ok: advanceOk,
            early_advance_ok: earlyAdvanceOk,
            margin: Number.isFinite(advanceMargin)
              ? Math.round(advanceMargin * 1000) / 1000
              : null,
            normal_margin: this.config.advanceRelativeMargin,
            strict_margin: this.config.advanceFlushStrictMargin,
          });
          this._emitDiagnostic({
            type: "pending_emission",
            action: "armed",
            ref: `${nextVerse.surah}:${nextVerse.ayah}`,
            margin: Number.isFinite(advanceMargin)
              ? Math.round(advanceMargin * 1000) / 1000
              : null,
            fresh_samples: 0,
          });

          // Update state as before (tracking enters next verse)
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
          if (this.config.nextVerseEmitMode === "candidate_until_confirmed") {
            messages.push({
              type: "verse_candidate",
              candidates: [{
                surah: nextVerse.surah,
                ayah: nextVerse.ayah,
                ayah_end: null,
                confidence: 0.99,
                rank: 1,
                source: "tracking",
              }],
              stable: true,
              final_flush: false,
            });
          } else if (this.config.nextVerseEmitMode === "immediate_on_completion") {
            messages.push(this.pendingEmissionMessage);
            this._clearPendingEmission();
          }
          // After sustained auto-advances, degrade to weak so stale-exit
          // triggers rediscovery instead of persisting
          if (this.consecutiveAutoAdvances >= 5) {
            this.lastCommitEvidence = {
              ...this.lastCommitEvidence!,
              strong: false,
            };
          }
        }
        if (!advanceOk) {
          this._emitDiagnostic({
            type: "advance_decision",
            from_ref: `${currentRef[0]}:${currentRef[1]}`,
            to_ref: `${nextVerse.surah}:${nextVerse.ayah}`,
            action: "blocked",
            reason: "advance margin failed",
            word_position: wordPos,
            total_words: totalWords,
            coverage,
            completion_target: completionWordCount,
            final_word: finalWordReached,
            advance_ok: advanceOk,
            early_advance_ok: earlyAdvanceOk,
            margin: Number.isFinite(advanceMargin)
              ? Math.round(advanceMargin * 1000) / 1000
              : null,
            normal_margin: this.config.advanceRelativeMargin,
            strict_margin: this.config.advanceFlushStrictMargin,
          });
        }
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

    if (!finalFlush && this.newAudioCount < this.samplesForSeconds(this.config.discoveryTriggerSec)) {
      return messages;
    }
    this.newAudioCount = 0;
    this.cyclesSinceCommit++;

    const result = await this.transcribe(this.utteranceAudio.slice());
    const text = result.text.trim();
    if (!text || text.length < 5) {
      // Short-utterance rescue: use CTC rescoring against short-verse candidates
      if (result.acoustic && (result.tokenIds?.length ?? 0) >= 2 && this.cyclesSinceCommit > 1) {
        const shortCandidates = this.db.getShortVerseCandidates();
        if (shortCandidates.length > 0) {
          const scored = scoreCtcCandidates(
            result.acoustic,
            shortCandidates.map((c) => ({ ids: c.phoneme_token_ids, meta: c })),
          );
          const feasible = scored.filter((s) => s.feasible);
          if (feasible.length >= 2) {
            const margin = feasible[1].acousticScore - feasible[0].acousticScore;
            if (margin >= this.config.acousticClearMargin) {
              const best = feasible[0].meta;
              const verse = this.db.getVerse(best.surah, best.ayah);
              if (verse) {
                const ref: [number, number] = [best.surah, best.ayah];
                const key = refKey(best.surah, best.ayah);
                // Skip if same as last emitted
                if (
                  !this.lastEmittedRef ||
                  this.lastEmittedRef[0] !== ref[0] ||
                  this.lastEmittedRef[1] !== ref[1]
                ) {
                  const confidence = Math.min(0.85, 0.5 + margin);
                  messages.push({
                    type: "verse_match",
                    surah: best.surah,
                    ayah: best.ayah,
                    verse_text: verse.text_uthmani,
                    surah_name: verse.surah_name,
                    confidence: Math.round(confidence * 100) / 100,
                    surrounding_verses: getSurroundingVerses(this.db, best.surah, best.ayah),
                  });
                  this.prevEmittedRef = this.lastEmittedRef;
                  this.prevEmittedText = this.lastEmittedText;
                  this.lastEmittedRef = ref;
                  this.lastEmittedText = verse.phonemes_joined;
                  this.lastCommitEvidence = { confidence, acousticMargin: margin, strong: margin >= 0.3 };
                  this.pendingLeader = null;
                  this.cyclesSinceCommit = 0;
                  this.consecutiveAutoAdvances = 0;
                  this._emitDiagnostic({
                    type: "commit",
                    ref: key,
                    reason: "short_rescue",
                    confidence,
                    origin: "short_rescue",
                    acoustic_margin: Math.round(margin * 1000) / 1000,
                  });
                  this._enterTracking(verse);
                  return messages;
                }
              }
            }
          }
        }
      }
      this._emitDiagnostic({
        type: "silence_skip",
        mode: "discovery",
        reason: "transcript too short",
      });
      return messages;
    }

    if (this.lastEmittedText && this.lastCommitEvidence?.strong) {
      const residual = partialRatio(text, this.lastEmittedText);
      const textChars = text.replace(/\s+/g, "").length;
      const emittedChars = this.lastEmittedText.replace(/\s+/g, "").length;
      const looksLikeLeftover = textChars <= Math.ceil(emittedChars * 1.15);
      if (residual > 0.7 && looksLikeLeftover && !finalFlush) {
        this._emitDiagnostic({
          type: "silence_skip",
          mode: "discovery",
          reason: `residual=${residual.toFixed(3)}`,
        });
        return messages;
      }
    }

    const championMatch = result.championMatch ?? null;
    const match = championMatch ?? this.db.matchVerse(
      text,
      RAW_TRANSCRIPT_THRESHOLD,
      DISCOVERY_MAX_SPAN,
      this.lastEmittedRef,
      5,
    );
    // Expand candidate set when text match is unreliable
    const textConfidenceLow = !match || match.score < this.config.verseMatchThreshold + 0.10;
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
        acousticMargin: Math.round(entry.acousticMargin * 1000) / 1000,
        lengthFit: Math.round(entry.lengthFit * 1000) / 1000,
        fusion: Math.round(entry.fusionScore * 1000) / 1000,
        feasible: entry.feasible,
      })),
    });

    let acousticMargin = 0;
    let lengthFit = 1;
    let effectiveMatch = match;
    let effectiveScore = match?.score ?? 0;
    const fusionBest = ranked[0] ?? null;

    if (fusionBest) {
      acousticMargin = fusionBest.acousticMargin;
      lengthFit = fusionBest.lengthFit;
    }

    if (match && fusionBest) {
      const matchKey = refKey(match.surah, match.ayah, match.ayah_end);
      const fusionKey = refKey(
        fusionBest.candidate.surah,
        fusionBest.candidate.ayah,
        fusionBest.candidate.ayah_end,
      );
      const fusionGap = fusionBest.fusionScore - match.score;
      if (fusionKey === matchKey) {
        effectiveScore = Math.max(
          effectiveScore,
          fusionBest.fusionScore,
          fusionBest.candidate.stage_a_score,
        );
      }
      const shouldOverride =
        !championMatch &&
        fusionKey !== matchKey &&
        (
          match.score < this.config.verseMatchThreshold + 0.10 ||
          textConfidenceLow ||
          fusionGap >= DISCOVERY_FUSION_SELECTION_GAP ||
          (fusionBest.candidate.kind === "span" && fusionBest.lengthFit >= 0.7)
        );

      if (shouldOverride) {
        effectiveMatch = {
          surah: fusionBest.candidate.surah,
          ayah: fusionBest.candidate.ayah,
          ayah_end: fusionBest.candidate.ayah_end,
          text: fusionBest.candidate.text,
          phonemes_joined: fusionBest.candidate.phonemes_joined,
          score: Math.max(
            match.score,
            fusionBest.fusionScore,
            fusionBest.candidate.stage_a_score,
            0.5,
          ),
          raw_score: fusionBest.candidate.raw_score,
          bonus: fusionBest.candidate.bonus,
        };
        effectiveScore = effectiveMatch.score;
        acousticMargin = fusionBest.acousticMargin;
        lengthFit = fusionBest.lengthFit;
      }
    } else if (!match && fusionBest) {
      effectiveMatch = {
        surah: fusionBest.candidate.surah,
        ayah: fusionBest.candidate.ayah,
        ayah_end: fusionBest.candidate.ayah_end,
        text: fusionBest.candidate.text,
        phonemes_joined: fusionBest.candidate.phonemes_joined,
        score: Math.max(fusionBest.fusionScore, fusionBest.candidate.stage_a_score),
        raw_score: fusionBest.candidate.raw_score,
        bonus: fusionBest.candidate.bonus,
      };
      effectiveScore = effectiveMatch.score;
    }

    if (effectiveMatch && fusionBest && this.lastEmittedRef && !finalFlush) {
      const nextAyah = this.lastEmittedRef[1] + 1;
      const effectiveEnd =
        effectiveMatch.ayah_end && effectiveMatch.ayah_end > effectiveMatch.ayah
          ? effectiveMatch.ayah_end
          : effectiveMatch.ayah;
      const top = fusionBest.candidate;
      const broadMatchCoversNext =
        effectiveMatch.surah === this.lastEmittedRef[0] &&
        effectiveMatch.ayah < nextAyah &&
        effectiveEnd >= nextAyah;
      const topIsNearbyForwardContinuation =
        top.surah === this.lastEmittedRef[0] &&
        top.ayah > nextAyah &&
        top.ayah <= this.lastEmittedRef[1] + 3;
      const topClearlyBetter =
        (fusionBest.feasible || !result.acoustic) &&
        fusionBest.lengthFit >= 0.6 &&
        fusionBest.fusionScore >= effectiveScore + 0.05;

      if (broadMatchCoversNext && topIsNearbyForwardContinuation && topClearlyBetter) {
        effectiveMatch = {
          surah: top.surah,
          ayah: top.ayah,
          ayah_end: top.ayah_end,
          text: top.text,
          phonemes_joined: top.phonemes_joined,
          score: Math.max(fusionBest.fusionScore, top.stage_a_score),
          raw_score: top.raw_score,
          bonus: top.bonus,
        };
        effectiveScore = effectiveMatch.score;
        acousticMargin = fusionBest.acousticMargin;
        lengthFit = fusionBest.lengthFit;
      }
    }

    if (effectiveMatch) {
      const effectiveKey = refKey(
        effectiveMatch.surah,
        effectiveMatch.ayah,
        effectiveMatch.ayah_end,
      );
      const selectedRanked = ranked.find(
        (entry) =>
          refKey(
            entry.candidate.surah,
            entry.candidate.ayah,
            entry.candidate.ayah_end,
          ) === effectiveKey,
      );
      if (selectedRanked) {
        acousticMargin = selectedRanked.acousticMargin;
        lengthFit = selectedRanked.lengthFit;
      }
    }

    if (effectiveMatch && this.lastEmittedRef && !finalFlush) {
      const nextAyah = this.lastEmittedRef[1] + 1;
      const effectiveEnd =
        effectiveMatch.ayah_end && effectiveMatch.ayah_end > effectiveMatch.ayah
          ? effectiveMatch.ayah_end
          : effectiveMatch.ayah;
      const shouldRebaseToNext =
        effectiveMatch.surah === this.lastEmittedRef[0] &&
        effectiveMatch.ayah !== nextAyah &&
        effectiveMatch.ayah <= nextAyah &&
        effectiveEnd >= nextAyah;
      const nextVerse = this.db.getVerse(effectiveMatch.surah, nextAyah);
      if (shouldRebaseToNext && nextVerse) {
        this._emitDiagnostic({
          type: "advance_decision",
          from_ref: refKey(
            effectiveMatch.surah,
            effectiveMatch.ayah,
            effectiveMatch.ayah_end,
          ),
          to_ref: `${nextVerse.surah}:${nextVerse.ayah}`,
          action: "blocked",
          reason: "live span rebased to next ayah",
          word_position: 0,
          total_words: 0,
          coverage: 0,
          completion_target: 0,
          final_word: false,
          advance_ok: false,
          early_advance_ok: false,
          margin: null,
          normal_margin: this.config.advanceRelativeMargin,
          strict_margin: this.config.advanceFlushStrictMargin,
        });
        effectiveMatch = {
          surah: nextVerse.surah,
          ayah: nextVerse.ayah,
          ayah_end: null,
          text: nextVerse.text_uthmani,
          phonemes_joined: nextVerse.phonemes_joined,
          score: effectiveScore,
          raw_score: effectiveScore,
          bonus: 0,
        };
      }
    }

    const threshold = this.lastEmittedRef ? this.config.verseMatchThreshold : this.config.firstMatchThreshold;

    if (effectiveMatch && effectiveScore >= threshold) {
      const key = refKey(effectiveMatch.surah, effectiveMatch.ayah, effectiveMatch.ayah_end);
      this.pendingLeader =
        this.pendingLeader?.key === key
          ? { key, count: this.pendingLeader.count + 1 }
          : { key, count: 1 };

      const isContinuation = this._isContinuation(effectiveMatch.surah, effectiveMatch.ayah);
      const clearMargin =
        lengthFit >= 0.6 &&
        acousticMargin >=
        (isContinuation ? this.config.acousticContinuationMargin : this.config.acousticClearMargin);
      const repeatedLeader =
        (this.pendingLeader?.count ?? 0) >= this.config.discoveryRepeatCycles;
      const candidateMessage = this._candidateMessage(
        effectiveMatch,
        effectiveScore,
        ranked,
        repeatedLeader || finalFlush,
        finalFlush,
      );
      if (candidateMessage) {
        messages.push(candidateMessage);
      }

      // Anti-cascade: shortly after a commit, require higher score for
      // non-continuation jumps to prevent false positives
      let effectivelyBlocked = false;
      if (this.lastEmittedRef && !isContinuation && !finalFlush) {
        effectivelyBlocked = true;
        this._emitDiagnostic({
          type: "advance_decision",
          from_ref: this.lastEmittedRef
            ? `${this.lastEmittedRef[0]}:${this.lastEmittedRef[1]}`
            : "none",
          to_ref: refKey(
            effectiveMatch.surah,
            effectiveMatch.ayah,
            effectiveMatch.ayah_end,
          ),
          action: "blocked",
          reason: "live non-continuation discovery blocked",
          word_position: 0,
          total_words: 0,
          coverage: 0,
          completion_target: 0,
          final_word: false,
          advance_ok: false,
          early_advance_ok: false,
          margin: null,
          normal_margin: this.config.advanceRelativeMargin,
          strict_margin: this.config.advanceFlushStrictMargin,
        });
      }
      if (
        !isContinuation &&
        this.lastEmittedRef &&
        this.cyclesSinceCommit <= 2
      ) {
        if (effectiveScore < this.config.nonContinuationJumpThreshold && !repeatedLeader) {
          effectivelyBlocked = true;
        }
      }

      // On final flush, commit if score is above threshold (no repeat needed)
      const finalFlushCommit = finalFlush && effectiveScore >= threshold;

      // Decode-stability gate: deny single-cycle clearMargin commits when the
      // current decode hasn't stabilized vs the previous cycle. Forces
      // commits onto the repeated-leader path (≥ DISCOVERY_REPEAT_CYCLES)
      // when the underlying decode is volatile.
      let clearMarginAllowed = clearMargin;
      if (
        DECODE_STABILITY_GATE &&
        this.config.decodeStabilityEnabled &&
        clearMargin &&
        !isContinuation
      ) {
        const prev = this.lastRawPhonemes;
        const stable =
          prev !== null && prev.length > 0 &&
          levRatio(prev, result.rawPhonemes) >= this.config.decodeStabilityRatio;
        if (!stable) clearMarginAllowed = false;
      }

      if (!effectivelyBlocked && (clearMarginAllowed || repeatedLeader || finalFlushCommit)) {
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
          effectiveScore,
          Math.min(0.99, 0.45 + acousticMargin + lengthFit * 0.2),
        );
        const selectedKey = refKey(
          effectiveMatch.surah,
          effectiveMatch.ayah,
          effectiveMatch.ayah_end,
        );
        const selectedRank = ranked.findIndex(
          (entry) =>
            refKey(
              entry.candidate.surah,
              entry.candidate.ayah,
              entry.candidate.ayah_end,
            ) === selectedKey,
        );
        const selectedDiagnostic = selectedRank >= 0 ? ranked[selectedRank] : null;
        const topDiagnostic = ranked[0] ?? null;

        messages.push({
          type: "verse_match",
          surah: effectiveMatch.surah,
          ayah: effectiveMatch.ayah,
          verse_text: verse?.text_uthmani ?? effectiveMatch.text ?? "",
          surah_name: verse?.surah_name ?? "",
          confidence: Math.round(confidence * 100) / 100,
          surrounding_verses: surrounding,
        });

        const ayahEnd = effectiveMatch.ayah_end;
        const liveSpanCollapsed = Boolean(
          ayahEnd && ayahEnd > effectiveMatch.ayah && !finalFlush,
        );
        const committedAyahEnd =
          ayahEnd && ayahEnd > effectiveMatch.ayah && finalFlush
            ? ayahEnd
            : effectiveMatch.ayah;
        if (liveSpanCollapsed) {
          this._emitDiagnostic({
            type: "advance_decision",
            from_ref: refKey(effectiveMatch.surah, effectiveMatch.ayah, ayahEnd),
            to_ref: `${effectiveMatch.surah}:${effectiveMatch.ayah}`,
            action: "blocked",
            reason: "live span collapsed to first ayah",
            word_position: 0,
            total_words: 0,
            coverage: 0,
            completion_target: 0,
            final_word: false,
            advance_ok: false,
            early_advance_ok: false,
            margin: null,
            normal_margin: this.config.advanceRelativeMargin,
            strict_margin: this.config.advanceFlushStrictMargin,
          });
        }

        // Only final flush commits every verse in a span. During live recitation,
        // committing the whole span jumps the UI past the ayah the user just began.
        if (committedAyahEnd > effectiveMatch.ayah) {
          for (let a = effectiveMatch.ayah + 1; a <= committedAyahEnd; a++) {
            const spanVerse = this.db.getVerse(effectiveMatch.surah, a);
            if (spanVerse) {
              messages.push({
                type: "verse_match",
                surah: spanVerse.surah,
                ayah: spanVerse.ayah,
                verse_text: spanVerse.text_uthmani,
                surah_name: spanVerse.surah_name,
                confidence: Math.round(confidence * 100) / 100,
                surrounding_verses: getSurroundingVerses(this.db, spanVerse.surah, spanVerse.ayah),
              });
            }
          }
        }

        this.prevEmittedRef = this.lastEmittedRef;
        this.prevEmittedText = this.lastEmittedText;
        const effectiveRef: [number, number] = [
          effectiveMatch.surah,
          committedAyahEnd,
        ];
        this.lastEmittedRef = effectiveRef;
        const lastSpanVerse =
          committedAyahEnd > effectiveMatch.ayah
            ? this.db.getVerse(effectiveMatch.surah, committedAyahEnd)
            : verse;
        this.lastEmittedText =
          lastSpanVerse?.phonemes_joined ?? effectiveMatch.phonemes_joined ?? verse?.phonemes_joined ?? "";
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
          ref: liveSpanCollapsed ? `${effectiveMatch.surah}:${effectiveMatch.ayah}` : key,
          reason: liveSpanCollapsed
            ? "live_span_collapsed"
            : clearMargin ? "acoustic_margin" : "repeat_leader",
          confidence: Math.round(confidence * 1000) / 1000,
          origin: "discovery",
          selected_rank: selectedRank >= 0 ? selectedRank + 1 : null,
          selected_feasible: selectedDiagnostic?.feasible ?? null,
          selected_fusion: selectedDiagnostic
            ? Math.round(selectedDiagnostic.fusionScore * 1000) / 1000
            : null,
          top_ref: topDiagnostic
            ? refKey(
                topDiagnostic.candidate.surah,
                topDiagnostic.candidate.ayah,
                topDiagnostic.candidate.ayah_end,
              )
            : null,
          top_fusion: topDiagnostic
            ? Math.round(topDiagnostic.fusionScore * 1000) / 1000
            : null,
          effective_score: Math.round(effectiveScore * 1000) / 1000,
          threshold,
          acoustic_margin: Math.round(acousticMargin * 1000) / 1000,
          length_fit: Math.round(lengthFit * 1000) / 1000,
          clear_margin: clearMarginAllowed,
          repeated_leader: repeatedLeader,
          final_flush_commit: finalFlushCommit,
          is_continuation: isContinuation,
        });

        // Live spans track the first committed ayah; final-flush spans track the last.
        const trackVerse = lastSpanVerse ?? verse;
        if (trackVerse) {
          this._enterTracking(trackVerse);
        } else {
          this._retainTailAfterCommit();
        }
      } else {
        messages.push({
          type: "raw_transcript",
          text,
          confidence: Math.round(effectiveScore * 100) / 100,
        });
      }
    } else {
      const score = effectiveMatch ? Math.round(effectiveScore * 100) / 100 : 0;
      messages.push({
        type: "raw_transcript",
        text,
        confidence: score,
      });
    }

    this.lastRawPhonemes = result.rawPhonemes;
    return messages;
  }

  private _candidateMessage(
    effectiveMatch: {
      surah: number;
      ayah: number;
      ayah_end?: number | null;
    } | null,
    effectiveScore: number,
    ranked: RankedCandidate[],
    stable: boolean,
    finalFlush: boolean,
  ): VerseCandidateMessage | null {
    const candidates: VerseCandidate[] = [];
    const seen = new Set<string>();
    const addCandidate = (
      surah: number,
      ayah: number,
      ayahEnd: number | null | undefined,
      confidence: number,
    ) => {
      const key = refKey(surah, ayah, ayahEnd);
      if (seen.has(key)) return;
      seen.add(key);
      candidates.push({
        surah,
        ayah,
        ayah_end: ayahEnd ?? null,
        confidence: Math.round(Math.max(0, Math.min(1, confidence)) * 100) / 100,
        rank: candidates.length + 1,
        source: "discovery",
      });
    };

    if (effectiveMatch) {
      addCandidate(
        effectiveMatch.surah,
        effectiveMatch.ayah,
        effectiveMatch.ayah_end,
        effectiveScore,
      );
    }

    for (const entry of ranked.slice(0, 4)) {
      addCandidate(
        entry.candidate.surah,
        entry.candidate.ayah,
        entry.candidate.ayah_end,
        entry.fusionScore,
      );
    }

    if (candidates.length === 0) return null;
    return {
      type: "verse_candidate",
      candidates,
      stable,
      final_flush: finalFlush,
    };
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
    const stable = chooseLongestStablePrefix(scored, this.config.trackingPrefixTolerance);
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
          fusionScore: candidate.stage_a_score,
        }))
        .sort((a, b) => b.candidate.stage_a_score - a.candidate.stage_a_score);
    }

    const observedLength = Math.max(result.tokenIds?.length ?? 0, 1);
    const observedWords = result.text.trim().split(/\s+/).filter(Boolean).length;
    const observedChars = result.text.replace(/\s+/g, "").length;
    const textWeak =
      observedWords <= DISCOVERY_LOW_CONFIDENCE_WORDS ||
      observedChars <= DISCOVERY_LOW_CONFIDENCE_CHARS;
    const textWeight = textWeak
      ? DISCOVERY_FUSION_LOW_TEXT_WEIGHT
      : DISCOVERY_FUSION_TEXT_WEIGHT;
    const acousticWeight = textWeak
      ? DISCOVERY_FUSION_LOW_ACOUSTIC_WEIGHT
      : DISCOVERY_FUSION_ACOUSTIC_WEIGHT;
    const lengthWeight = textWeak
      ? DISCOVERY_FUSION_LOW_LENGTH_WEIGHT
      : DISCOVERY_FUSION_LENGTH_WEIGHT;
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
      const fusionScore = Math.min(
        1,
        entry.meta.stage_a_score * textWeight +
          acousticFit * acousticWeight +
          lengthFit * lengthWeight,
      );

      return {
        candidate: entry.meta,
        acousticScore: entry.acousticScore,
        acousticMargin:
          (scored[idx + 1]?.acousticScore ?? entry.acousticScore) - entry.acousticScore,
        feasible: entry.feasible,
        lengthFit,
        fusionScore,
      };
    });

    ranked.sort((a, b) => {
      if (b.fusionScore !== a.fusionScore) {
        return b.fusionScore - a.fusionScore;
      }
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
    // Full state rollback if pending emission was never confirmed
    if (this.trackingPendingEmission && this.preAdvanceSnapshot) {
      this.lastEmittedRef = this.preAdvanceSnapshot.emittedRef;
      this.lastEmittedText = this.preAdvanceSnapshot.emittedText;
      this.prevEmittedRef = this.preAdvanceSnapshot.prevEmittedRef;
      this.prevEmittedText = this.preAdvanceSnapshot.prevEmittedText;
      this.lastCommitEvidence = this.preAdvanceSnapshot.commitEvidence;
      this.consecutiveAutoAdvances = 0;
    }
    this._clearPendingEmission();

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
      const keepSeconds = this.trackingPendingEmission
        ? this.config.tailAfterPendingAdvanceSec
        : this.config.tailAfterCommitSec;
      const keepAmount = this.samplesForSeconds(keepSeconds);
      const keepSamples = Math.min(this.utteranceAudio.length, keepAmount);
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
    this.lastRawPhonemes = null;
    this.hypothesis.reset();
  }

  private _isContinuation(surah: number, ayah: number): boolean {
    if (!this.lastEmittedRef) return false;
    return (
      surah === this.lastEmittedRef[0] &&
      ayah >= this.lastEmittedRef[1] + 1 &&
      ayah <= this.lastEmittedRef[1] + 3
    );
  }

  private _clearPendingEmission(): void {
    this.trackingPendingEmission = false;
    this.pendingEmissionMessage = null;
    this.pendingEmissionMargin = Number.POSITIVE_INFINITY;
    this.preAdvanceSnapshot = null;
  }

  private _emitDiagnostic(event: TrackerDiagnosticEvent): void {
    this.options.onDiagnostic?.(event);
  }

  private samplesForSeconds(seconds: number): number {
    return Math.max(1, Math.round(SAMPLE_RATE * seconds));
  }
}
