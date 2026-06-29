// ---------------------------------------------------------------------------
// Message protocol (Worker <-> Main Thread)
// ---------------------------------------------------------------------------
export interface VerseMatchMessage {
  type: "verse_match";
  surah: number;
  ayah: number;
  verse_text: string;
  surah_name: string;
  confidence: number;
  surrounding_verses: SurroundingVerse[];
}

export interface VerseCandidate {
  surah: number;
  ayah: number;
  ayah_end?: number | null;
  confidence: number;
  rank: number;
  source: "discovery" | "tracking";
}

export interface VerseCandidateMessage {
  type: "verse_candidate";
  candidates: VerseCandidate[];
  stable: boolean;
  final_flush: boolean;
}

export interface FinalSequenceVerse {
  surah: number;
  ayah: number;
  confidence: number;
}

export interface FinalSequenceMessage {
  type: "final_sequence";
  verses: FinalSequenceVerse[];
  confidence: number;
}

export interface WordProgressMessage {
  type: "word_progress";
  surah: number;
  ayah: number;
  word_index: number;
  total_words: number;
  matched_indices: number[];
}

export interface RawTranscriptMessage {
  type: "raw_transcript";
  text: string;
  confidence: number;
}

export interface DebugMessage {
  type: "debug";
  event: string;
  at: number;
  data: Record<string, unknown>;
}

export interface SurroundingVerse {
  surah: number;
  ayah: number;
  text: string;
  is_current: boolean;
}

// Main -> Worker
export type WorkerInbound =
  | { type: "init" }
  | { type: "audio"; samples: Float32Array }
  | { type: "reset" }
  | { type: "set_debug"; enabled: boolean }
  | { type: "set_config"; config: StreamingConfig };

// Worker -> Main
export type WorkerOutbound =
  | { type: "loading"; percent: number }
  | { type: "loading_status"; message: string }
  | { type: "ready" }
  | { type: "error"; message: string }
  | VerseMatchMessage
  | VerseCandidateMessage
  | FinalSequenceMessage
  | WordProgressMessage
  | RawTranscriptMessage
  | DebugMessage;

// ---------------------------------------------------------------------------
// Quran data (from quran.json)
// ---------------------------------------------------------------------------
export interface QuranVerse {
  surah: number;
  ayah: number;
  text_uthmani: string;
  text_clean?: string;
  surah_name: string;
  surah_name_en: string;
  /** Legacy tracker adapter fields. For the text CTC pipeline these contain normalized Arabic text/tokens. */
  phonemes: string;
  phonemes_joined: string;
  phoneme_tokens?: string[];
  phoneme_tokens_no_bsm?: string[] | null;
  phoneme_token_ids?: number[];
  phoneme_token_ids_no_bsm?: number[] | null;
  word_token_ends?: number[];
  phonemes_joined_no_bsm?: string | null;
  phonemes_joined_ns?: string;              // no-space version (pre-computed)
  phonemes_joined_no_bsm_ns?: string | null; // no-space no-bismillah (pre-computed)
  phoneme_words: string[];
}

export interface SurahData {
  surah: number;
  surah_name: string;
  surah_name_en: string;
  verses: { ayah: number; text_uthmani: string }[];
}

// ---------------------------------------------------------------------------
// Constants (matching server.py exactly)
// ---------------------------------------------------------------------------
export const SAMPLE_RATE = 16000;
export const TRIGGER_SECONDS = 2.0;
export const TRIGGER_SAMPLES = SAMPLE_RATE * TRIGGER_SECONDS;
export const MAX_WINDOW_SECONDS = 30.0;
export const MAX_WINDOW_SAMPLES = SAMPLE_RATE * MAX_WINDOW_SECONDS;
export const SILENCE_RMS_THRESHOLD = 0.005;
export const UTTERANCE_FINAL_SILENCE_SECONDS = 1.2;
export const UTTERANCE_FINAL_SILENCE_SAMPLES =
  SAMPLE_RATE * UTTERANCE_FINAL_SILENCE_SECONDS;

export const VERSE_MATCH_THRESHOLD = 0.45;
export const FIRST_MATCH_THRESHOLD = 0.75;
export const RAW_TRANSCRIPT_THRESHOLD = 0.25;
export const SURROUNDING_CONTEXT = 2;
export const DISCOVERY_REPEAT_CYCLES = 2;
export const DISCOVERY_TOP_SINGLE_CANDIDATES = 64;
export const DISCOVERY_TOP_SURAHS = 5;
export const DISCOVERY_MAX_SPAN = 4;
export const ACOUSTIC_CLEAR_MARGIN = 0.12;
export const ACOUSTIC_CONTINUATION_MARGIN = 0.08;
export const NON_CONTINUATION_JUMP_THRESHOLD = 0.65;
export const ACOUSTIC_OVERRIDE_TEXT_THRESHOLD = 0.55;
export const ACOUSTIC_OVERRIDE_MIN_MARGIN = 0.25;
export const DISCOVERY_EXPANDED_CANDIDATES = 200;
export const DISCOVERY_LOW_CONFIDENCE_WORDS = 4;
export const DISCOVERY_LOW_CONFIDENCE_CHARS = 18;
export const DISCOVERY_FUSION_TEXT_WEIGHT = 0.6;
export const DISCOVERY_FUSION_ACOUSTIC_WEIGHT = 0.25;
export const DISCOVERY_FUSION_LENGTH_WEIGHT = 0.15;
export const DISCOVERY_FUSION_LOW_TEXT_WEIGHT = 0.45;
export const DISCOVERY_FUSION_LOW_ACOUSTIC_WEIGHT = 0.4;
export const DISCOVERY_FUSION_LOW_LENGTH_WEIGHT = 0.15;
export const DISCOVERY_FUSION_SELECTION_GAP = 0.08;

export const TRACKING_TRIGGER_SECONDS = 0.5;
export const TRACKING_TRIGGER_SAMPLES = SAMPLE_RATE * TRACKING_TRIGGER_SECONDS;
export const TRACKING_SILENCE_TIMEOUT = 4.0;
export const TRACKING_SILENCE_SAMPLES = SAMPLE_RATE * TRACKING_SILENCE_TIMEOUT;
export const TRACKING_MAX_WINDOW_SECONDS = 30.0;
export const TRACKING_MAX_WINDOW_SAMPLES =
  SAMPLE_RATE * TRACKING_MAX_WINDOW_SECONDS;
export const STALE_CYCLE_LIMIT = 4;
export const LOOKAHEAD = 5;
export const TRACKING_PREFIX_TOLERANCE = 0.12;
export const TRACKING_WEAK_COMMIT_CONFIDENCE = 0.6;
export const TRACKING_COMPLETION_COVERAGE = 0.95;
export const ADVANCE_RELATIVE_MARGIN = 3.0;
export const ADVANCE_PREFIX_TOKENS = 15;
// Stricter margin required to emit a pending next-verse advance at final flush,
// where we can't wait for fresh-audio confirmation. Must be smaller (= more
// demanding) than ADVANCE_RELATIVE_MARGIN.
export const ADVANCE_FLUSH_STRICT_MARGIN = 0.5;

export type NextVerseEmitMode =
  | "deferred_confirm"
  | "candidate_until_confirmed"
  | "immediate_on_completion";

export interface StreamingConfig {
  audioChunkMs: number;
  discoveryTriggerSec: number;
  trackingTriggerSec: number;
  discoveryMaxWindowSec: number;
  trackingMaxWindowSec: number;
  tailAfterCommitSec: number;
  tailAfterPendingAdvanceSec: number;
  finalSilenceSec: number;
  silenceRmsThreshold: number;
  firstMatchThreshold: number;
  verseMatchThreshold: number;
  discoveryRepeatCycles: number;
  acousticClearMargin: number;
  acousticContinuationMargin: number;
  decodeStabilityEnabled: boolean;
  decodeStabilityRatio: number;
  nonContinuationJumpThreshold: number;
  nextVerseEmitMode: NextVerseEmitMode;
  trackingCompletionCoverage: number;
  trackingPrefixTolerance: number;
  lookaheadWords: number;
  staleCycleLimit: number;
  trackingSilenceTimeoutSec: number;
  advanceRelativeMargin: number;
  advancePrefixTokens: number;
  advanceFlushStrictMargin: number;
}

export const CONSERVATIVE_STREAMING_CONFIG: StreamingConfig = {
  audioChunkMs: 300,
  discoveryTriggerSec: TRIGGER_SECONDS,
  trackingTriggerSec: TRACKING_TRIGGER_SECONDS,
  discoveryMaxWindowSec: MAX_WINDOW_SECONDS,
  trackingMaxWindowSec: TRACKING_MAX_WINDOW_SECONDS,
  tailAfterCommitSec: TRIGGER_SECONDS,
  tailAfterPendingAdvanceSec: TRACKING_TRIGGER_SECONDS,
  finalSilenceSec: UTTERANCE_FINAL_SILENCE_SECONDS,
  silenceRmsThreshold: SILENCE_RMS_THRESHOLD,
  firstMatchThreshold: FIRST_MATCH_THRESHOLD,
  verseMatchThreshold: VERSE_MATCH_THRESHOLD,
  discoveryRepeatCycles: DISCOVERY_REPEAT_CYCLES,
  acousticClearMargin: ACOUSTIC_CLEAR_MARGIN,
  acousticContinuationMargin: ACOUSTIC_CONTINUATION_MARGIN,
  decodeStabilityEnabled: true,
  decodeStabilityRatio: 0.70,
  nonContinuationJumpThreshold: NON_CONTINUATION_JUMP_THRESHOLD,
  nextVerseEmitMode: "deferred_confirm",
  trackingCompletionCoverage: TRACKING_COMPLETION_COVERAGE,
  trackingPrefixTolerance: TRACKING_PREFIX_TOLERANCE,
  lookaheadWords: LOOKAHEAD,
  staleCycleLimit: STALE_CYCLE_LIMIT,
  trackingSilenceTimeoutSec: TRACKING_SILENCE_TIMEOUT,
  advanceRelativeMargin: ADVANCE_RELATIVE_MARGIN,
  advancePrefixTokens: ADVANCE_PREFIX_TOKENS,
  advanceFlushStrictMargin: ADVANCE_FLUSH_STRICT_MARGIN,
};

export const BALANCED_STREAMING_CONFIG: StreamingConfig = {
  ...CONSERVATIVE_STREAMING_CONFIG,
  audioChunkMs: 150,
  trackingTriggerSec: 0.25,
  trackingMaxWindowSec: 12,
  tailAfterCommitSec: 0.75,
  nextVerseEmitMode: "candidate_until_confirmed",
  trackingCompletionCoverage: 0.82,
  acousticContinuationMargin: 0.06,
  advanceRelativeMargin: 3.5,
};

export const AGGRESSIVE_ADVANCE_STREAMING_CONFIG: StreamingConfig = {
  ...BALANCED_STREAMING_CONFIG,
  audioChunkMs: 150,
  discoveryTriggerSec: 1.5,
  trackingTriggerSec: 0.25,
  trackingMaxWindowSec: 12,
  tailAfterCommitSec: 0.75,
  trackingCompletionCoverage: 0.85,
  discoveryRepeatCycles: 1,
  acousticContinuationMargin: 0.04,
  advanceRelativeMargin: 4.0,
  advanceFlushStrictMargin: 1.0,
};

export type StreamingPresetName = "conservative" | "balanced" | "aggressiveAdvance";

export const STREAMING_PRESETS: Record<StreamingPresetName, StreamingConfig> = {
  conservative: CONSERVATIVE_STREAMING_CONFIG,
  balanced: BALANCED_STREAMING_CONFIG,
  aggressiveAdvance: AGGRESSIVE_ADVANCE_STREAMING_CONFIG,
};

export const DEFAULT_STREAMING_CONFIG = BALANCED_STREAMING_CONFIG;

export function normalizeStreamingConfig(
  partial: Partial<StreamingConfig> | null | undefined,
): StreamingConfig {
  const config = { ...DEFAULT_STREAMING_CONFIG, ...(partial ?? {}) };
  return {
    audioChunkMs: clamp(config.audioChunkMs, 100, 1000),
    discoveryTriggerSec: clamp(config.discoveryTriggerSec, 0.5, 6),
    trackingTriggerSec: clamp(config.trackingTriggerSec, 0.15, 3),
    discoveryMaxWindowSec: clamp(config.discoveryMaxWindowSec, 3, 45),
    trackingMaxWindowSec: clamp(config.trackingMaxWindowSec, 3, 45),
    tailAfterCommitSec: clamp(config.tailAfterCommitSec, 0, 6),
    tailAfterPendingAdvanceSec: clamp(config.tailAfterPendingAdvanceSec, 0, 3),
    finalSilenceSec: clamp(config.finalSilenceSec, 0.3, 5),
    silenceRmsThreshold: clamp(config.silenceRmsThreshold, 0.001, 0.05),
    firstMatchThreshold: clamp(config.firstMatchThreshold, 0.1, 0.99),
    verseMatchThreshold: clamp(config.verseMatchThreshold, 0.1, 0.99),
    discoveryRepeatCycles: Math.round(clamp(config.discoveryRepeatCycles, 1, 5)),
    acousticClearMargin: clamp(config.acousticClearMargin, 0, 1),
    acousticContinuationMargin: clamp(config.acousticContinuationMargin, 0, 1),
    decodeStabilityEnabled: Boolean(config.decodeStabilityEnabled),
    decodeStabilityRatio: clamp(config.decodeStabilityRatio, 0, 1),
    nonContinuationJumpThreshold: clamp(config.nonContinuationJumpThreshold, 0.1, 0.99),
    nextVerseEmitMode: isNextVerseEmitMode(config.nextVerseEmitMode)
      ? config.nextVerseEmitMode
      : DEFAULT_STREAMING_CONFIG.nextVerseEmitMode,
    trackingCompletionCoverage: clamp(config.trackingCompletionCoverage, 0.5, 1),
    trackingPrefixTolerance: clamp(config.trackingPrefixTolerance, 0, 1),
    lookaheadWords: Math.round(clamp(config.lookaheadWords, 1, 15)),
    staleCycleLimit: Math.round(clamp(config.staleCycleLimit, 1, 12)),
    trackingSilenceTimeoutSec: clamp(config.trackingSilenceTimeoutSec, 0.5, 10),
    advanceRelativeMargin: clamp(config.advanceRelativeMargin, -2, 8),
    advancePrefixTokens: Math.round(clamp(config.advancePrefixTokens, 3, 60)),
    advanceFlushStrictMargin: clamp(config.advanceFlushStrictMargin, -2, 8),
  };
}

function clamp(value: number, min: number, max: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return min;
  return Math.min(max, Math.max(min, parsed));
}

function isNextVerseEmitMode(value: unknown): value is NextVerseEmitMode {
  return (
    value === "deferred_confirm" ||
    value === "candidate_until_confirmed" ||
    value === "immediate_on_completion"
  );
}
