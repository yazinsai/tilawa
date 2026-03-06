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

export interface WordProgressMessage {
  type: "word_progress";
  surah: number;
  ayah: number;
  word_index: number;
  total_words: number;
  matched_indices: number[];
}

export interface WordCorrection {
  word_index: number;
  expected: string;
  got: string;
  error_type: "substitution" | "deletion" | "insertion";
}

export interface WordCorrectionMessage {
  type: "word_correction";
  surah: number;
  ayah: number;
  corrections: WordCorrection[];
}

export interface RawTranscriptMessage {
  type: "raw_transcript";
  text: string;
  confidence: number;
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
  | { type: "reset" };

// Worker -> Main
export type WorkerOutbound =
  | { type: "loading"; percent: number }
  | { type: "loading_status"; message: string }
  | { type: "ready" }
  | { type: "error"; message: string }
  | VerseMatchMessage
  | WordProgressMessage
  | WordCorrectionMessage
  | RawTranscriptMessage;

// ---------------------------------------------------------------------------
// Quran data (from quran.json)
// ---------------------------------------------------------------------------
export interface QuranVerse {
  surah: number;
  ayah: number;
  text_uthmani: string;
  surah_name: string;
  surah_name_en: string;
  phonemes: string;
  phonemes_joined: string;
  phonemes_joined_no_bsm?: string | null;
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
export const MAX_WINDOW_SECONDS = 10.0;
export const MAX_WINDOW_SAMPLES = SAMPLE_RATE * MAX_WINDOW_SECONDS;
export const SILENCE_RMS_THRESHOLD = 0.005;

export const VERSE_MATCH_THRESHOLD = 0.45;
export const FIRST_MATCH_THRESHOLD = 0.75;
export const RAW_TRANSCRIPT_THRESHOLD = 0.25;
export const SURROUNDING_CONTEXT = 2;

export const TRACKING_TRIGGER_SECONDS = 0.5;
export const TRACKING_TRIGGER_SAMPLES = SAMPLE_RATE * TRACKING_TRIGGER_SECONDS;
export const TRACKING_SILENCE_TIMEOUT = 4.0;
export const TRACKING_SILENCE_SAMPLES = SAMPLE_RATE * TRACKING_SILENCE_TIMEOUT;
export const TRACKING_MAX_WINDOW_SECONDS = 5.0;
export const TRACKING_MAX_WINDOW_SAMPLES =
  SAMPLE_RATE * TRACKING_MAX_WINDOW_SECONDS;
export const STALE_CYCLE_LIMIT = 4;
export const LOOKAHEAD = 5;
