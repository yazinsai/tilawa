/*
 * Engine event and evidence types, mirrored from the research engine's packages/core/src/events.ts and
 * evidence.ts (schema v1, commit 441dddc3) plus the V1 optional capability fields (contract section 4).
 * They mirror engine events.ts v1: when the vendored engine lands in src/engine/, replace this block with
 * `export type { ... } from "../engine/events.js"` and keep the report types below.
 */

export const EVENT_SCHEMA_VERSION = 1;

export interface WordRef {
  surah: number;
  ayah: number;
  word: number;
}

export interface WordSpan {
  start: WordRef;
  end: WordRef;
}

export type AlignmentOp =
  | { op: "match"; ref: string; refAt: number; obsAt: number }
  | { op: "sub"; ref: string; obs: string; refAt: number; obsAt: number }
  | { op: "del"; ref: string; refAt: number }
  | { op: "ins"; obs: string; obsAt: number };

export interface PositionCandidate {
  word: WordRef;
  span: WordSpan;
  confidence: number;
  cost: number;
  per: number;
  prefix?: "basmala";
}

export type TrackerState = "acquiring" | "locked" | "ambiguous" | "lost";

export interface TraceCandidate {
  span: WordSpan;
  offset: number;
  length: number;
  votes: number;
  cost: number;
  per: number;
}

export interface TrackerTrace {
  shortlist: TraceCandidate[];
  transition: { from: TrackerState | null; to: TrackerState };
  path: AlignmentOp[] | null;
  decision: { fit: number; separation: number; leadEdits: number; requiredEdits: number } | null;
}

export interface PositionEvent {
  type: "position";
  v: typeof EVENT_SCHEMA_VERSION;
  timeMs: number;
  referenceVersion: string;
  expected: WordRef | null;
  observed: PositionCandidate[];
  state: TrackerState;
  hold?: "start";
  relocated?: { from: WordRef; to: WordRef };
  spanHold?: true;
  trace?: TrackerTrace;
}

export type MemorizationErrorClass = "wrong_word" | "omitted_word" | "added_word" | "skipped_passage";

export type PronunciationErrorClass =
  | "short_vowel"
  | "consonant_h_ha"
  | "consonant_ain_hamza"
  | "consonant_sad_sin"
  | "consonant_dad_dal"
  | "consonant_ta_ta"
  | "consonant_qaf_kaf"
  | "consonant_dhal_zay"
  | "consonant_tha_sin"
  | "missing_shadda"
  | "madd_natural";

export type ErrorClass = MemorizationErrorClass | PronunciationErrorClass;

export type CheckerState = "suspected" | "confirmed" | "resolved" | "uncertain";

export interface CheckerEvent {
  type: "checker";
  v: typeof EVENT_SCHEMA_VERSION;
  id: string;
  timeMs: number;
  state: CheckerState;
  errorClass: ErrorClass;
  location: WordRef;
  expected?: string;
  observed?: string;
  confidence: number;
}

export interface CapabilityEvent {
  type: "capabilities";
  v: typeof EVENT_SCHEMA_VERSION;
  modelVersion: string;
  referenceVersion: string;
  supported: ErrorClass[];
  unsupported: ErrorClass[];
  /** V1 optional additions (contract section 4). */
  sdkVersion?: string;
  modelSha256?: string;
  tier?: string;
  backend?: string;
}

export type EngineEvent = PositionEvent | CheckerEvent | CapabilityEvent;

export interface SeamInsertionEvidence {
  word: WordRef;
  observed: string;
  fromFrame: number;
  toFrame: number;
  marginal: number;
}

export interface DecodedPhoneme {
  p: string;
  start: number;
  end: number;
  conf: number;
  seamInsertions?: SeamInsertionEvidence[];
}

export interface TopKPosteriors {
  k: number;
  startFrame: number;
  nFrames: number;
  ids: ArrayLike<number>;
  probs: ArrayLike<number>;
}

export interface PhonemeEvidence {
  inventory: string;
  frameMs: number;
  phonemes: DecodedPhoneme[];
  posteriors?: TopKPosteriors;
  decodedTo?: number;
}

/** Mirrors research packages/sdk/src/session.ts `SessionPolicy`. `confirm` is the checker's policy, kept opaque here. */
export interface SessionPolicy {
  countSelfCorrections?: boolean;
  confirm?: unknown;
  minConfidence?: number;
  evidenceMs?: number;
}

/* ------------------------------------------------------------------------------------------------------------ */
/* Report artifact (contract section 6, schema "tilawa.report/1").                                               */
/* ------------------------------------------------------------------------------------------------------------ */

export const REPORT_SCHEMA = "tilawa.report/1";

export interface ReportPassage {
  surah: number;
  fromAyah: number;
  toAyah: number;
}

export type IncidentKind =
  | "false_warning"
  | "missed_mistake"
  | "wrong_position"
  | "stumble"
  | "unflagged_sample"
  | "other";

export type ClaimAnswer = "yes" | "no" | "unsure" | "skipped" | null;

export interface IncidentClaim {
  question?: string;
  answer: ClaimAnswer;
  note?: string;
  text?: string;
}

export interface Incident {
  id: string;
  kind: IncidentKind;
  startMs: number;
  endMs: number;
  eventIds: string[];
  markedAt: { sessionMs: number; wallMs: number };
  claim: IncidentClaim;
}

export interface ReportConsent {
  policy: string;
  version: string;
  send: true;
  training: boolean;
  publicDataset: boolean;
}

export interface ReportEngine {
  sdkVersion: string;
  modelId: string;
  modelVersion: string;
  modelSha256: string | null;
  referenceVersion: string;
  eventSchema: 1;
  tier?: string;
  backend?: string;
}

export interface ReportAudio {
  format: "pcm_s16le";
  sampleRate: 16000;
  channels: 1;
  startMs: number;
  durationMs: number;
  sha256: string;
  requested: { startMs: number; endMs: number };
  truncatedStart: boolean;
  truncatedEnd: boolean;
}

export interface TilawaReport {
  schema: typeof REPORT_SCHEMA;
  reportId: string;
  sessionId: string;
  createdAt: string;
  source: string;
  contributor?: string;
  engine: ReportEngine;
  session: {
    mode: "free" | "memorization";
    passage: ReportPassage | null;
    policy: SessionPolicy;
    startedAt: string;
    vad: boolean;
  };
  audio: ReportAudio;
  incidents: Incident[];
  observations: { events: EngineEvent[]; evidence?: PhonemeEvidence[] };
  consent: ReportConsent;
}

/** Limits from contract section 6 "Validation". */
export const REPORT_LIMITS = {
  maxIncidents: 100,
  maxNoteChars: 2000,
  maxJsonBytes: 4 * 1024 * 1024,
  maxAudioMs: 300_000,
  maxContributorChars: 64,
  sampleRate: 16_000,
} as const;
