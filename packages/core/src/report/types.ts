/*
 * Engine event and evidence types come from the vendored engine (schema v1) and the session wrapper; this module
 * adds the report artifact types (contract section 6).
 */
export { EVENT_SCHEMA_VERSION } from "../engine/index.js";
export type {
  CheckerEvent,
  CheckerState,
  DecodedPhoneme,
  ErrorClass,
  PhonemeEvidence,
  PositionCandidate,
  PositionEvent,
  TopKPosteriors,
  TrackerState,
  WordRef,
  WordSpan,
} from "../engine/index.js";
export type { CapabilityEvent, EngineEvent, SessionPolicy } from "../session.js";
import type { EngineEvent, SessionPolicy } from "../session.js";
import type { PhonemeEvidence } from "../engine/index.js";

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
  /** A warning the app showed and asked the user about; the claim holds the answer. Neutral until reviewed. */
  | "warning"
  /** The app's UI asserts the warning is wrong (e.g. a "this is wrong" button). */
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
