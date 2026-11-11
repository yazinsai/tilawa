/**
 * `@tilawa/core/report`: opt-in report capture, the portable `report.json` + `audio.wav` artifact, its validator,
 */
export { createRecorder, ConsentError, RecorderError } from "./recorder.js";
export type {
  ContributionInput,
  MarkInput,
  RecordableSession,
  Recorder,
  RecorderOptions,
  RecorderState,
  RecorderStats,
  ReportInput,
  RetainOptions,
  SessionState,
} from "./recorder.js";
export { decodeReport, encodeReport, reportAudio, ReportValidationError } from "./artifact.js";
export type { EncodedReport } from "./artifact.js";
export { validateReport } from "./validate.js";
export type { ReportValidation } from "./validate.js";
export { encodeWav } from "./wav.js";
export { REPORT_LIMITS, REPORT_SCHEMA } from "./types.js";
export type {
  ClaimAnswer,
  Incident,
  IncidentClaim,
  IncidentKind,
  ReportAudio,
  ReportConsent,
  ReportEngine,
  ReportPassage,
  TilawaReport,
} from "./types.js";
