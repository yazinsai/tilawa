/**
 * `@tilawa/core/report`: opt-in report capture, the portable `report.json` + `audio.wav` artifact, its validator,
 * and an optional upload helper. Nothing is retained without a recorder and nothing is sent without `sendReport`.
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
export { sendReport, ReportSendError } from "./send.js";
export type { ReportReceipt, SendReportOptions } from "./send.js";
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
