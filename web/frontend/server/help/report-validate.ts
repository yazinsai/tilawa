// Server twin of `validateReport(json, wav)` (contract §6 "Validation").
// Self-contained on purpose: the SDK (`@tilawa/core/report`) and Python
// (`tools/report_validate.py`) carry rule-for-rule copies. Keep the rule list
// and the error codes in step with them; never import from packages/core here.
//
// Rules (error code: rule):
//   size.json        report.json ≤ 4 MiB (UTF-8 bytes)
//   json             valid UTF-8 JSON object
//   schema           schema === "tilawa.report/1" (unknown majors rejected; extra fields allowed)
//   uuid.*           reportId, sessionId, incidents[].id are lowercase 8-4-4-4-12 hex uuids
//   fields.*         required field types (see checkShape)
//   wav.header       canonical 44-byte header: "RIFF" size=len-8 "WAVE", "fmt " size 16,
//                    (format, channels, rate, byteRate, blockAlign, bits) = (1,1,16000,32000,2,16),
//                    "data" at offset 36 with size = len-44 (even)
//   audio.max        ≤ 300 s: durationMs ≤ 300000 and samples ≤ 4_800_000
//   audio.duration   |samples/16 − durationMs| ≤ 1
//   audio.sha256     sha256(wav bytes) lowercase hex === audio.sha256
//   incidents.max    ≤ 100 incidents
//   incident.span    startMs ≤ endMs; startMs ≥ audio.startMs unless truncatedStart;
//                    endMs ≤ audio.startMs + durationMs unless truncatedEnd
//   claim.note       claim.note and claim.text ≤ 2000 chars each
//   consent.send     consent.send === true
//   consent.version  (server policy option only) consent.version in the allowlist
import { createHash } from "node:crypto";

export const REPORT_SCHEMA = "tilawa.report/1";
export const MAX_REPORT_JSON_BYTES = 4 * 1024 * 1024;
export const MAX_AUDIO_MS = 300_000;
export const SAMPLE_RATE = 16_000;
export const WAV_HEADER_BYTES = 44;
export const MAX_SAMPLES = (MAX_AUDIO_MS / 1000) * SAMPLE_RATE;
export const MAX_WAV_BYTES = WAV_HEADER_BYTES + MAX_SAMPLES * 2;
export const MAX_INCIDENTS = 100;
export const MAX_NOTE_CHARS = 2000;
export const INCIDENT_KINDS = ["false_warning", "missed_mistake", "wrong_position", "stumble", "unflagged_sample", "other"] as const;
const ANSWERS = new Set(["yes", "no", "unsure", "skipped"]);
const MODES = new Set(["free", "memorization"]);
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const CONTRIBUTOR = /^[A-Za-z0-9_-]{1,64}$/;
const SHA256 = /^[0-9a-f]{64}$/;

export type ReportIncidentKind = (typeof INCIDENT_KINDS)[number];
export interface ValidatedReport {
  schema: typeof REPORT_SCHEMA;
  reportId: string;
  sessionId: string;
  createdAt: string;
  source: string;
  contributor?: string;
  audio: { startMs: number; durationMs: number; sha256: string; truncatedStart: boolean; truncatedEnd: boolean; [k: string]: unknown };
  incidents: { id: string; kind: ReportIncidentKind; startMs: number; endMs: number; [k: string]: unknown }[];
  consent: { policy: string; version: string; send: true; training: boolean; publicDataset: boolean };
  [k: string]: unknown;
}
export type ValidationResult = { ok: true; report: ValidatedReport } | { ok: false; errors: string[] };
export interface ValidateOptions {
  /** When given, consent.version must be one of these (server policy, not an artifact rule). */
  consentVersions?: readonly string[];
}

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isStr = (v: unknown, max = 256): v is string => typeof v === "string" && v.length > 0 && v.length <= max;
const isBool = (v: unknown): v is boolean => typeof v === "boolean";
const isIso = (v: unknown) => isStr(v, 64) && !Number.isNaN(Date.parse(v));
const isUuid = (v: unknown) => typeof v === "string" && UUID.test(v);

/** Sample count of a canonical 44-byte PCM16 LE mono 16 kHz WAV, or null when the header is not canonical. */
export function canonicalWavSamples(wav: Uint8Array): number | null {
  if (wav.length < WAV_HEADER_BYTES) return null;
  const v = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);
  const tag = (o: number) => String.fromCharCode(wav[o], wav[o + 1], wav[o + 2], wav[o + 3]);
  const dataBytes = wav.length - WAV_HEADER_BYTES;
  const ok =
    tag(0) === "RIFF" && v.getUint32(4, true) === wav.length - 8 && tag(8) === "WAVE" &&
    tag(12) === "fmt " && v.getUint32(16, true) === 16 &&
    v.getUint16(20, true) === 1 && v.getUint16(22, true) === 1 && v.getUint32(24, true) === 16000 &&
    v.getUint32(28, true) === 32000 && v.getUint16(32, true) === 2 && v.getUint16(34, true) === 16 &&
    tag(36) === "data" && v.getUint32(40, true) === dataBytes && dataBytes % 2 === 0;
  return ok ? dataBytes / 2 : null;
}

function checkIncident(inc: unknown, i: number, need: (ok: boolean, code: string) => void) {
  need(isObj(inc) && isUuid(inc.id), `uuid.incidents[${i}].id`);
  const m = isObj(inc) ? inc.markedAt : undefined;
  const c = isObj(inc) ? inc.claim : undefined;
  need(
    isObj(inc) && (INCIDENT_KINDS as readonly unknown[]).includes(inc.kind) && isNum(inc.startMs) && isNum(inc.endMs) &&
      Array.isArray(inc.eventIds) && inc.eventIds.every((x) => typeof x === "string") &&
      isObj(m) && isNum(m.sessionMs) && isNum(m.wallMs) &&
      isObj(c) && (c.answer === null || (typeof c.answer === "string" && ANSWERS.has(c.answer))) &&
      (c.question === undefined || typeof c.question === "string") &&
      (c.note === undefined || typeof c.note === "string") &&
      (c.text === undefined || typeof c.text === "string"),
    `fields.incidents[${i}]`,
  );
}

function checkShape(r: Obj, need: (ok: boolean, code: string) => void) {
  need(r.schema === REPORT_SCHEMA, "schema");
  need(isUuid(r.reportId), "uuid.reportId");
  need(isUuid(r.sessionId), "uuid.sessionId");
  need(isIso(r.createdAt), "fields.createdAt");
  need(isStr(r.source, 128), "fields.source");
  need(r.contributor === undefined || (typeof r.contributor === "string" && CONTRIBUTOR.test(r.contributor)), "fields.contributor");

  const e = r.engine;
  need(
    isObj(e) && isStr(e.sdkVersion) && isStr(e.modelId) && isStr(e.modelVersion) &&
      (e.modelSha256 === null || (typeof e.modelSha256 === "string" && SHA256.test(e.modelSha256))) &&
      isStr(e.referenceVersion) && e.eventSchema === 1 &&
      (e.tier === undefined || isStr(e.tier)) && (e.backend === undefined || isStr(e.backend)),
    "fields.engine",
  );

  const s = r.session;
  const p = isObj(s) ? s.passage : undefined;
  need(
    isObj(s) && typeof s.mode === "string" && MODES.has(s.mode) && isObj(s.policy) && isIso(s.startedAt) && isBool(s.vad) &&
      (p === null || (isObj(p) && Number.isInteger(p.surah) && Number.isInteger(p.fromAyah) && Number.isInteger(p.toAyah))),
    "fields.session",
  );

  const a = r.audio;
  const req = isObj(a) ? a.requested : undefined;
  need(
    isObj(a) && a.format === "pcm_s16le" && a.sampleRate === 16000 && a.channels === 1 &&
      isNum(a.startMs) && a.startMs >= 0 && isNum(a.durationMs) && a.durationMs >= 0 &&
      typeof a.sha256 === "string" && SHA256.test(a.sha256) &&
      isObj(req) && isNum(req.startMs) && isNum(req.endMs) && isBool(a.truncatedStart) && isBool(a.truncatedEnd),
    "fields.audio",
  );

  need(Array.isArray(r.incidents), "fields.incidents");
  if (Array.isArray(r.incidents)) r.incidents.forEach((inc, i) => checkIncident(inc, i, need));

  const o = r.observations;
  need(isObj(o) && Array.isArray(o.events) && (o.evidence === undefined || Array.isArray(o.evidence)), "fields.observations");

  const k = r.consent;
  need(isObj(k) && isStr(k.policy, 128) && isStr(k.version, 64) && isBool(k.training) && isBool(k.publicDataset), "fields.consent");
  need(isObj(k) && k.send === true, "consent.send");
}

export function validateReport(json: string | Uint8Array, wav: Uint8Array, options: ValidateOptions = {}): ValidationResult {
  const bytes = typeof json === "string" ? new TextEncoder().encode(json) : json;
  if (bytes.length > MAX_REPORT_JSON_BYTES) return { ok: false, errors: ["size.json"] };
  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    return { ok: false, errors: ["json"] };
  }
  if (!isObj(parsed)) return { ok: false, errors: ["json"] };

  const errors: string[] = [];
  const need = (ok: boolean, code: string) => { if (!ok) errors.push(code); };
  checkShape(parsed, need);

  const a: Obj = isObj(parsed.audio) ? parsed.audio : {};
  const samples = canonicalWavSamples(wav);
  need(samples !== null, "wav.header");
  if (samples !== null && isNum(a.durationMs)) {
    need(a.durationMs <= MAX_AUDIO_MS && samples <= MAX_SAMPLES, "audio.max");
    need(Math.abs(samples / 16 - a.durationMs) <= 1, "audio.duration");
  }
  if (typeof a.sha256 === "string") need(createHash("sha256").update(wav).digest("hex") === a.sha256, "audio.sha256");

  if (Array.isArray(parsed.incidents)) {
    need(parsed.incidents.length <= MAX_INCIDENTS, "incidents.max");
    parsed.incidents.forEach((inc, i) => {
      if (!isObj(inc)) return;
      if (isNum(inc.startMs) && isNum(inc.endMs) && isNum(a.startMs) && isNum(a.durationMs)) {
        const end = a.startMs + a.durationMs;
        need(
          inc.startMs <= inc.endMs &&
            (a.truncatedStart === true || inc.startMs >= a.startMs) &&
            (a.truncatedEnd === true || inc.endMs <= end),
          `incident.span[${i}]`,
        );
      }
      const c = inc.claim;
      if (isObj(c)) {
        need(!(typeof c.note === "string" && c.note.length > MAX_NOTE_CHARS) &&
          !(typeof c.text === "string" && c.text.length > MAX_NOTE_CHARS), `claim.note[${i}]`);
      }
    });
  }

  if (options.consentVersions) {
    const v = isObj(parsed.consent) ? parsed.consent.version : undefined;
    need(typeof v === "string" && options.consentVersions.includes(v), "consent.version");
  }
  return errors.length ? { ok: false, errors } : { ok: true, report: parsed as unknown as ValidatedReport };
}
