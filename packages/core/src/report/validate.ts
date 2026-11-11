import { sha256Hex } from "./crypto.js";
import { REPORT_LIMITS } from "./types.js";
import type { TilawaReport } from "./types.js";
import { WAV_HEADER_BYTES, wavProblems } from "./wav.js";

/*
 * The report validator (contract section 6, "Validation"). tools/report_validate.py mirrors every rule here; change
 * both together.
 */

export const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const ISO_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$/;
const SCHEMA_RE = /^tilawa\.report\/(\d+)(\.\d+)?$/;
const SHA_RE = /^[0-9a-f]{64}$/;
const CONTRIBUTOR_RE = /^[A-Za-z0-9_-]{1,64}$/;
const MAX_SOURCE_CHARS = 128;
const INCIDENT_KINDS = new Set(["false_warning", "missed_mistake", "wrong_position", "stumble", "unflagged_sample", "other"]);
const ANSWERS = new Set(["yes", "no", "unsure", "skipped"]);

export interface ReportValidation {
  ok: boolean;
  errors: string[];
  /** The parsed report when `ok`. */
  report?: TilawaReport;
}

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isStr = (v: unknown, max = Infinity): v is string => typeof v === "string" && v.length > 0 && v.length <= max;
const isInt = (v: unknown, min: number): v is number => isNum(v) && Number.isInteger(v) && v >= min;

/** Structural checks on a parsed report, without the audio. Returns every problem found. */
export function reportProblems(r: unknown): string[] {
  const e: string[] = [];
  if (!isObj(r)) return ["report.json is not an object"];
  const schema = typeof r.schema === "string" ? SCHEMA_RE.exec(r.schema) : null;
  if (!schema) e.push("schema is not tilawa.report/<major>");
  else if (schema[1] !== "1") e.push(`unsupported report schema major ${schema[1]}`);
  if (typeof r.reportId !== "string" || !UUID_RE.test(r.reportId)) e.push("reportId is not a uuid");
  if (typeof r.sessionId !== "string" || !UUID_RE.test(r.sessionId)) e.push("sessionId is not a uuid");
  if (typeof r.createdAt !== "string" || !ISO_RE.test(r.createdAt)) e.push("createdAt is not an ISO timestamp");
  if (!isStr(r.source, MAX_SOURCE_CHARS)) e.push(`source must be 1-${MAX_SOURCE_CHARS} chars`);
  if (r.contributor !== undefined && (typeof r.contributor !== "string" || !CONTRIBUTOR_RE.test(r.contributor)))
    e.push("contributor must be 1-64 chars of [A-Za-z0-9_-]");

  const engine = r.engine;
  if (!isObj(engine)) e.push("engine is missing");
  else {
    for (const k of ["sdkVersion", "modelId", "modelVersion", "referenceVersion"])
      if (!isStr(engine[k])) e.push(`engine.${k} is missing`);
    if (engine.modelSha256 !== null && (typeof engine.modelSha256 !== "string" || !SHA_RE.test(engine.modelSha256)))
      e.push("engine.modelSha256 must be null or a sha256 hex digest");
    if (engine.eventSchema !== 1) e.push("engine.eventSchema must be 1");
    for (const k of ["tier", "backend"])
      if (engine[k] !== undefined && typeof engine[k] !== "string") e.push(`engine.${k} must be a string`);
  }

  const session = r.session;
  if (!isObj(session)) e.push("session is missing");
  else {
    if (session.mode !== "free" && session.mode !== "memorization") e.push("session.mode must be free or memorization");
    const p = session.passage;
    if (p !== null) {
      if (!isObj(p) || !isInt(p.surah, 1) || p.surah > 114 || !isInt(p.fromAyah, 1) || !isInt(p.toAyah, 1) || p.toAyah < p.fromAyah)
        e.push("session.passage must be null or {surah, fromAyah, toAyah}");
    }
    if (!isObj(session.policy)) e.push("session.policy must be an object");
    if (typeof session.startedAt !== "string" || !ISO_RE.test(session.startedAt)) e.push("session.startedAt is not an ISO timestamp");
    if (typeof session.vad !== "boolean") e.push("session.vad must be a boolean");
  }

  const audio = r.audio;
  let span: { start: number; end: number; truncatedStart: boolean; truncatedEnd: boolean } | null = null;
  if (!isObj(audio)) e.push("audio is missing");
  else {
    if (audio.format !== "pcm_s16le" || audio.sampleRate !== 16000 || audio.channels !== 1)
      e.push("audio must be pcm_s16le, 16000 Hz, mono");
    const okStart = isNum(audio.startMs) && audio.startMs >= 0;
    const okDur = isNum(audio.durationMs) && audio.durationMs > 0 && audio.durationMs <= REPORT_LIMITS.maxAudioMs;
    if (!okStart) e.push("audio.startMs must be a non-negative number");
    if (!okDur) e.push(`audio.durationMs must be in (0, ${REPORT_LIMITS.maxAudioMs}]`);
    if (typeof audio.sha256 !== "string" || !SHA_RE.test(audio.sha256)) e.push("audio.sha256 is not a sha256 hex digest");
    const q = audio.requested;
    if (!isObj(q) || !isNum(q.startMs) || !isNum(q.endMs) || q.endMs < q.startMs) e.push("audio.requested must be {startMs <= endMs}");
    const okFlags = typeof audio.truncatedStart === "boolean" && typeof audio.truncatedEnd === "boolean";
    if (!okFlags) e.push("audio.truncatedStart and audio.truncatedEnd must be booleans");
    if (okStart && okDur && okFlags)
      span = {
        start: audio.startMs as number,
        end: (audio.startMs as number) + (audio.durationMs as number),
        truncatedStart: audio.truncatedStart as boolean,
        truncatedEnd: audio.truncatedEnd as boolean,
      };
  }

  const incidents = r.incidents;
  if (!Array.isArray(incidents)) e.push("incidents must be an array");
  else {
    if (incidents.length > REPORT_LIMITS.maxIncidents) e.push(`at most ${REPORT_LIMITS.maxIncidents} incidents`);
    incidents.forEach((inc, i) => {
      const at = `incidents[${i}]`;
      if (!isObj(inc)) return void e.push(`${at} is not an object`);
      if (typeof inc.id !== "string" || !UUID_RE.test(inc.id)) e.push(`${at}.id is not a uuid`);
      if (typeof inc.kind !== "string" || !INCIDENT_KINDS.has(inc.kind)) e.push(`${at}.kind is unknown`);
      if (!isNum(inc.startMs) || !isNum(inc.endMs) || inc.endMs < inc.startMs) e.push(`${at} needs startMs <= endMs`);
      else if (span) {
        if (!span.truncatedStart && inc.startMs < span.start) e.push(`${at} starts before the audio`);
        if (!span.truncatedEnd && inc.endMs > span.end) e.push(`${at} ends after the audio`);
      }
      if (!Array.isArray(inc.eventIds) || inc.eventIds.some((x) => typeof x !== "string")) e.push(`${at}.eventIds must be strings`);
      const m = inc.markedAt;
      if (!isObj(m) || !isNum(m.sessionMs) || !isNum(m.wallMs)) e.push(`${at}.markedAt must be {sessionMs, wallMs}`);
      const c = inc.claim;
      if (!isObj(c)) return void e.push(`${at}.claim is missing`);
      if (!(c.answer === null || (typeof c.answer === "string" && ANSWERS.has(c.answer)))) e.push(`${at}.claim.answer is unknown`);
      for (const k of ["question", "text"])
        if (c[k] !== undefined && typeof c[k] !== "string") e.push(`${at}.claim.${k} must be a string`);
      if (c.note !== undefined && (typeof c.note !== "string" || c.note.length > REPORT_LIMITS.maxNoteChars))
        e.push(`${at}.claim.note must be at most ${REPORT_LIMITS.maxNoteChars} chars`);
    });
  }

  const obs = r.observations;
  if (!isObj(obs) || !Array.isArray(obs.events)) e.push("observations.events must be an array");
  else {
    if (obs.events.some((ev) => !isObj(ev) || typeof ev.type !== "string")) e.push("observations.events entries need a type");
    if (obs.evidence !== undefined && !Array.isArray(obs.evidence)) e.push("observations.evidence must be an array");
  }

  const consent = r.consent;
  if (!isObj(consent)) e.push("consent is missing");
  else {
    if (!isStr(consent.policy) || !isStr(consent.version)) e.push("consent.policy and consent.version are required");
    if (consent.send !== true) e.push("consent.send must be true");
    if (typeof consent.training !== "boolean" || typeof consent.publicDataset !== "boolean")
      e.push("consent.training and consent.publicDataset must be booleans");
  }
  return e;
}

/** Audio checks that need no hashing: canonical header, length against `durationMs`, the 300 s cap. */
export function audioProblems(r: unknown, wav: Uint8Array): string[] {
  const e = wavProblems(wav);
  const samples = Math.max(0, Math.floor((wav.length - WAV_HEADER_BYTES) / 2));
  if (samples > (REPORT_LIMITS.maxAudioMs / 1000) * REPORT_LIMITS.sampleRate) e.push("audio.wav is longer than 300 s");
  const audio = isObj(r) ? r.audio : undefined;
  if (isObj(audio) && isNum(audio.durationMs) && Math.abs(samples / 16 - audio.durationMs) > 1)
    e.push(`audio.wav holds ${samples / 16} ms but audio.durationMs is ${audio.durationMs}`);
  return e;
}

const utf8 = new TextEncoder();

/**
 * Validates a `report.json` + `audio.wav` pair. Shared by the SDK, the server and tools/report_validate.py.
 * Never throws for bad input; `errors` lists every problem.
 */
export async function validateReport(json: string | Uint8Array, wav: Uint8Array): Promise<ReportValidation> {
  const bytes = typeof json === "string" ? utf8.encode(json) : json;
  if (bytes.length > REPORT_LIMITS.maxJsonBytes) return { ok: false, errors: ["report.json is larger than 4 MiB"] };
  let parsed: unknown;
  try {
    parsed = JSON.parse(typeof json === "string" ? json : new TextDecoder().decode(json));
  } catch {
    return { ok: false, errors: ["report.json is not valid JSON"] };
  }
  const errors = [...reportProblems(parsed), ...audioProblems(parsed, wav)];
  const sha = isObj(parsed) && isObj(parsed.audio) ? parsed.audio.sha256 : undefined;
  if (typeof sha === "string" && SHA_RE.test(sha) && (await sha256Hex(wav)) !== sha)
    errors.push("audio.sha256 does not match audio.wav");
  return errors.length ? { ok: false, errors } : { ok: true, errors, report: parsed as TilawaReport };
}
