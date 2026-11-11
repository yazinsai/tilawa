import type { TilawaReport } from "./types.js";
import { validateReport } from "./validate.js";

/**
 * A report's audio travels beside it, not inside the JSON: the recorder and `decodeReport` attach the canonical WAV
 * to the report object they return. Copies of the object (spread, structuredClone, JSON round trip) lose it.
 */
const audioOf = new WeakMap<object, Uint8Array>();

/** @internal Pairs a report object with its WAV bytes. */
export function attachAudio<T extends TilawaReport>(report: T, wav: Uint8Array): T {
  audioOf.set(report, wav);
  return report;
}

/** The attached WAV bytes, or undefined for a report object the recorder or `decodeReport` did not return. */
export function reportAudio(report: TilawaReport): Uint8Array | undefined {
  return audioOf.get(report);
}

export interface EncodedReport {
  /** `report.json`, UTF-8 text. */
  json: string;
  /** `audio.wav`: canonical 44-byte header, PCM16 LE, mono, 16 kHz. */
  wav: Uint8Array;
}

/** The portable artifact. Throws when `report` has no attached audio (see `reportAudio`). */
export function encodeReport(report: TilawaReport): EncodedReport {
  const wav = audioOf.get(report);
  if (!wav) throw new TypeError("This report has no audio attached; use the object createRecorder or decodeReport returned");
  return { json: JSON.stringify(report), wav };
}

export class ReportValidationError extends Error {
  constructor(readonly errors: string[]) {
    super(`Invalid Tilawa report: ${errors.join("; ")}`);
    this.name = "ReportValidationError";
  }
}

/** Parses and validates an artifact; the returned report has the audio attached, so it can be re-encoded or sent. */
export async function decodeReport(json: string | Uint8Array, wav: Uint8Array): Promise<TilawaReport> {
  const result = await validateReport(json, wav);
  if (!result.ok || !result.report) throw new ReportValidationError(result.errors);
  return attachAudio(result.report, wav);
}
