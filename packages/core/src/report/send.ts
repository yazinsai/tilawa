import { encodeReport } from "./artifact.js";
import type { TilawaReport } from "./types.js";
import { validateReport } from "./validate.js";
import { ConsentError } from "./recorder.js";

export interface ReportReceipt {
  reportId: string;
  /** ISO time the server stored it; null when a duplicate reply carried no receipt. */
  receivedAt: string | null;
  /** Shown to the contributor once, for `DELETE …/reports/:id` with `X-Delete-Token`. Null when not returned. */
  deleteToken: string | null;
  /** The server already had this `reportId` (HTTP 409, or a 200 for a repeated upload). */
  duplicate: boolean;
  status: number;
}

export interface SendReportOptions {
  /** The app's own collection endpoint, e.g. `https://tilawa.dev/help/api/v2/reports`. Required: nothing is built in. */
  endpoint: string;
  fetch?: typeof globalThis.fetch;
  /** Extra headers, e.g. a scoped per-app ingestion key. Never baked into the SDK. */
  headers?: Record<string, string>;
  signal?: AbortSignal;
  /** Retries after the first attempt, on network errors, 429 and 5xx. Default 3. */
  retries?: number;
  /** First backoff in ms, doubled per retry. Default 500. */
  backoffMs?: number;
}

export class ReportSendError extends Error {
  constructor(message: string, readonly status: number | null, readonly body?: string, options?: { cause?: unknown }) {
    super(message);
    this.name = "ReportSendError";
    if (options?.cause !== undefined) (this as { cause?: unknown }).cause = options.cause;
  }
}

/**
 * POSTs multipart `report` (report.json) + `audio` (audio.wav) to the app's endpoint. Idempotent on `reportId`, so
 * retries are safe and a duplicate reply counts as success. This is the only network I/O in @tilawa/core/report.
 */
export async function sendReport(report: TilawaReport, options: SendReportOptions): Promise<ReportReceipt> {
  if (!options || typeof options.endpoint !== "string" || !options.endpoint) throw new TypeError("sendReport needs options.endpoint");
  if (report?.consent?.send !== true) throw new ConsentError();
  const { json, wav } = encodeReport(report);
  const valid = await validateReport(json, wav);
  if (!valid.ok) throw new ReportSendError(`Refusing to send an invalid report: ${valid.errors.join("; ")}`, null);
  const doFetch = options.fetch ?? globalThis.fetch;
  if (typeof doFetch !== "function") throw new TypeError("sendReport needs fetch (pass options.fetch)");
  const retries = Math.max(0, Math.floor(options.retries ?? 3));
  const backoff = Math.max(0, options.backoffMs ?? 500);

  let last: ReportSendError | null = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    if (attempt > 0) await sleep(backoff * 2 ** (attempt - 1), options.signal);
    options.signal?.throwIfAborted?.();
    const body = new FormData();
    body.append("report", new Blob([json], { type: "application/json" }), "report.json");
    body.append("audio", new Blob([wav as unknown as ArrayBuffer], { type: "audio/wav" }), "audio.wav");
    let res: Response;
    try {
      res = await doFetch(options.endpoint, { method: "POST", body, headers: { ...(options.headers ?? {}) }, signal: options.signal });
    } catch (e) {
      if (isAbort(e, options.signal)) throw e;
      last = new ReportSendError(`Network error sending report: ${(e as Error)?.message ?? e}`, null, undefined, { cause: e });
      continue;
    }
    const text = await res.text().catch(() => "");
    if (res.status === 409 || res.ok) return receiptOf(report.reportId, res.status, text);
    last = new ReportSendError(`Report upload failed with HTTP ${res.status}`, res.status, text);
    if (res.status !== 429 && res.status < 500) throw last;
  }
  throw last ?? new ReportSendError("Report upload failed", null);
}

function receiptOf(reportId: string, status: number, text: string): ReportReceipt {
  let body: Record<string, unknown> = {};
  try {
    const parsed = JSON.parse(text);
    if (parsed && typeof parsed === "object") body = parsed;
  } catch {
    /* duplicates may carry no body */
  }
  if (typeof body.reportId === "string" && body.reportId !== reportId)
    throw new ReportSendError(`The server acknowledged report ${body.reportId}, not ${reportId}`, status, text);
  return {
    reportId,
    receivedAt: typeof body.receivedAt === "string" ? body.receivedAt : null,
    deleteToken: typeof body.deleteToken === "string" ? body.deleteToken : null,
    duplicate: status === 409 || body.duplicate === true,
    status,
  };
}

function isAbort(e: unknown, signal?: AbortSignal) {
  return signal?.aborted === true || (e as { name?: string })?.name === "AbortError";
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason ?? new DOMException("Aborted", "AbortError"));
    const t = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(t);
      reject(signal?.reason ?? new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}
