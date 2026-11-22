export { ModelBudgetError, ModelIntegrityError } from "./runtime/models.js";

/** `pushAudio`, `pushEvidence` or `end` on a session that has ended or been disposed. */
export class SessionEndedError extends Error {
  override name = "SessionEndedError";
  constructor(message = "The session has ended") {
    super(message);
  }
}

export type LoadStage = "model" | "reference" | "warmup";

/** `createTilawa` could not load its model or reference. `cause` is the underlying error. */
export class TilawaLoadError extends Error {
  override name = "TilawaLoadError";
  constructor(readonly stage: LoadStage, override readonly cause: unknown) {
    super(`Could not load the ${stage}: ${cause instanceof Error ? cause.message : String(cause)}`, { cause });
  }
}

export type MicrophoneErrorReason = "denied" | "unavailable" | "insecure-context";

/** `listen()` could not open the microphone. */
export class MicrophoneError extends Error {
  override name = "MicrophoneError";
  constructor(readonly reason: MicrophoneErrorReason, message: string, options?: { cause?: unknown }) {
    super(message, options);
  }
}

export function abortError(signal?: AbortSignal): Error {
  if (signal?.reason instanceof Error) return signal.reason;
  if (typeof DOMException === "function") return new DOMException("The operation was aborted", "AbortError");
  return Object.assign(new Error("The operation was aborted"), { name: "AbortError" });
}

export const isAbortError = (e: unknown) => e instanceof Error && e.name === "AbortError";
