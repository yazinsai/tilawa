// The one seam between /help and the engine (contract §2, §3, §6). Everything engine-shaped goes through here:
// the UI never imports @tilawa/core/web itself. `?mock=1` swaps in a scripted engine with synthetic audio
// (engine-mock.ts) that drives the real report recorder, so the whole flow runs without a model.
import {
  createRecorder,
  encodeReport,
  type Incident,
  type RecordableSession,
  type Recorder,
  type ReportConsent,
  type TilawaReport,
} from "@tilawa/core/report";
import type { EngineEvent, TrackerState, WordRef } from "./types";

export const REPORT_SOURCE = "tilawa.dev/help";
/** decisions.md item 5: /help sends the whole take, capped at 5 minutes. */
export const TAKE_MAX_MS = 300_000;
/** Post-stop review can be slow for some readers; keep the take that long before the recorder drops it. */
const RETAIN_AFTER_END_MS = 6 * 3_600_000;

export interface LoadProgress {
  /** 0..1, or null while the size is unknown. */
  fraction: number | null;
  fromCache: boolean;
}

export interface LivePosition {
  timeMs: number;
  /** The most recently recited word, or null while we are still finding the reader's place. */
  word: WordRef | null;
  state: TrackerState;
}

export interface LiveWarning {
  id: string;
  timeMs: number;
  state: "suspected" | "confirmed" | "resolved" | "uncertain";
  errorClass: string;
  location: WordRef;
}

export interface TakeOptions {
  consent: ReportConsent;
  contributor?: string;
  maxMs?: number;
  onPosition?(p: LivePosition): void;
  onChecker?(c: LiveWarning): void;
  /** The take reached `maxMs`; the UI should call `stop()`. */
  onLimit?(): void;
}

export interface Take {
  readonly session: RecordableSession;
  readonly recorder: Recorder;
  /** Session-clock ms heard so far. */
  now(): number;
  /** "I stumbled here": a stumble incident at session-clock now. Never blocks the audio path. */
  bookmark(): Incident;
  /** Stops the microphone, drains the engine and returns the recorded take. */
  stop(): Promise<TakeRecording>;
  /** Throws the take away (no report possible afterwards). */
  cancel(): void;
}

export interface TakeRecording {
  /** End of the take on the session clock; the audio covers [0, durationMs]. */
  durationMs: number;
  /** Position and checker events, in arrival order. */
  events: EngineEvent[];
  /** The reader's bookmarks (stumble incidents). */
  bookmarks: Incident[];
  /** The whole take as canonical PCM16 16 kHz WAV, for review clips. */
  wav: Uint8Array;
  /** The take hit the 5-minute cap. */
  truncated: boolean;
  /** The take's report: one contribution with the given incidents (whole take, ≤5 min). */
  report(incidents: Incident[], consent: ReportConsent): Promise<TilawaReport>;
  dispose(): void;
}

/** The part of `Tilawa` (@tilawa/core/web) the adapter uses. The real one and the `?mock=1` one both fit. */
export interface HostSession extends RecordableSession {
  dispose(): void;
}
export interface EngineHost {
  startFreeRecitation(): HostSession;
  listen(session: HostSession, options?: { vad?: false }): Promise<{ stop(): Promise<unknown> }>;
  dispose(): void | Promise<void>;
}

export interface HelpEngine {
  readonly mock: boolean;
  startTake(options: TakeOptions): Promise<Take>;
  dispose(): void;
}

export type HelpEngineErrorKind = "load" | "mic-denied" | "mic-unavailable" | "insecure" | "empty" | "failed";

export class HelpEngineError extends Error {
  readonly kind: HelpEngineErrorKind;
  constructor(kind: HelpEngineErrorKind, message: string, options?: { cause?: unknown }) {
    super(message);
    this.kind = kind;
    this.name = "HelpEngineError";
    if (options?.cause !== undefined) (this as { cause?: unknown }).cause = options.cause;
  }
}

export function isMockMode(search = globalThis.location?.search ?? ""): boolean {
  return new URLSearchParams(search).get("mock") === "1";
}

// Stage weights for one overall progress number: the model dominates the download.
const STAGE: Record<string, [number, number]> = { model: [0, 0.85], reference: [0.85, 0.97], warmup: [0.97, 1] };

/** Prepares the engine (download once, then cached). Rejects with HelpEngineError("load"). */
export async function loadEngine(onProgress: (p: LoadProgress) => void, options: { mock?: boolean; signal?: AbortSignal } = {}): Promise<HelpEngine> {
  const mock = options.mock ?? isMockMode();
  const progress = (p: { stage: string; fraction: number; fromCache: boolean }) => {
    const [a, b] = STAGE[p.stage] ?? [0, 1];
    const f = Number.isFinite(p.fraction) ? Math.min(1, Math.max(0, p.fraction)) : null;
    onProgress({ fraction: f === null ? null : a + (b - a) * f, fromCache: p.fromCache });
  };
  let tilawa: EngineHost;
  try {
    if (mock) tilawa = await (await import("./engine-mock")).createMockTilawa({ onProgress: progress, signal: options.signal });
    else {
      const web = await import("@tilawa/core/web");
      const real = await web.createTilawa({ onProgress: progress, signal: options.signal });
      tilawa = {
        startFreeRecitation: () => real.startFreeRecitation(),
        // VAD off: the session clock is then the whole take, so clips line up with what was said.
        listen: (session, o) => real.listen(session as Parameters<typeof real.listen>[0], { vad: false, ...o }),
        dispose: () => real.dispose(),
      };
    }
  } catch (e) {
    if ((e as { name?: string })?.name === "AbortError") throw e;
    throw new HelpEngineError("load", `Could not prepare the engine: ${(e as Error)?.message ?? e}`, { cause: e });
  }
  return {
    mock,
    startTake: (o) => startTake(tilawa, o),
    dispose: () => void tilawa.dispose(),
  };
}

function micError(e: unknown): HelpEngineError {
  const err = e as { name?: string; reason?: string; message?: string };
  const reason =
    err?.name === "MicrophoneError"
      ? err.reason
      : err?.name === "NotAllowedError" || err?.name === "SecurityError"
        ? "denied"
        : err?.name === "NotFoundError" || err?.name === "NotReadableError" || err?.name === "OverconstrainedError"
          ? "unavailable"
          : null;
  const kind: HelpEngineErrorKind = reason === "denied" ? "mic-denied" : reason === "unavailable" ? "mic-unavailable" : reason === "insecure-context" ? "insecure" : "failed";
  return new HelpEngineError(kind, err?.message ?? String(e), { cause: e });
}

async function startTake(tilawa: EngineHost, o: TakeOptions): Promise<Take> {
  const maxMs = Math.min(o.maxMs ?? TAKE_MAX_MS, TAKE_MAX_MS);
  const session = tilawa.startFreeRecitation();
  const recorder = createRecorder(session, {
    retain: { take: true, maxMs },
    source: REPORT_SOURCE,
    contributor: o.contributor,
    consent: o.consent,
    retainAfterEndMs: RETAIN_AFTER_END_MS,
  });
  const events: EngineEvent[] = [];
  const off = session.subscribe((e) => {
    if (e.type === "capabilities") return;
    events.push(e);
    try {
      if (e.type === "position") o.onPosition?.({ timeMs: e.timeMs, word: e.observed[0]?.word ?? null, state: e.state });
      else o.onChecker?.({ id: e.id, timeMs: e.timeMs, state: e.state, errorClass: e.errorClass, location: e.location });
    } catch (err) {
      console.error(err); // UI trouble must never reach the engine
    }
  });

  let mic: { stop(): Promise<unknown> };
  try {
    mic = await tilawa.listen(session);
  } catch (e) {
    off();
    recorder.dispose();
    session.dispose();
    throw micError(e);
  }

  let limitHit = false;
  const watch = setInterval(() => {
    if (!limitHit && session.now() >= maxMs - 300) {
      limitHit = true;
      o.onLimit?.();
    }
  }, 250);

  let stopping: Promise<TakeRecording> | null = null;
  const stop = () =>
    (stopping ??= (async () => {
      clearInterval(watch);
      await mic.stop();
      await session.idle?.();
      off();
      const bookmarks = recorder.incidents();
      let preview: TilawaReport;
      try {
        preview = await recorder.contribution({ consent: o.consent });
      } catch (e) {
        recorder.dispose();
        session.dispose();
        if ((e as { reason?: string })?.reason === "empty") throw new HelpEngineError("empty", "Nothing was recorded", { cause: e });
        throw new HelpEngineError("failed", (e as Error)?.message ?? String(e), { cause: e });
      }
      const wav = encodeReport(preview).wav;
      const durationMs = preview.audio.startMs + preview.audio.durationMs;
      return {
        durationMs,
        events,
        bookmarks,
        wav,
        truncated: preview.audio.truncatedEnd || limitHit,
        report: (incidents, consent) => recorder.contribution({ incidents, consent }),
        dispose: () => {
          recorder.dispose();
          session.dispose();
        },
      };
    })());

  return {
    session,
    recorder,
    now: () => session.now(),
    bookmark: () => recorder.mark({ kind: "stumble" }),
    stop,
    cancel: () => {
      clearInterval(watch);
      off();
      void mic.stop().catch(() => undefined);
      recorder.dispose();
      session.dispose();
    },
  };
}
