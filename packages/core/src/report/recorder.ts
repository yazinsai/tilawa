import { attachAudio, ReportValidationError } from "./artifact.js";
import { sha256Hex, uuidV4 } from "./crypto.js";
import { PcmStore } from "./pcm-store.js";
import { REPORT_LIMITS, REPORT_SCHEMA } from "./types.js";
import type {
  CapabilityEvent,
  CheckerEvent,
  EngineEvent,
  Incident,
  IncidentClaim,
  IncidentKind,
  PhonemeEvidence,
  ReportConsent,
  ReportEngine,
  ReportPassage,
  SessionPolicy,
  TilawaReport,
  WordRef,
} from "./types.js";
import { reportProblems } from "./validate.js";
import { encodeWav } from "./wav.js";

const RATE = REPORT_LIMITS.sampleRate;
const PER_MS = RATE / 1000;
const SLACK_MS = 30_000;
const DEFAULT_BEFORE_MS = 10_000;
const DEFAULT_AFTER_MS = 5_000;
const DEFAULT_TAKE_MS = 300_000;
const DEFAULT_RETAIN_AFTER_END_MS = 10 * 60_000;
const END_POLL_MS = 250;
/** Clock tolerance for consecutive tap chunks, in samples (1 ms). */
const CONTIGUITY_SAMPLES = PER_MS;
const MAX_CHECKER_EVENTS = 4_000;
const MAX_TIMED_EVENTS = 50_000;

export type SessionState = "active" | "ending" | "ended" | "disposed";

/**
 * The part of a V1 `Session` the recorder needs (contract sections 3-4). Structural, so any session that offers it
 * can be recorded. Optional members are used when present.
 */
export interface RecordableSession {
  readonly id: string;
  /** ISO wall clock at the first sample; null/undefined before it. */
  readonly startedAt: string | null | undefined;
  /** Session-clock ms of audio received so far. */
  now(): number;
  readonly mode: "free" | "memorization";
  /** Research sessions expose `{from, to}` word refs; the report wants `{surah, fromAyah, toAyah}`. Both are accepted. */
  readonly passage: ReportPassage | { from: WordRef; to: WordRef } | null;
  readonly policy?: SessionPolicy;
  readonly capabilities: CapabilityEvent;
  readonly state?: SessionState;
  /** Whether VAD gates the session clock. */
  readonly vad?: boolean;
  /**
   * Internal audio tap: 16 kHz samples the listener receives, after resampling and gating, with the session-clock ms
   * of the first one. A third `sampleRate` argument, when given, must be 16000.
   */
  onAudio(fn: (pcm16k: Float32Array, startMs: number, sampleRate?: number) => void): () => void;
  subscribe(fn: (event: EngineEvent) => void): () => void;
  /** Resolves once queued pushes are processed, so events for every tapped sample are out. */
  idle?(): Promise<void>;
  /** Calls `fn` once when the session reaches `ended` or `disposed`. Without it the recorder polls `state`. */
  onEnded?(fn: () => void): () => void;
  /** Listener evidence so far (research `Session.evidence()`), for `evidence: true`. */
  evidence?(): PhonemeEvidence[];
}

export type RetainOptions = { beforeMs?: number; afterMs?: number } | { take: true; maxMs?: number; beforeMs?: number; afterMs?: number };

export interface RecorderOptions {
  /** Ring for incident clips (default `{beforeMs: 10_000, afterMs: 5_000}`) or the whole take (`{take: true, maxMs}`). */
  retain?: RetainOptions;
  /** App tag, required. */
  source: string;
  /** Pseudonymous app-provided id, at most 64 chars of `[A-Za-z0-9_-]`. */
  contributor?: string;
  /** Include the bounded phoneme evidence slice. Default false. */
  evidence?: boolean;
  /** Default consent for `report` and `contribution`; either call can pass its own. `send` must be true. */
  consent?: ReportConsent;
  /** How long buffers stay after the session ends, unless `dispose()` comes first. Default 10 min. */
  retainAfterEndMs?: number;
  /** Fills or overrides engine identity the capabilities event lacks (e.g. `sdkVersion`). */
  engine?: Partial<Omit<ReportEngine, "eventSchema">>;
  /** Overrides `session.vad`. */
  vad?: boolean;
  /** Wall clock, ms since the epoch. Default `Date.now`. */
  clock?: () => number;
}

export interface MarkInput {
  kind: IncidentKind;
  eventIds?: string[];
  /** Incident span on the session clock; default `[now, now]`. */
  startMs?: number;
  endMs?: number;
  claim?: Partial<IncidentClaim>;
}

export interface ReportInput {
  kind: IncidentKind;
  /** The checker event this report is about. */
  eventId?: string;
  eventIds?: string[];
  /** Clip anchor. Default: the linked checker events' span, else `session.now()`. */
  atMs?: number;
  /** Explicit incident span (overrides `atMs`). */
  startMs?: number;
  endMs?: number;
  claim?: Partial<IncidentClaim>;
  consent?: ReportConsent;
}

export interface ContributionInput {
  /** Incidents to include, typically `recorder.incidents()`; default none (a clean contribution). */
  incidents?: Incident[];
  consent?: ReportConsent;
}

export type RecorderState = "recording" | "ended" | "expired" | "disposed" | "failed";

export interface RecorderStats {
  state: RecorderState;
  /** Bytes allocated for audio. */
  capacityBytes: number;
  /** The memory bound from the contract for this configuration. */
  boundBytes: number;
  /** Retained audio span on the session clock. */
  retainedFromMs: number;
  retainedToMs: number;
  events: number;
}

export interface Recorder {
  readonly state: RecorderState;
  /** Why the recorder failed (tap misuse), when `state` is "failed". */
  readonly error: Error | null;
  /** Records an incident at session-clock now, synchronously. */
  mark(input: MarkInput): Incident;
  /** One incident clip, `[t - beforeMs, t + afterMs]`. Resolves once the after-context arrives or the session ends. */
  report(input: ReportInput): Promise<TilawaReport>;
  /** The whole retained take, from session start to now. */
  contribution(input?: ContributionInput): Promise<TilawaReport>;
  /** Every incident marked or reported so far, oldest first. */
  incidents(): Incident[];
  stats(): RecorderStats;
  /** Stops the tap and drops all retained audio and events. Pending reports reject. Idempotent. */
  dispose(): void;
}

export class RecorderError extends Error {
  constructor(message: string, readonly reason: "disposed" | "expired" | "failed" | "empty" | "limit" | "misuse", options?: { cause?: unknown }) {
    super(message);
    this.name = "RecorderError";
    if (options?.cause !== undefined) (this as { cause?: unknown }).cause = options.cause;
  }
}

export class ConsentError extends Error {
  constructor(message = "Reporting needs consent with send: true") {
    super(message);
    this.name = "ConsentError";
  }
}

interface Stored {
  seq: number;
  event: EngineEvent;
}

interface Waiter {
  untilSample: number;
  resolve(): void;
  reject(e: Error): void;
}

const CONTRIBUTOR_RE = /^[A-Za-z0-9_-]{1,64}$/;

function finiteNonNeg(v: unknown, name: string): number {
  if (typeof v !== "number" || !Number.isFinite(v) || v < 0) throw new RangeError(`${name} must be a finite number >= 0`);
  return v;
}

export function createRecorder(session: RecordableSession, options: RecorderOptions): Recorder {
  if (!options || typeof options.source !== "string" || !options.source) throw new TypeError("createRecorder needs options.source (app tag)");
  if (options.source.length > 200) throw new RangeError("options.source must be at most 200 chars");
  if (options.contributor !== undefined && !CONTRIBUTOR_RE.test(options.contributor))
    throw new RangeError("options.contributor must be 1-64 chars of [A-Za-z0-9_-]");
  const retain = options.retain ?? {};
  const take = "take" in retain && retain.take === true;
  const beforeMs = finiteNonNeg(retain.beforeMs ?? DEFAULT_BEFORE_MS, "retain.beforeMs");
  const afterMs = finiteNonNeg(retain.afterMs ?? DEFAULT_AFTER_MS, "retain.afterMs");
  if (beforeMs + afterMs > REPORT_LIMITS.maxAudioMs) throw new RangeError("retain.beforeMs + retain.afterMs must be at most 300 s");
  const maxMs = take ? finiteNonNeg((retain as { maxMs?: number }).maxMs ?? DEFAULT_TAKE_MS, "retain.maxMs") : 0;
  if (take && (maxMs <= 0 || maxMs > REPORT_LIMITS.maxAudioMs)) throw new RangeError("retain.maxMs must be in (0, 300000]");
  const retainAfterEndMs = finiteNonNeg(options.retainAfterEndMs ?? DEFAULT_RETAIN_AFTER_END_MS, "retainAfterEndMs");
  const clock = options.clock ?? Date.now;

  const capacityMs = take ? maxMs : beforeMs + afterMs + SLACK_MS;
  const capacity = Math.ceil(capacityMs * PER_MS);
  let store: PcmStore | null = new PcmStore(take ? "take" : "ring", capacity);

  let state: RecorderState = "recording";
  let error: Error | null = null;
  let seq = 0;
  let capabilities: CapabilityEvent = session.capabilities;
  let checkerEvents: Stored[] = [];
  let timedEvents: Stored[] = [];
  const marked: Incident[] = [];
  let waiters: Waiter[] = [];
  let expiry: ReturnType<typeof setTimeout> | null = null;
  let poll: ReturnType<typeof setInterval> | null = null;

  const offAudio = session.onAudio(onAudio);
  const offEvents = session.subscribe(onEvent);
  let offEnded: (() => void) | null = null;
  if (session.onEnded) offEnded = session.onEnded(sessionEnded);
  else if (session.state !== undefined) {
    poll = setInterval(() => {
      if (session.state === "ended" || session.state === "disposed") sessionEnded();
    }, END_POLL_MS);
    unref(poll);
  }
  if (session.state === "ended" || session.state === "disposed") sessionEnded();

  function onAudio(pcm: Float32Array, startMs: number, sampleRate?: number) {
    if (state !== "recording" || !store) return;
    try {
      if (sampleRate !== undefined && sampleRate !== RATE)
        throw new RecorderError(`The audio tap delivered ${sampleRate} Hz; it must deliver 16 kHz`, "misuse");
      if (!(pcm instanceof Float32Array)) throw new RecorderError("The audio tap must deliver a Float32Array", "misuse");
      if (typeof startMs !== "number" || !Number.isFinite(startMs) || startMs < 0)
        throw new RecorderError("The audio tap must deliver a session-clock startMs", "misuse");
      const at = Math.round(startMs * PER_MS);
      const next = store.next;
      if (next !== null && Math.abs(at - next) > CONTIGUITY_SAMPLES)
        throw new RecorderError(
          `Audio tap clock jumped: expected ${next / PER_MS} ms, got ${startMs} ms. The tap must deliver contiguous 16 kHz audio`,
          "misuse",
        );
      store.write(pcm, next ?? at);
      flushWaiters(false);
    } catch (e) {
      // Never throw into the session's audio path: reporting must not change what the engine hears.
      fail(e instanceof Error ? e : new Error(String(e)));
    }
  }

  function onEvent(event: EngineEvent) {
    if (state === "disposed" || state === "expired" || state === "failed") return;
    if (event.type === "capabilities") {
      capabilities = event;
      return;
    }
    const s = { seq: seq++, event };
    if (event.type === "checker") {
      checkerEvents.push(s);
      if (checkerEvents.length > MAX_CHECKER_EVENTS) checkerEvents = checkerEvents.slice(-MAX_CHECKER_EVENTS);
      return;
    }
    if (take && store && store.full && event.timeMs > store.range.end / PER_MS) return;
    timedEvents.push(s);
    if (!take && store && !store.empty) {
      // Keep position events no older than the ring, with headroom so pruning is amortised.
      const from = store.range.start / PER_MS;
      let old = 0;
      while (old < timedEvents.length && (timedEvents[old].event as { timeMs: number }).timeMs < from) old++;
      if (old > 64 || (old > 0 && old === timedEvents.length)) timedEvents.splice(0, old);
    }
    if (timedEvents.length > MAX_TIMED_EVENTS) timedEvents = timedEvents.slice(-MAX_TIMED_EVENTS);
  }

  function flushWaiters(ended: boolean) {
    if (!waiters.length) return;
    const end = store ? store.range.end : 0;
    const full = store?.full ?? false;
    const keep: Waiter[] = [];
    for (const w of waiters) {
      if (ended || full || end >= w.untilSample) w.resolve();
      else keep.push(w);
    }
    waiters = keep;
  }

  function rejectWaiters(e: Error) {
    const ws = waiters;
    waiters = [];
    for (const w of ws) w.reject(e);
  }

  function sessionEnded() {
    if (state !== "recording") return;
    state = "ended";
    stopWatching();
    offAudio();
    flushWaiters(true);
    expiry = setTimeout(expire, retainAfterEndMs);
    unref(expiry);
  }

  function stopWatching() {
    if (poll) clearInterval(poll);
    poll = null;
    offEnded?.();
    offEnded = null;
  }

  function release() {
    store?.release();
    store = null;
    checkerEvents = [];
    timedEvents = [];
  }

  function expire() {
    if (state !== "ended") return;
    state = "expired";
    offEvents();
    release();
    rejectWaiters(new RecorderError("The recorder's retention after session end expired", "expired"));
  }

  function fail(e: Error) {
    if (state !== "recording") return;
    state = "failed";
    error = e;
    stopWatching();
    offAudio();
    offEvents();
    release();
    rejectWaiters(new RecorderError(`The recorder stopped: ${e.message}`, "failed", { cause: e }));
  }

  function unusable(): RecorderError | null {
    if (state === "disposed") return new RecorderError("The recorder is disposed", "disposed");
    if (state === "expired") return new RecorderError("The recorder's retention after session end expired", "expired");
    if (state === "failed") return new RecorderError(`The recorder stopped: ${error?.message}`, "failed", { cause: error });
    return null;
  }

  function wallMs() {
    return clock();
  }

  function claimOf(c: Partial<IncidentClaim> | undefined): IncidentClaim {
    const claim: IncidentClaim = { answer: c?.answer ?? null };
    if (c?.question !== undefined) claim.question = c.question;
    if (c?.note !== undefined) claim.note = c.note;
    if (c?.text !== undefined) claim.text = c.text;
    if (claim.note !== undefined && claim.note.length > REPORT_LIMITS.maxNoteChars)
      throw new RangeError(`claim.note must be at most ${REPORT_LIMITS.maxNoteChars} chars`);
    return claim;
  }

  function addIncident(inc: Incident) {
    if (marked.length >= REPORT_LIMITS.maxIncidents)
      throw new RecorderError(`A recorder holds at most ${REPORT_LIMITS.maxIncidents} incidents`, "limit");
    marked.push(inc);
  }

  function linkedSpan(ids: string[]): { start: number; end: number } | null {
    let start = Infinity;
    let end = -Infinity;
    for (const s of checkerEvents) {
      const e = s.event as CheckerEvent;
      if (ids.includes(e.id)) {
        start = Math.min(start, e.timeMs);
        end = Math.max(end, e.timeMs);
      }
    }
    return Number.isFinite(start) ? { start, end } : null;
  }

  function newIncident(input: MarkInput | ReportInput, ids: string[]): Incident {
    const nowMs = session.now();
    let span: { start: number; end: number };
    if (input.startMs !== undefined || input.endMs !== undefined) {
      const start = input.startMs ?? input.endMs!;
      span = { start, end: input.endMs ?? start };
    } else if ("atMs" in input && input.atMs !== undefined) span = { start: input.atMs, end: input.atMs };
    else span = ("eventId" in input || ids.length ? linkedSpan(ids) : null) ?? { start: nowMs, end: nowMs };
    if (!Number.isFinite(span.start) || !Number.isFinite(span.end) || span.start < 0 || span.end < span.start)
      throw new RangeError("An incident needs 0 <= startMs <= endMs");
    return {
      id: uuidV4(),
      kind: input.kind,
      startMs: span.start,
      endMs: span.end,
      eventIds: ids,
      markedAt: { sessionMs: nowMs, wallMs: wallMs() },
      claim: claimOf(input.claim),
    };
  }

  function consentOf(c: ReportConsent | undefined): ReportConsent {
    const consent = c ?? options.consent;
    if (!consent || consent.send !== true) throw new ConsentError();
    if (typeof consent.policy !== "string" || !consent.policy || typeof consent.version !== "string" || !consent.version)
      throw new ConsentError("Consent needs a policy and a version");
    return {
      policy: consent.policy,
      version: consent.version,
      send: true,
      training: consent.training === true,
      publicDataset: consent.publicDataset === true,
    };
  }

  function waitForSample(untilSample: number): Promise<void> {
    const bad = unusable();
    if (bad) return Promise.reject(bad);
    const end = store ? store.range.end : 0;
    if (state !== "recording" || end >= untilSample || store?.full) return Promise.resolve();
    return new Promise((resolve, reject) => waiters.push({ untilSample, resolve, reject }));
  }

  async function settle(): Promise<void> {
    if (session.idle && state === "recording") await session.idle();
  }

  async function build(requested: { startMs: number; endMs: number }, incidents: Incident[], consent: ReportConsent): Promise<TilawaReport> {
    const bad = unusable();
    if (bad || !store) throw bad ?? new RecorderError("The recorder holds no audio", "empty");
    const range = store.range;
    let startMs = Math.max(requested.startMs, Math.ceil(range.start / PER_MS));
    const endMs = Math.min(requested.endMs, Math.floor(range.end / PER_MS));
    if (store.empty || endMs <= startMs) throw new RecorderError("No retained audio in the requested span", "empty");
    if (endMs - startMs > REPORT_LIMITS.maxAudioMs) startMs = endMs - REPORT_LIMITS.maxAudioMs;
    const pcm = store.read(startMs * PER_MS, endMs * PER_MS);
    const wav = encodeWav(pcm);
    const linked = new Set(incidents.flatMap((i) => i.eventIds));
    const events = collectEvents(startMs, endMs, linked);
    const evidence = options.evidence ? clipEvidence(session.evidence?.() ?? [], startMs, endMs) : undefined;
    const report: TilawaReport = {
      schema: REPORT_SCHEMA,
      reportId: uuidV4(),
      sessionId: session.id,
      createdAt: new Date(wallMs()).toISOString(),
      source: options.source,
      ...(options.contributor !== undefined ? { contributor: options.contributor } : {}),
      engine: engineOf(capabilities, options.engine),
      session: {
        mode: session.mode,
        passage: passageOf(session.passage),
        policy: { ...(session.policy ?? {}) },
        startedAt: session.startedAt ?? new Date(wallMs() - session.now()).toISOString(),
        vad: options.vad ?? session.vad ?? false,
      },
      audio: {
        format: "pcm_s16le",
        sampleRate: 16000,
        channels: 1,
        startMs,
        durationMs: endMs - startMs,
        sha256: await sha256Hex(wav),
        requested: { ...requested },
        truncatedStart: startMs > requested.startMs,
        truncatedEnd: endMs < requested.endMs,
      },
      incidents: incidents.map((i) => structuredCloneIncident(i)),
      observations: evidence ? { events, evidence } : { events },
      consent,
    };
    fitJson(report);
    const problems = reportProblems(report);
    if (problems.length) throw new ReportValidationError(problems);
    return attachAudio(report, wav);
  }

  function collectEvents(startMs: number, endMs: number, linked: Set<string>): EngineEvent[] {
    const picked: Stored[] = [];
    for (const s of checkerEvents) {
      const e = s.event as CheckerEvent;
      if ((e.timeMs >= startMs && e.timeMs <= endMs) || linked.has(e.id)) picked.push(s);
    }
    for (const s of timedEvents) {
      const t = (s.event as { timeMs: number }).timeMs;
      if (t >= startMs && t <= endMs) picked.push(s);
    }
    picked.sort((a, b) => a.seq - b.seq);
    return [capabilities, ...picked.map((s) => s.event)];
  }

  const recorder: Recorder = {
    get state() {
      return state;
    },
    get error() {
      return error;
    },
    mark(input) {
      const bad = unusable();
      if (bad) throw bad;
      const inc = newIncident(input, [...(input.eventIds ?? [])]);
      addIncident(inc);
      return inc;
    },
    async report(input) {
      const consent = consentOf(input.consent);
      const bad = unusable();
      if (bad) throw bad;
      const ids = [...new Set([...(input.eventId !== undefined ? [input.eventId] : []), ...(input.eventIds ?? [])])];
      const inc = newIncident(input, ids);
      addIncident(inc);
      const requested = {
        startMs: Math.max(0, Math.floor(inc.startMs - beforeMs)),
        endMs: Math.ceil(inc.endMs + afterMs),
      };
      try {
        await waitForSample(requested.endMs * PER_MS);
        await settle();
        return await build(requested, [inc], consent);
      } catch (e) {
        // A report that could not be built leaves no incident behind.
        const at = marked.indexOf(inc);
        if (at >= 0) marked.splice(at, 1);
        throw e;
      }
    },
    async contribution(input = {}) {
      const consent = consentOf(input.consent);
      const bad = unusable();
      if (bad) throw bad;
      const incidents = input.incidents ?? [];
      if (incidents.length > REPORT_LIMITS.maxIncidents)
        throw new ReportValidationError([`at most ${REPORT_LIMITS.maxIncidents} incidents`]);
      await settle();
      const endMs = Math.floor(store && !store.empty ? Math.max(session.now(), store.range.end / PER_MS) : session.now());
      return build({ startMs: 0, endMs }, incidents, consent);
    },
    incidents() {
      return marked.map(structuredCloneIncident);
    },
    stats() {
      const range = store?.range ?? { start: 0, end: 0 };
      return {
        state,
        capacityBytes: store?.bytes ?? 0,
        boundBytes: capacity * 2,
        retainedFromMs: range.start / PER_MS,
        retainedToMs: range.end / PER_MS,
        events: checkerEvents.length + timedEvents.length,
      };
    },
    dispose() {
      if (state === "disposed") return;
      const wasRecording = state === "recording";
      state = "disposed";
      stopWatching();
      if (wasRecording) offAudio();
      offEvents();
      if (expiry) clearTimeout(expiry);
      expiry = null;
      release();
      rejectWaiters(new RecorderError("The recorder is disposed", "disposed"));
    },
  };
  return recorder;
}

function structuredCloneIncident(i: Incident): Incident {
  return { ...i, eventIds: [...i.eventIds], markedAt: { ...i.markedAt }, claim: { ...i.claim } };
}

function unref(t: unknown) {
  (t as { unref?: () => void } | null)?.unref?.();
}

export function passageOf(p: RecordableSession["passage"]): ReportPassage | null {
  if (!p) return null;
  if ("surah" in p) return { surah: p.surah, fromAyah: p.fromAyah, toAyah: p.toAyah };
  return { surah: p.from.surah, fromAyah: p.from.ayah, toAyah: p.to.ayah };
}

/** Engine identity from the capabilities event. `modelVersion` there is a model id such as `name@version`. */
export function engineOf(caps: CapabilityEvent, override: Partial<Omit<ReportEngine, "eventSchema">> = {}): ReportEngine {
  const at = caps.modelVersion.lastIndexOf("@");
  const modelId = at > 0 ? caps.modelVersion.slice(0, at) : caps.modelVersion;
  const modelVersion = at > 0 ? caps.modelVersion.slice(at + 1) : caps.modelVersion;
  const engine: ReportEngine = {
    sdkVersion: override.sdkVersion ?? caps.sdkVersion ?? "unknown",
    modelId: override.modelId ?? modelId,
    modelVersion: override.modelVersion ?? modelVersion,
    modelSha256: override.modelSha256 !== undefined ? override.modelSha256 : (caps.modelSha256 ?? null),
    referenceVersion: override.referenceVersion ?? caps.referenceVersion,
    eventSchema: 1,
  };
  const tier = override.tier ?? caps.tier;
  const backend = override.backend ?? caps.backend;
  if (tier !== undefined) engine.tier = tier;
  if (backend !== undefined) engine.backend = backend;
  return engine;
}

/**
 * Evidence clipped to `[startMs, endMs]`, assuming frame `f` starts at session ms `f * frameMs` (the listener runs on
 * the session clock). Phonemes overlapping the span are kept; posterior rows are sliced to it.
 */
export function clipEvidence(chunks: PhonemeEvidence[], startMs: number, endMs: number): PhonemeEvidence[] {
  const out: PhonemeEvidence[] = [];
  for (const c of chunks) {
    const ms = c.frameMs;
    const phonemes = c.phonemes.filter((p) => p.start * ms < endMs && (p.end + 1) * ms > startMs).map((p) => ({ ...p }));
    let posteriors: PhonemeEvidence["posteriors"];
    if (c.posteriors) {
      const q = c.posteriors;
      const from = Math.max(q.startFrame, Math.ceil(startMs / ms));
      const to = Math.min(q.startFrame + q.nFrames, Math.ceil(endMs / ms));
      if (to > from) {
        const a = (from - q.startFrame) * q.k;
        const b = (to - q.startFrame) * q.k;
        posteriors = {
          k: q.k,
          startFrame: from,
          nFrames: to - from,
          ids: Array.from({ length: b - a }, (_, i) => q.ids[a + i]),
          probs: Array.from({ length: b - a }, (_, i) => q.probs[a + i]),
        };
      }
    }
    if (!phonemes.length && !posteriors) continue;
    const clipped: PhonemeEvidence = { inventory: c.inventory, frameMs: ms, phonemes };
    if (posteriors) clipped.posteriors = posteriors;
    if (c.decodedTo !== undefined) clipped.decodedTo = c.decodedTo;
    out.push(clipped);
  }
  return out;
}

const utf8 = new TextEncoder();

/** Keeps `report.json` under 4 MiB: drops posteriors, then tracker traces, then evidence. */
function fitJson(report: TilawaReport) {
  const size = () => utf8.encode(JSON.stringify(report)).length;
  if (size() <= REPORT_LIMITS.maxJsonBytes) return;
  const ev = report.observations.evidence;
  if (ev) report.observations.evidence = ev.map(({ posteriors: _p, ...rest }) => rest);
  if (size() <= REPORT_LIMITS.maxJsonBytes) return;
  report.observations.events = report.observations.events.map((e) => {
    if (e.type !== "position" || !e.trace) return e;
    const { trace: _t, ...rest } = e;
    return rest;
  });
  if (size() <= REPORT_LIMITS.maxJsonBytes) return;
  delete report.observations.evidence;
  if (size() > REPORT_LIMITS.maxJsonBytes) throw new ReportValidationError(["report.json is larger than 4 MiB"]);
}
