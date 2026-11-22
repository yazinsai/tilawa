/**
 * The public session: the research session (algorithm unchanged) inside a lifecycle (contract §3) and an audio tap.
 * Nothing here reads or changes what the listener, tracker or checker compute.
 */
import type {
  CapabilityEvent as ResearchCapabilityEvent,
  CheckerEvent,
  PhonemeEvidence,
  PositionEvent,
  WordRef,
} from "./engine/index.js";
import { Resampler } from "./runtime/audio.js";
import { createEngine as createResearchEngine } from "./runtime/session.js";
import type {
  EngineOptions as ResearchEngineOptions,
  MemorizationPassage,
  Session as ResearchSession,
  SessionPolicy,
} from "./runtime/session.js";
import { DEFAULTS } from "./defaults.js";
import { SessionEndedError } from "./errors.js";
import { uuid } from "./util.js";
import { SDK_VERSION } from "./version.js";

export type { MemorizationPassage, SessionPolicy };

/** Schema v1 `capabilities`, plus the optional fields V1 adds. */
export interface CapabilityEvent extends ResearchCapabilityEvent {
  sdkVersion?: string;
  /** Registry version of the model, e.g. `1-int8` (`modelVersion` above is the model id, e.g. `spliced-clean`). */
  modelRelease?: string;
  modelSha256?: string;
  tier?: string;
  backend?: string;
}

export type EngineEvent = PositionEvent | CheckerEvent | CapabilityEvent;

/** `SessionPolicy` with every default filled in. */
export interface ResolvedSessionPolicy {
  countSelfCorrections: boolean;
  confirm: { after: "delay"; ms: number } | { after: "next-word" };
  minConfidence: number;
  evidenceMs: number;
}

/** The declared passage: its first and last words, and the ayah range it was declared with. */
export interface DeclaredPassage extends MemorizationPassage {
  from: WordRef;
  to: WordRef;
}

export type SessionState = "active" | "ending" | "ended" | "disposed";

/** Receives exactly the 16 kHz samples the listener gets, in order; `startMs` is the first sample's session-clock time. */
export type AudioTap = (pcm16k: Float32Array, startMs: number) => void;

export interface TilawaSession {
  /** uuid v4. */
  readonly id: string;
  readonly mode: "memorization" | "free";
  readonly passage: DeclaredPassage | null;
  readonly policy: Readonly<ResolvedSessionPolicy>;
  readonly capabilities: CapabilityEvent;
  readonly state: SessionState;
  /** ISO wall-clock time of the first sample pushed; null before any audio. */
  readonly startedAt: string | null;
  /** True while `listen()` gates this session's audio with the VAD. */
  readonly vad: boolean;
  /** Session-clock ms: 16 kHz audio delivered to the listener so far (after any VAD gating). */
  now(): number;
  /** The event channel. Called with `capabilities` first, then every event. Returns an unsubscribe function. */
  subscribe(listener: (event: EngineEvent) => void): () => void;
  /**
   * Taps the audio the listener receives (after resampling to 16 kHz). Never changes it; don't mutate the array.
   * Returns an unsubscribe function.
   */
  onAudio(fn: AudioTap): () => void;
  /**
   * Audio at any sample rate. Calls run one at a time in call order; callers need not await. Rejects with
   * `SessionEndedError` once `end()` has started, or with the backend error if inference fails (the session then ends).
   * Resolves with the events this audio produced, which subscribers also get (a convenience for tests).
   */
  pushAudio(samples: Float32Array, sampleRate: number): Promise<EngineEvent[]>;
  /** Drains the listener and decides every open error. Idempotent: every call returns the same promise. */
  end(): Promise<EngineEvent[]>;
  /** Drops subscribers, taps and retained buffers and rejects queued pushes. Does not release the shared model. */
  dispose(): void;
  /** Resolves once every queued `pushAudio`/`end` has run: all events for the audio received so far are out. */
  idle(): Promise<void>;
  /**
   * Called once when the session ends (`end()` settles, `endEvidence()`, or an inference failure) or is disposed;
   * at once if that already happened. Returns an unsubscribe function.
   */
  onEnded(fn: () => void): () => void;
  /** Evidence from another listener, bypassing the model (tests, replays). Don't mix with `pushAudio`. */
  pushEvidence(evidence: PhonemeEvidence): EngineEvent[];
  endEvidence(): EngineEvent[];
  /** Confirmed errors, plus resolved ones when the policy counts self-corrections. */
  mistakes(): CheckerEvent[];
  /** The evidence so far when the engine records it (`recordEvidence`); otherwise empty. */
  evidence(): PhonemeEvidence[];
}

/** Alias of `TilawaSession`. */
export type Session = TilawaSession;

export interface EngineOptions extends ResearchEngineOptions {
  /** Filled into `capabilities`; `backend` and `modelSha256` default to the model's. */
  info?: { tier?: string; backend?: string; modelSha256?: string; modelRelease?: string };
}

export interface Engine {
  readonly capabilities: CapabilityEvent;
  startMemorization(passage: MemorizationPassage, policy?: SessionPolicy): TilawaSession;
  startFreeRecitation(policy?: SessionPolicy): TilawaSession;
}

/** Internal: the hooks `listen()` and the Tilawa instance use. */
export const internals = new WeakMap<TilawaSession, { setVad(on: boolean): void; onClose(fn: () => void): void }>();

export function resolvePolicy(p: SessionPolicy = {}, frameMs: number): ResolvedSessionPolicy {
  return {
    countSelfCorrections: p.countSelfCorrections ?? DEFAULTS.policy.countSelfCorrections,
    confirm: p.confirm ? { ...p.confirm } : { ...DEFAULTS.policy.confirm },
    minConfidence: p.minConfidence ?? DEFAULTS.policy.minConfidence,
    evidenceMs: p.evidenceMs ?? frameMs,
  };
}

export function createEngine(options: EngineOptions): Engine {
  const inner = createResearchEngine(options);
  const { model } = options;
  const info = options.info ?? {};
  const backend = info.backend ?? model.backend.name;
  const modelSha256 = info.modelSha256 ?? model.manifest.sha256;
  let caps: CapabilityEvent | null = null;
  const capsOf = (s: ResearchSession): CapabilityEvent =>
    (caps ??= {
      ...s.capabilities,
      sdkVersion: SDK_VERSION,
      ...(info.modelRelease && { modelRelease: info.modelRelease }),
      ...(modelSha256 && { modelSha256 }),
      ...(info.tier && { tier: info.tier }),
      ...(backend && { backend }),
    });
  const frameMs = model.manifest.frameMs;
  const rate = model.manifest.sampleRate;
  const engine: Engine = {
    get capabilities() {
      return caps ?? capsOf(inner.startFreeRecitation());
    },
    startMemorization(passage, policy) {
      const s = inner.startMemorization(passage, policy);
      const declared: DeclaredPassage = { surah: passage.surah, fromAyah: passage.fromAyah, toAyah: passage.toAyah, ...s.passage! };
      return wrap(s, declared, resolvePolicy(policy, frameMs), capsOf(s), rate);
    },
    startFreeRecitation(policy) {
      const s = inner.startFreeRecitation(policy);
      return wrap(s, null, resolvePolicy(policy, frameMs), capsOf(s), rate);
    },
  };
  return engine;
}

/** Rethrown outside the engine loop, so one failing subscriber can't stop the others or the session. */
const raise = (e: unknown) => queueMicrotask(() => {
  throw e;
});

interface Job {
  run: () => Promise<EngineEvent[]>;
  resolve: (events: EngineEvent[]) => void;
  reject: (e: unknown) => void;
}

function wrap(
  research: ResearchSession,
  passage: DeclaredPassage | null,
  policy: ResolvedSessionPolicy,
  capabilities: CapabilityEvent,
  rate: number,
): TilawaSession {
  let inner: ResearchSession | null = research;
  let state: SessionState = "active";
  let failed = false;
  let vad = false;
  let startedAt: string | null = null;
  /** Samples at `rate` delivered to the listener. */
  let delivered = 0;
  const subscribers = new Set<(e: EngineEvent) => void>();
  const taps = new Set<AudioTap>();
  const closers: (() => void)[] = [];
  const jobs: Job[] = [];
  let running = false;
  const idlers: (() => void)[] = [];
  let ending: Promise<EngineEvent[]> | null = null;

  // The research session calls a new subscriber with its capabilities at once; ours get the V1 ones instead.
  let attached = false;
  research.subscribe((e) => {
    if (!attached) return;
    for (const s of [...subscribers]) {
      try {
        s(e);
      } catch (err) {
        raise(err);
      }
    }
  });
  attached = true;

  // The listener resamples anything not at the model rate with its own Resampler; the tap mirrors it exactly
  // (same class, same input, the same swap on a rate change, the same flush at end) on a second instance.
  let resampler: Resampler | null = null;
  const tapIn = (samples: Float32Array, sampleRate: number) => {
    let pcm = samples;
    if (sampleRate !== rate) {
      if (!resampler || resampler.from !== sampleRate) resampler = new Resampler(sampleRate, rate);
      pcm = resampler.push(samples);
    }
    deliver(pcm);
  };
  const deliver = (pcm: Float32Array) => {
    const startMs = (delivered * 1000) / rate;
    delivered += pcm.length;
    if (!taps.size || !pcm.length) return;
    const copy = Float32Array.from(pcm);
    for (const t of [...taps]) {
      try {
        t(copy, startMs);
      } catch (err) {
        raise(err);
      }
    }
  };

  const close = () => {
    for (const f of closers.splice(0)) f();
  };
  const pump = async () => {
    if (running) return;
    running = true;
    while (jobs.length) {
      const job = jobs.shift()!;
      try {
        job.resolve(await job.run());
      } catch (e) {
        job.reject(e);
      }
    }
    running = false;
    for (const f of idlers.splice(0)) f();
  };
  const enqueue = (run: () => Promise<EngineEvent[]>) =>
    new Promise<EngineEvent[]>((resolve, reject) => {
      jobs.push({ run, resolve, reject });
      void pump();
    });
  const ended = (why: string) => new SessionEndedError(why);

  const session: TilawaSession = {
    id: uuid(),
    mode: research.mode,
    passage,
    policy: Object.freeze(policy),
    capabilities,
    get state() {
      return state;
    },
    get startedAt() {
      return startedAt;
    },
    get vad() {
      return vad;
    },
    now: () => (delivered * 1000) / rate,
    subscribe(fn) {
      if (state === "disposed") return () => {};
      subscribers.add(fn);
      fn(capabilities);
      return () => void subscribers.delete(fn);
    },
    onAudio(fn) {
      if (state === "disposed") return () => {};
      taps.add(fn);
      return () => void taps.delete(fn);
    },
    pushAudio(samples, sampleRate) {
      if (state !== "active") return Promise.reject(ended(`pushAudio after the session ${state === "disposed" ? "was disposed" : "ended"}`));
      if (startedAt === null && samples.length) startedAt = new Date().toISOString();
      return enqueue(async () => {
        if (failed || !inner) throw ended("The session ended after an earlier inference failure");
        tapIn(samples, sampleRate);
        try {
          return await inner.pushAudio(samples, sampleRate);
        } catch (e) {
          failed = true;
          if (state !== "disposed") state = "ended";
          close();
          throw e;
        }
      });
    },
    end() {
      if (ending) return ending;
      if (state === "disposed") return Promise.reject(ended("end() after the session was disposed"));
      if (state === "ended") return (ending = Promise.resolve([]));
      state = "ending";
      ending = enqueue(async () => {
        if (failed || !inner) return [];
        // The listener flushes its resampler's tail into the model at end; so does the tap.
        if (resampler) deliver(resampler.flush());
        return inner.end();
      });
      ending.then(
        () => {
          if (state !== "disposed") state = "ended";
          close();
        },
        () => {
          if (state !== "disposed") state = "ended";
          close();
        },
      );
      return ending;
    },
    dispose() {
      if (state === "disposed") return;
      state = "disposed";
      subscribers.clear();
      taps.clear();
      for (const job of jobs.splice(0)) job.reject(ended("The session was disposed"));
      if (!running) for (const f of idlers.splice(0)) f();
      inner = null;
      close();
    },
    pushEvidence(e) {
      if (state !== "active" || !inner) throw ended("pushEvidence after the session ended");
      return inner.pushEvidence(e);
    },
    endEvidence() {
      if (state !== "active" || !inner) throw ended("endEvidence after the session ended");
      const events = inner.endEvidence();
      state = "ended";
      ending = Promise.resolve(events);
      close();
      return events;
    },
    idle: () => (running || jobs.length ? new Promise<void>((r) => void idlers.push(r)) : Promise.resolve()),
    onEnded(fn) {
      if (state === "ended" || state === "disposed") {
        fn();
        return () => {};
      }
      let on = true;
      closers.push(() => on && fn());
      return () => void (on = false);
    },
    mistakes: () => inner?.mistakes() ?? [],
    evidence: () => inner?.evidence() ?? [],
  };
  internals.set(session, {
    setVad: (on) => void (vad = on),
    onClose: (fn) => (state === "ended" || state === "disposed" ? fn() : void closers.push(fn)),
  });
  return session;
}
