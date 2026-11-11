import { randomUUID } from "node:crypto";
import type { RecordableSession, SessionState } from "../../src/report/index.js";
import type { CapabilityEvent, CheckerEvent, EngineEvent, PhonemeEvidence, PositionEvent } from "../../src/report/types.js";

/** Session sample `s` carries a ramp that survives Float32 -> Int16 exactly. */
export const rampInt16 = (s: number) => (s % 2000) - 1000;
const ramp = (s: number) => rampInt16(s) / 32768;

export const CAPS: CapabilityEvent = {
  type: "capabilities",
  v: 1,
  modelVersion: "zipformer-tajweed@2026.11.1",
  referenceVersion: "ref-v2",
  supported: ["wrong_word", "omitted_word", "added_word", "skipped_passage"],
  unsupported: ["short_vowel"],
  sdkVersion: "1.0.0",
  modelSha256: "a".repeat(64),
  tier: "full",
  backend: "wasm",
};

export interface FakeOptions {
  /** Expose `onEnded`; without it the recorder polls `state`. Default true. */
  onEnded?: boolean;
  mode?: "free" | "memorization";
  evidence?: PhonemeEvidence[];
}

/** A deterministic stand-in for the V1 Session: taps audio, emits events, ends. */
export class FakeSession implements RecordableSession {
  readonly id = randomUUID();
  startedAt: string | null = null;
  readonly mode: "free" | "memorization";
  readonly passage;
  readonly policy = { minConfidence: 0.6 };
  readonly capabilities = CAPS;
  state: SessionState = "active";
  readonly vad = false;
  readonly audioSubs = new Set<(pcm: Float32Array, startMs: number, rate?: number) => void>();
  readonly eventSubs = new Set<(e: EngineEvent) => void>();
  private readonly endedSubs = new Set<() => void>();
  /** Session samples delivered so far. */
  samples = 0;
  readonly onEnded?: (fn: () => void) => () => void;
  evidenceChunks: PhonemeEvidence[];

  constructor(options: FakeOptions = {}) {
    this.mode = options.mode ?? "free";
    this.passage = this.mode === "memorization" ? { from: { surah: 18, ayah: 60, word: 1 }, to: { surah: 18, ayah: 70, word: 9 } } : null;
    this.evidenceChunks = options.evidence ?? [];
    if (options.onEnded !== false)
      this.onEnded = (fn) => {
        this.endedSubs.add(fn);
        return () => this.endedSubs.delete(fn);
      };
  }

  now() {
    return this.samples / 16;
  }

  onAudio(fn: (pcm: Float32Array, startMs: number, rate?: number) => void) {
    this.audioSubs.add(fn);
    return () => void this.audioSubs.delete(fn);
  }

  subscribe(fn: (e: EngineEvent) => void) {
    this.eventSubs.add(fn);
    fn(this.capabilities);
    return () => void this.eventSubs.delete(fn);
  }

  evidence() {
    return this.evidenceChunks;
  }

  /** Delivers `n` samples of the ramp through the tap, in one chunk. */
  pushSamples(n: number) {
    if (this.state !== "active") throw new Error("ended");
    if (this.startedAt === null) this.startedAt = "2026-11-11T10:00:00.000Z";
    const pcm = new Float32Array(n);
    for (let i = 0; i < n; i++) pcm[i] = ramp(this.samples + i);
    const startMs = this.samples / 16;
    this.samples += n;
    for (const fn of this.audioSubs) fn(pcm, startMs);
  }

  /** Delivers `ms` of audio in chunks of `chunk` samples (not ms-aligned by default). */
  pushMs(ms: number, chunk = 333) {
    let left = Math.round(ms * 16);
    while (left > 0) {
      const n = Math.min(chunk, left);
      this.pushSamples(n);
      left -= n;
    }
  }

  emit(e: EngineEvent) {
    for (const fn of this.eventSubs) fn(e);
  }

  checker(id: string, timeMs: number, state: CheckerEvent["state"] = "confirmed"): CheckerEvent {
    const e: CheckerEvent = { type: "checker", v: 1, id, timeMs, state, errorClass: "wrong_word", location: { surah: 1, ayah: 2, word: 3 }, confidence: 0.9 };
    this.emit(e);
    return e;
  }

  position(timeMs: number): PositionEvent {
    const e: PositionEvent = { type: "position", v: 1, timeMs, referenceVersion: "ref-v2", expected: null, observed: [], state: "locked" };
    this.emit(e);
    return e;
  }

  end() {
    this.state = "ended";
    for (const fn of [...this.endedSubs]) fn();
  }
}

export const CONSENT = { policy: "tilawa-report", version: "2026-11-11", send: true as const, training: true, publicDataset: false };
