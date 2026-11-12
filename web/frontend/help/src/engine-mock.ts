// `?mock=1`: a scripted engine for building and testing /help without a model. It walks through real ayahs word by
// word, raises a confirmed, a resolved and an uncertain warning and one place wobble, and feeds synthetic audio
// through the same audio tap the real session offers, so the real report recorder captures a real WAV.
//
// Query knobs: speed=N (N× audio per tick, default 1), mic=denied|unavailable|insecure|skip, mocksurah=S (start).
import type { EngineHost } from "./engine";
import { loadQuran } from "./quran";
import type { CapabilityEvent, CheckerEvent, EngineEvent, PositionEvent, WordRef } from "./types";

const RATE = 16_000;
const LEAD_MS = 1_500;
const WORD_MS = 650;
const TICK_MS = 100;

export interface MockParams {
  speed: number;
  mic: "fake" | "denied" | "unavailable" | "insecure" | "skip";
  surah: number;
}

export function mockParams(search = globalThis.location?.search ?? ""): MockParams {
  const q = new URLSearchParams(search);
  const mic = q.get("mic");
  return {
    speed: Math.max(1, Math.min(20, Number(q.get("speed")) || 1)),
    mic: mic === "denied" || mic === "unavailable" || mic === "insecure" || mic === "skip" ? mic : "fake",
    surah: Math.max(1, Math.min(114, Number(q.get("mocksurah")) || 1)),
  };
}

const CAPS: CapabilityEvent = {
  type: "capabilities",
  v: 1,
  modelVersion: "mock-engine@0.0.0",
  referenceVersion: "ref-v2",
  supported: ["wrong_word", "omitted_word", "added_word", "skipped_passage"],
  unsupported: [],
  sdkVersion: "mock",
};

function micError(reason: "denied" | "unavailable" | "insecure-context") {
  return Object.assign(new Error(`mock microphone: ${reason}`), { name: "MicrophoneError", reason });
}

/** The scripted timeline: what the engine "hears" at each session-clock ms. */
export function mockScript(words: WordRef[]): EngineEvent[] {
  const out: EngineEvent[] = [];
  const at = (n: number) => LEAD_MS + n * WORD_MS;
  const pos = (timeMs: number, n: number | null, state: PositionEvent["state"]): PositionEvent => ({
    type: "position",
    v: 1,
    timeMs,
    referenceVersion: CAPS.referenceVersion,
    expected: null,
    observed: n === null ? [] : [{ word: words[n], span: { start: words[0], end: words[n] }, confidence: 0.9, cost: 0, per: 0 }],
    state,
  });
  const check = (timeMs: number, id: string, n: number, state: CheckerEvent["state"]): CheckerEvent => ({
    type: "checker",
    v: 1,
    id,
    timeMs,
    state,
    errorClass: "wrong_word",
    location: words[n],
    confidence: 0.8,
  });
  out.push(pos(400, null, "acquiring"));
  for (let n = 0; n < words.length; n++) {
    const t = at(n);
    out.push(pos(t, n, n === 22 ? "ambiguous" : "locked"));
    if (n === 4) out.push(check(t + 300, "mock-w1", n, "suspected"), check(t + 1200, "mock-w1", n, "confirmed"));
    if (n === 10) out.push(check(t + 300, "mock-w2", n, "suspected"), check(t + 1200, "mock-w2", n, "resolved"));
    if (n === 16) out.push(check(t + 300, "mock-w3", n, "suspected"), check(t + 1500, "mock-w3", n, "uncertain"));
  }
  return out.sort((a, b) => (a as { timeMs: number }).timeMs - (b as { timeMs: number }).timeMs);
}

/** A voice-ish tone per word with short gaps, so clips are audibly different. */
function synth(fromSample: number, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const s = fromSample + i;
    const ms = (s * 1000) / RATE;
    if (ms < LEAD_MS) {
      out[i] = (Math.random() - 0.5) * 0.002;
      continue;
    }
    const k = Math.floor((ms - LEAD_MS) / WORD_MS);
    const inWord = (ms - LEAD_MS) % WORD_MS;
    const env = inWord < 560 ? Math.sin((Math.PI * inWord) / 560) : 0;
    const f = 140 + ((k * 37) % 120);
    out[i] = env * (0.18 * Math.sin((2 * Math.PI * f * s) / RATE) + 0.06 * Math.sin((2 * Math.PI * 2.5 * f * s) / RATE)) + (Math.random() - 0.5) * 0.002;
  }
  return out;
}

class MockSession {
  readonly id = crypto.randomUUID();
  readonly mode = "free" as const;
  readonly passage = null;
  readonly policy = {};
  readonly vad = false;
  readonly capabilities = CAPS;
  startedAt: string | null = null;
  state: "active" | "ending" | "ended" | "disposed" = "active";
  private samples = 0;
  private next = 0;
  private subs = new Set<(e: EngineEvent) => void>();
  private taps = new Set<(pcm: Float32Array, startMs: number) => void>();
  private ended = new Set<() => void>();

  private script: EngineEvent[];
  constructor(script: EngineEvent[]) {
    this.script = script;
  }

  now() {
    return (this.samples * 1000) / RATE;
  }
  onAudio(fn: (pcm: Float32Array, startMs: number) => void) {
    this.taps.add(fn);
    return () => void this.taps.delete(fn);
  }
  subscribe(fn: (e: EngineEvent) => void) {
    fn(CAPS);
    this.subs.add(fn);
    return () => void this.subs.delete(fn);
  }
  idle() {
    return Promise.resolve();
  }
  onEnded(fn: () => void) {
    this.ended.add(fn);
    return () => void this.ended.delete(fn);
  }
  evidence() {
    return [];
  }
  /** Synthesizes `n` samples, taps them, then emits every scripted event now due. */
  feed(n: number) {
    if (this.state !== "active") return;
    this.startedAt ??= new Date().toISOString();
    const startMs = this.now();
    const pcm = synth(this.samples, n);
    this.samples += n;
    for (const t of this.taps) t(pcm, startMs);
    const now = this.now();
    while (this.next < this.script.length && (this.script[this.next] as { timeMs: number }).timeMs <= now) {
      const e = this.script[this.next++];
      for (const s of this.subs) s(e);
    }
  }
  async pushAudio() {
    return [];
  }
  async end() {
    if (this.state === "active") {
      this.state = "ended";
      for (const f of this.ended) f();
    }
    return [];
  }
  dispose() {
    this.state = "disposed";
    for (const f of this.ended) f();
    this.subs.clear();
    this.taps.clear();
  }
}

async function wordsFrom(surah: number, count: number): Promise<WordRef[]> {
  const quran = await loadQuran();
  const out: WordRef[] = [];
  for (let s = surah; s <= 114 && out.length < count; s++) {
    const info = quran.surahs[s - 1];
    for (let a = 1; a <= (info?.ayahs ?? 0) && out.length < count; a++) {
      const ayah = quran.ayah(s, a);
      ayah?.words.forEach((_, i) => out.push({ surah: s, ayah: a, word: i + 1 }));
    }
  }
  return out.slice(0, count);
}

export async function createMockTilawa(options: {
  onProgress?: (p: { stage: string; fraction: number; fromCache: boolean }) => void;
  signal?: AbortSignal;
  params?: MockParams;
}): Promise<EngineHost> {
  const params = options.params ?? mockParams();
  const steps = 6;
  for (let i = 1; i <= steps; i++) {
    options.signal?.throwIfAborted();
    await new Promise((r) => setTimeout(r, 160 / params.speed));
    options.onProgress?.({ stage: i < steps ? "model" : "warmup", fraction: i < steps ? i / (steps - 1) : 1, fromCache: false });
  }
  const words = await wordsFrom(params.surah, 600);
  const script = mockScript(words);

  return {
    startFreeRecitation: () => new MockSession(script),
    async listen(session) {
      const s = session as unknown as MockSession;
      if (params.mic === "denied") throw micError("denied");
      if (params.mic === "unavailable") throw micError("unavailable");
      if (params.mic === "insecure") throw micError("insecure-context");
      let stream: MediaStream | null = null;
      if (params.mic === "fake") {
        if (!navigator.mediaDevices?.getUserMedia) throw micError("insecure-context");
        try {
          stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (e) {
          const name = (e as { name?: string })?.name;
          throw micError(name === "NotAllowedError" || name === "SecurityError" ? "denied" : "unavailable");
        }
      }
      const perTick = Math.round((RATE * TICK_MS * params.speed) / 1000);
      const timer = setInterval(() => s.feed(perTick), TICK_MS);
      return {
        async stop() {
          clearInterval(timer);
          stream?.getTracks().forEach((t) => t.stop());
          await s.end();
        },
      };
    },
    dispose() {},
  };
}
