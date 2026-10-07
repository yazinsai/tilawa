import { EVENT_SCHEMA_VERSION, createChecker, createTracker } from "@tilawa/core";
import type {
  CapabilityEvent,
  CheckerEvent,
  CheckerPolicy,
  EngineEvent,
  PhonemeEvidence,
  PhraseAsset,
  ReferenceIndex,
  WordRef,
} from "@tilawa/core";
import { capabilities } from "./capabilities.js";
import { createListener } from "./listener.js";
import type { ListenerOptions } from "./listener.js";
import type { LoadedModel } from "./model.js";

export interface SessionPolicy {
  /** Decision 5: a self-correction (or a recovered skip) goes `suspected` then `resolved`; count it as a mistake? Default false. */
  countSelfCorrections?: boolean;
  /** When a suspected wrong or added word is confirmed. Default 2.5 s after it. */
  confirm?: CheckerPolicy["confirm"];
  /** Below this mean decoder confidence an error ends `uncertain`. Default 0.6. */
  minConfidence?: number;
  /**
   * Batch listener output into evidence chunks of at least this much audio before the tracker and checker see it.
   * Default 1000, the cadence every scoreboard number is measured at: at 320 ms the free-recitation checker raises
   * more false warnings on the help clips (precision 81.8% → 73.0%); the declared arm is unchanged.
   */
  evidenceMs?: number;
}

export interface MemorizationPassage {
  surah: number;
  fromAyah: number;
  toAyah: number;
}

export interface EngineOptions {
  reference: ReferenceIndex;
  phrases?: PhraseAsset;
  model: LoadedModel;
  listener?: ListenerOptions;
  /** Keep every evidence chunk the listener produced, for `Session.evidence()` (benchmarks, bug reports). */
  recordEvidence?: boolean;
}

export interface Session {
  readonly mode: "memorization" | "free";
  /** The declared passage's first and last words; null in free recitation. */
  readonly passage: { from: WordRef; to: WordRef } | null;
  readonly capabilities: CapabilityEvent;
  /** Called with every event, starting with `capabilities`. Returns an unsubscribe function. */
  subscribe(listener: (event: EngineEvent) => void): () => void;
  /** Microphone or file audio at any sample rate. */
  pushAudio(samples: Float32Array, sampleRate: number): Promise<EngineEvent[]>;
  /** Drains the listener and decides every open error. */
  end(): Promise<EngineEvent[]>;
  /** Evidence from another listener, bypassing the model; use `endEvidence` to finish. */
  pushEvidence(evidence: PhonemeEvidence): EngineEvent[];
  endEvidence(): EngineEvent[];
  /** Confirmed errors, plus resolved ones when the policy counts self-corrections. */
  mistakes(): CheckerEvent[];
  /** The evidence so far, when the engine records it; otherwise empty. */
  evidence(): PhonemeEvidence[];
}

export interface Engine {
  startMemorization(passage: MemorizationPassage, policy?: SessionPolicy): Session;
  startFreeRecitation(policy?: SessionPolicy): Session;
}

export function createEngine(options: EngineOptions): Engine {
  const { reference: index, phrases, model } = options;
  if (model.manifest.inventory !== index.inventoryId)
    throw new Error(`Model inventory ${model.manifest.inventory} does not match reference ${index.inventoryId}`);
  const caps: CapabilityEvent = {
    type: "capabilities",
    v: EVENT_SCHEMA_VERSION,
    modelVersion: model.manifest.id,
    referenceVersion: index.version,
    ...capabilities(model.supported),
  };

  function declare(p: MemorizationPassage) {
    const words = (ayah: number) => index.words.filter((w) => w.surah === p.surah && w.ayah === ayah && w.end > w.start);
    const first = words(p.fromAyah)[0];
    const last = words(p.toAyah).at(-1);
    if (!first || !last || p.toAyah < p.fromAyah) throw new RangeError(`No passage ${p.surah}:${p.fromAyah}-${p.toAyah}`);
    const ref = (w: WordRef): WordRef => ({ surah: w.surah, ayah: w.ayah, word: w.word });
    return { from: ref(first), to: ref(last) };
  }

  function start(mode: Session["mode"], passage: Session["passage"], policy: SessionPolicy = {}): Session {
    const tracker = createTracker(index, { mode, phrases });
    const checker = createChecker(index, {
      phrases,
      policy: { countResolved: policy.countSelfCorrections ?? false, confirm: policy.confirm, minConfidence: policy.minConfidence },
      ...(passage && { passage }),
    });
    const listener = createListener(model, options.listener);
    const subscribers = new Set<(e: EngineEvent) => void>();
    let queue = Promise.resolve();
    const emit = (events: EngineEvent[]) => {
      for (const e of events) for (const s of subscribers) s(e);
      return events;
    };
    const recorded: PhonemeEvidence[] = [];
    const evidenceMs = policy.evidenceMs ?? 1000;
    let held: PhonemeEvidence[] = [];
    let heldFrom = 0;
    let heldTo = 0;
    /** Listener chunks end at the start of the open token run; the batch spans up to the last one's end. */
    const batch = (e: PhonemeEvidence | null, final: boolean) => {
      if (e) {
        if (!held.length) heldFrom = heldTo;
        held.push(e);
        heldTo = Math.max(heldTo, e.posteriors ? e.posteriors.startFrame + e.posteriors.nFrames : (e.phonemes.at(-1)?.end ?? heldTo - 1) + 1);
      }
      if (!held.length || (!final && (heldTo - heldFrom) * model.manifest.frameMs < evidenceMs)) return [];
      const merged = merge(held);
      held = [];
      return step(merged);
    };
    const step = (e: PhonemeEvidence) => {
      if (options.recordEvidence) recorded.push(e);
      const out = tracker.push(e);
      return emit([...out, ...checker.push(e, out)]);
    };
    const finish = () => {
      const out = tracker.end();
      return emit([...out, ...checker.end(out)]);
    };
    /** Audio calls run one at a time even when the caller doesn't await them (an AudioWorklet callback won't). */
    const serial = <T>(f: () => Promise<T>) => {
      const run = queue.then(f);
      queue = run.then(() => undefined, () => undefined);
      return run;
    };
    return {
      mode,
      passage,
      capabilities: caps,
      subscribe(fn) {
        subscribers.add(fn);
        fn(caps);
        return () => subscribers.delete(fn);
      },
      pushAudio: (samples, rate) =>
        serial(async () => {
          return batch(await listener.push(samples, rate), false);
        }),
      end: () =>
        serial(async () => {
          return [...batch(await listener.end(), true), ...finish()];
        }),
      pushEvidence: step,
      endEvidence: finish,
      mistakes: () => checker.mistakes(),
      evidence: () => recorded,
    };
  }

  return {
    startMemorization: (p, policy) => start("memorization", declare(p), policy),
    startFreeRecitation: (policy) => start("free", null, policy),
  };
}

function merge(chunks: PhonemeEvidence[]): PhonemeEvidence {
  if (chunks.length === 1) return chunks[0];
  const e: PhonemeEvidence = { inventory: chunks[0].inventory, frameMs: chunks[0].frameMs, phonemes: chunks.flatMap((c) => c.phonemes) };
  const ps = chunks.map((c) => c.posteriors);
  if (ps.every((p) => p)) {
    const k = ps[0]!.k;
    e.posteriors = {
      k,
      startFrame: ps[0]!.startFrame,
      nFrames: ps.reduce((n, p) => n + p!.nFrames, 0),
      ids: ps.flatMap((p) => Array.from(p!.ids)),
      probs: ps.flatMap((p) => Array.from(p!.probs)),
    };
  }
  return e;
}
