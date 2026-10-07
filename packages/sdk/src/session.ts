import { EVENT_SCHEMA_VERSION, createChecker, createTracker } from "@tilawa/core";
import type {
  CapabilityEvent,
  CheckerEvent,
  CheckerPolicy,
  DecodedPhoneme,
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
   * The tracker and checker see evidence in cells of this much audio on a fixed grid, so their events don't depend on
   * how the caller chunks audio or evidence. Default: one CTC frame (40 ms), so a cell is passed on as soon as it is decoded.
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
    const tracker = createTracker(index, { mode, phrases, acquireMs: ACQUIRE_MS });
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
    const cells = evidenceGrid(policy.evidenceMs ?? model.manifest.frameMs);
    const feed = (ready: PhonemeEvidence[]) => ready.flatMap(step);
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
          const e = await listener.push(samples, rate);
          if (e) cells.add(e);
          return feed(cells.ready(listener.decoded));
        }),
      end: () =>
        serial(async () => {
          const e = await listener.end();
          if (e) cells.add(e);
          return [...feed(cells.rest()), ...finish()];
        }),
      pushEvidence: (e) => {
        cells.add(e);
        return feed(cells.ready(e.posteriors && e.posteriors.startFrame + e.posteriors.nFrames));
      },
      endEvidence: () => [...feed(cells.rest()), ...finish()],
      mistakes: () => checker.mistakes(),
      evidence: () => recorded,
    };
  }

  return {
    startMemorization: (p, policy) => start("memorization", declare(p), policy),
    startFreeRecitation: (policy) => start("free", null, policy),
  };
}

/** Before its first lock the tracker searches once per second of evidence: shorter prefixes lock on the wrong ayah. */
const ACQUIRE_MS = 1000;

/**
 * Cuts an evidence stream into cells of `ms` audio by phoneme start, whatever chunks it arrives in. A cell is ready once
 * a phoneme from a later cell arrives or decoding has passed its end.
 */
function evidenceGrid(ms: number) {
  let held: DecodedPhoneme[] = [];
  let frameMs = 0;
  let inventory = "";
  let post: { k: number; startFrame: number; ids: number[]; probs: number[] } | null = null;
  const cellOf = (frame: number) => Math.floor((frame * frameMs) / ms);
  const cut = (n: number, until: number, decodedTo?: number): PhonemeEvidence => {
    const e: PhonemeEvidence = { inventory, frameMs, phonemes: held.slice(0, n), ...(decodedTo !== undefined && { decodedTo }) };
    held = held.slice(n);
    if (post) {
      const take = Math.max(0, Math.min(until, post.startFrame + post.ids.length / post.k) - post.startFrame);
      e.posteriors = { k: post.k, startFrame: post.startFrame, nFrames: take, ids: post.ids.splice(0, take * post.k), probs: post.probs.splice(0, take * post.k) };
      post.startFrame += take;
    }
    return e;
  };
  return {
    add(e: PhonemeEvidence) {
      frameMs = e.frameMs;
      inventory = e.inventory;
      held.push(...e.phonemes);
      if (e.posteriors) {
        post ??= { k: e.posteriors.k, startFrame: e.posteriors.startFrame, ids: [], probs: [] };
        post.ids.push(...Array.from(e.posteriors.ids));
        post.probs.push(...Array.from(e.posteriors.probs));
      }
    },
    /** Complete cells; `decoded`: the first frame whose phonemes may still be coming, when the source knows it. */
    ready(decoded?: number): PhonemeEvidence[] {
      const out: PhonemeEvidence[] = [];
      while (held.length) {
        const c = cellOf(held[0].start);
        const n = held.findIndex((p) => cellOf(p.start) > c);
        if (n < 0 && !(decoded !== undefined && cellOf(decoded) > c)) break;
        const end = Math.ceil(((c + 1) * ms) / frameMs);
        out.push(cut(n < 0 ? held.length : n, n < 0 ? end : held[n].start, end));
      }
      return out;
    },
    rest(): PhonemeEvidence[] {
      const out = this.ready();
      if (held.length || post?.ids.length) out.push(cut(held.length, Infinity));
      return out;
    },
  };
}
