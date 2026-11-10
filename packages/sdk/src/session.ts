import { EVENT_SCHEMA_VERSION, createChecker, createTracker, trackerInput } from "@tilawa/core";
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
import { Resampler } from "./audio.js";
import { evidenceGrid } from "./evidence-grid.js";
import { capabilities } from "./capabilities.js";
import { addInsertionEvidence, insertionCells } from "./seam-insertion.js";
import { spliceContextFreeGaps } from "./gap-decode.js";
import { addCycleEvidence, CycleObservations, cycleComparison, cycleReversal } from "./cycle-turbulence.js";
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
  /** Drains the listener and decides every open error, including warm-crop seam judgments when a teacher is installed. */
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
    if (model.cycle && (model.manifest.sampleRate !== 16000 || model.manifest.frameMs !== 40))
      throw new Error("cycle turbulence scorer requires 16 kHz audio and a 40 ms evidence clock");
    const cycleObservations = model.cycle ? new CycleObservations() : null;
    const cycleDecisions = new Map<string,boolean>();
    const checkerOptions = {
      phrases,
      policy: { countResolved: policy.countSelfCorrections ?? false, confirm: policy.confirm, minConfidence: policy.minConfidence },
      ...(passage && { passage }),
      ...(model.cycle && { sequenceEvidence(c: import("@tilawa/core").SequenceEvidenceCandidate) {
        const key=JSON.stringify(c), cached=cycleDecisions.get(key);
        if (cached !== undefined) return cached;
        const frames=cycleObservations!.frames;
        if (c.fromFrame<0 || c.toFrame*4>frames.length) return true;
        const comparison=cycleComparison(model.cycle!,frames.slice(c.fromFrame*4,c.toFrame*4),c);
        const permitted=!(comparison && cycleReversal(comparison));
        cycleDecisions.set(key,permitted);
        return permitted;
      } }),
    };
    const checker = createChecker(index, checkerOptions);
    const listener = createListener(model, { posteriors: true, ...options.listener });
    const wordIndex = new Map(index.words.map((w, i) => [`${w.surah}:${w.ayah}:${w.word}`, i]));
    const locks: { wi: number; timeMs: number }[] = [];
    const noteLocks = (events: EngineEvent[]) => {
      for (const ev of events) {
        if (ev.type !== "position" || ev.state !== "locked" || !ev.expected) continue;
        const wi = wordIndex.get(`${ev.expected.surah}:${ev.expected.ayah}:${ev.expected.word}`);
        if (wi !== undefined && locks.at(-1)?.wi !== wi) locks.push({ wi, timeMs: ev.timeMs });
      }
    };
    const rate = model.manifest.sampleRate;
    const pcmParts: Float32Array[] = [];
    let pcmResampler: Resampler | null = null;
    const keepAudio = (samples: Float32Array, sampleRate: number) => {
      if (!options.recordEvidence && !model.insertionTeacher && !model.cycle) return;
      let pcm = samples;
      if (sampleRate !== rate) {
        if (!pcmResampler || pcmResampler.from !== sampleRate) pcmResampler = new Resampler(sampleRate, rate);
        pcm = pcmResampler.push(samples);
      }
      if (pcm.length) pcmParts.push(Float32Array.from(pcm));
      cycleObservations?.push(pcm);
    };
    const subscribers = new Set<(e: EngineEvent) => void>();
    let queue = Promise.resolve();
    const emit = (events: EngineEvent[]) => {
      for (const e of events) for (const s of subscribers) s(e);
      return events;
    };
    const recorded: PhonemeEvidence[] = [];
    const auxiliaryMistakes: CheckerEvent[] = [];
    const cells = evidenceGrid(policy.evidenceMs ?? model.manifest.frameMs);
    const feed = (ready: PhonemeEvidence[]) => ready.flatMap(step);
    const step = (e: PhonemeEvidence) => {
      if (options.recordEvidence || model.insertionTeacher || model.cycle) recorded.push(e);
      // Posteriors feed the checker's local-window check only; the tracker runs on the decode.
      const out = tracker.push(trackerInput(e));
      noteLocks(out);
      return emit([...out, ...checker.push(e, out)]);
    };
    const finish = () => {
      const out = tracker.end();
      noteLocks(out);
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
      pushAudio: (samples, sampleRate) =>
        serial(async () => {
          keepAudio(samples, sampleRate);
          const e = await listener.push(samples, sampleRate);
          if (e) cells.add(e);
          return feed(cells.ready(listener.decoded));
        }),
      end: () =>
        serial(async () => {
          if (pcmResampler) {
            const tail = pcmResampler.flush();
            if (tail.length) pcmParts.push(Float32Array.from(tail));
            cycleObservations?.push(tail);
          }
          const e = await listener.end();
          if (e) cells.add(e);
          const events = [...feed(cells.rest()), ...finish()];
          if ((options.recordEvidence || model.insertionTeacher || model.cycle) && pcmParts.length) {
            const n = pcmParts.reduce((s, p) => s + p.length, 0);
            const audio = new Float32Array(n);
            let o = 0;
            for (const part of pcmParts) {
              audio.set(part, o);
              o += part.length;
            }
            const spliced = options.recordEvidence && locks.length > 1
              ? await spliceContextFreeGaps(model, audio, rate, recorded, locks, index) : recorded;
            const insertions = await addInsertionEvidence(model, audio, spliced, index, phrases);
            const enriched = addCycleEvidence(model.cycle, audio, insertions, index, phrases, checkerOptions, mode, undefined, policy.evidenceMs);
            recorded.splice(0, recorded.length, ...enriched);
            if (model.insertionTeacher) {
              // Auxiliary replay cannot replace primary events or change the live tracker.
              // Only newly certified seam additions are published at end().
              const reviewTracker = createTracker(index, { mode, phrases, acquireMs: ACQUIRE_MS });
              const review = createChecker(index, checkerOptions);
              for (const e of insertionCells(enriched)) review.push(e, reviewTracker.push(trackerInput(e)));
              review.end(reviewTracker.end());
              const known = new Set(checker.mistakes().filter(e => e.errorClass === "added_word")
                .map(e => JSON.stringify(e.location)));
              const certificates = enriched.flatMap(e => e.phonemes.flatMap(p => p.seamInsertions ?? []));
              for (const e of review.mistakes()) {
                if (e.errorClass !== "added_word" || known.has(JSON.stringify(e.location)) ||
                  !certificates.some(c => c.marginal > 0.5 && c.observed === e.observed &&
                    c.word.surah === e.location.surah && c.word.ayah === e.location.ayah && c.word.word === e.location.word)) continue;
                auxiliaryMistakes.push(e);
                known.add(JSON.stringify(e.location));
              }
              events.push(...emit(auxiliaryMistakes));
            }
          }
          return events;
        }),
      pushEvidence: (e) => {
        cells.add(e);
        return feed(cells.ready(e.posteriors && e.posteriors.startFrame + e.posteriors.nFrames));
      },
      endEvidence: () => [...feed(cells.rest()), ...finish()],
      mistakes: () => [...checker.mistakes(), ...auxiliaryMistakes],
      evidence: () => options.recordEvidence ? recorded : [],
    };
  }

  return {
    startMemorization: (p, policy) => start("memorization", declare(p), policy),
    startFreeRecitation: (policy) => start("free", null, policy),
  };
}

/** Before its first lock the tracker searches once per second of evidence: shorter prefixes lock on the wrong ayah. */
const ACQUIRE_MS = 1000;
