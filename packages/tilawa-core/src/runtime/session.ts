import { EVENT_SCHEMA_VERSION, createChecker, createTracker, trackerInput } from "../engine/index.js";
import type {
  CapabilityEvent,
  CheckerRefinements,
  CheckerEvent,
  CheckerPolicy,
  DecodedPhoneme,
  EngineEvent,
  PhonemeEvidence,
  PhraseAsset,
  ReferenceIndex,
  TrackerRefinements,
  WordRef,
} from "../engine/index.js";
import { Resampler } from "./audio.js";
import { capabilities } from "./capabilities.js";
import { addInsertionEvidence, insertionCells } from "./seam-insertion.js";
import { spliceContextFreeGaps } from "./gap-decode.js";
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
  /**
   * Refinements of the 1.0.0-rc.1 engine's rules ({@link Refinements}): each on unless set false, but `markWhenSure`,
   * off unless set true.
   */
  refinements?: Refinements;
}

/**
 * Refinements of the 1.0.0-rc.1 engine's rules: how it follows a reciter and what it marks. Every field defaults to
 * `true` but one: `markWhenSure` defaults to `false`, since it still gets cases wrong that it cannot yet tell from the
 * ones it is for (its doc says which); set it `true` to turn it on. A field set `false` keeps the rc.1 rule it
 * replaces, so each can be measured alone, and with every field `false` a session behaves exactly as rc.1's. The
 * checker's fields are {@link CheckerRefinements}'s, the tracker's {@link TrackerRefinements}'s.
 */
export interface Refinements extends CheckerRefinements, TrackerRefinements {
  /**
   * Display smoothing: words completed by one model answer appear in the rhythm they were said, not at once. It changes
   * only when a position or checker event is delivered, never what the tracker follows or the checker marks, and it
   * never shows a word earlier than without it. An app that draws the words in its own layer (a native rebuild) may
   * carry it there instead, and set this `false`. A position that moves to a later word, arriving in the same
   * `pushAudio` as the one before it (or while that one is held), is due that one's due time plus the evidence time
   * between them, on the audio clock (milliseconds of audio pushed), and is shown at the last push that ends by then,
   * taking the next push to be as long as the longest of the latest eight. So no word is shown later than the time
   * between it and the word before it, as said, requires, unless the pushes grow (a word due inside a push longer than
   * the latest eight shows when it ends); an ayah's first word is never held (what is held before it is shown with it);
   * and pushes longer than that time hold nothing: what pace separates depends on the push size (pushes of P ms
   * separate words said P ms or more apart). Events are never reordered: checker events behind a held position wait
   * with it, and `mistakes()` lists a confirmation once its event is shown. The checker itself sees everything at once.
   * `end`, and evidence pushed with `pushEvidence` (which has no audio clock), show everything held at once; held
   * events wait for the next push or `end`, so an app that stops pushing audio should call `end`. Risk: a word
   * completed in the same push as the one before it shows up to the time between them later: at most the audio one push
   * decodes (one model window, 320 ms, at pushes up to that long).
   */
  pace?: boolean;
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

  /** refinements.pace: each ayah's first spoken word, as `surah:ayah:word`. */
  let ayahFirst: Set<string> | null = null;
  const opensAyah = (w: WordRef) => {
    if (!ayahFirst) {
      ayahFirst = new Set();
      let ayah = "";
      for (const x of index.words) {
        if (x.end <= x.start || `${x.surah}:${x.ayah}` === ayah) continue;
        ayah = `${x.surah}:${x.ayah}`;
        ayahFirst.add(`${ayah}:${x.word}`);
      }
    }
    return ayahFirst.has(`${w.surah}:${w.ayah}:${w.word}`);
  };

  function start(mode: Session["mode"], passage: Session["passage"], policy: SessionPolicy = {}): Session {
    const trackerOptions = { mode, phrases, acquireMs: ACQUIRE_MS, refinements: policy.refinements, ...(passage && { passage }) };
    const tracker = createTracker(index, trackerOptions);
    const checkerOptions = {
      phrases,
      policy: {
        countResolved: policy.countSelfCorrections ?? false, confirm: policy.confirm, minConfidence: policy.minConfidence, refinements: policy.refinements,
      },
      ...(passage && { passage }),
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
      if (!options.recordEvidence && !model.insertionTeacher) return;
      let pcm = samples;
      if (sampleRate !== rate) {
        if (!pcmResampler || pcmResampler.from !== sampleRate) pcmResampler = new Resampler(sampleRate, rate);
        pcm = pcmResampler.push(samples);
      }
      if (pcm.length) pcmParts.push(Float32Array.from(pcm));
    };
    const subscribers = new Set<(e: EngineEvent) => void>();
    let queue = Promise.resolve();
    const send = (events: EngineEvent[]) => {
      for (const e of events) for (const s of subscribers) s(e);
      return events;
    };
    const pacer = policy.refinements?.pace === false ? null : createPacer(opensAyah);
    /** True while a `pushAudio` feeds its cells: the only time `pace` holds a position. */
    let pacing = false;
    /** The audio clock: milliseconds of audio pushed so far. */
    let clockMs = 0;
    const emit = (events: EngineEvent[]) => send(pacing && pacer ? pacer.admit(events) : events);
    /** A push of `ms` of audio: the positions held earlier that are due now. */
    const tick = (ms: number) => {
      clockMs += ms;
      return pacer ? send(pacer.tick(clockMs, ms)) : [];
    };
    /** Everything held, now. */
    const flush = () => (pacer ? send(pacer.flush()) : []);
    const recorded: PhonemeEvidence[] = [];
    const auxiliaryMistakes: CheckerEvent[] = [];
    const cells = evidenceGrid(policy.evidenceMs ?? model.manifest.frameMs);
    const feed = (ready: PhonemeEvidence[]) => ready.flatMap(step);
    /**
     * Decoding has reached frame `decoded`: the cells ready, then (refinements.showSaid) what the silence decoded since
     * the last phoneme settles, since no cell is cut while nothing is heard. Off, the silence returns nothing, and
     * nothing is sent.
     */
    const decodedTo = (decoded: number | undefined, ready: PhonemeEvidence[]) => {
      const events = feed(ready);
      const quiet = cells.quietTo(decoded);
      const settled = quiet === undefined ? [] : tracker.silence(quiet);
      return settled.length ? [...events, ...emit(settled)] : events;
    };
    const step = (e: PhonemeEvidence) => {
      if (options.recordEvidence || model.insertionTeacher) recorded.push(e);
      // Posteriors feed the checker's local-window check only; the tracker runs on the decode.
      const out = tracker.push(trackerInput(e));
      noteLocks(out);
      // The checker reads where the reciter is; an app is sent what to show (refinements.showSaid).
      const shown = tracker.shown();
      return emit([...shown, ...checker.push(e, out)]);
    };
    const finish = () => {
      const out = tracker.end();
      noteLocks(out);
      const shown = tracker.shown();
      return emit([...shown, ...checker.end(out)]);
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
          const released = tick((samples.length / sampleRate) * 1000);
          const e = await listener.push(samples, sampleRate);
          if (e) cells.add(e);
          pacing = true;
          try {
            return [...released, ...decodedTo(listener.decoded, cells.ready(listener.decoded))];
          } finally {
            pacing = false;
          }
        }),
      end: () =>
        serial(async () => {
          if (pcmResampler) {
            const tail = pcmResampler.flush();
            if (tail.length) pcmParts.push(Float32Array.from(tail));
          }
          const released = flush();
          const e = await listener.end();
          if (e) cells.add(e);
          const events = [...released, ...feed(cells.rest()), ...finish()];
          if ((options.recordEvidence || model.insertionTeacher) && pcmParts.length) {
            const n = pcmParts.reduce((s, p) => s + p.length, 0);
            const audio = new Float32Array(n);
            let o = 0;
            for (const part of pcmParts) {
              audio.set(part, o);
              o += part.length;
            }
            const spliced = options.recordEvidence && locks.length > 1
              ? await spliceContextFreeGaps(model, audio, rate, recorded, locks, index) : recorded;
            const enriched = await addInsertionEvidence(model, audio, spliced, index, phrases);
            recorded.splice(0, recorded.length, ...enriched);
            if (model.insertionTeacher) {
              // Auxiliary replay cannot replace primary events or change the live tracker.
              // Only newly certified seam additions are published at end().
              const reviewTracker = createTracker(index, trackerOptions);
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
        const released = flush();
        cells.add(e);
        const decoded = e.posteriors && e.posteriors.startFrame + e.posteriors.nFrames;
        return [...released, ...decodedTo(decoded ?? e.decodedTo, cells.ready(decoded))];
      },
      endEvidence: () => [...flush(), ...feed(cells.rest()), ...finish()],
      // refinements.pace: a confirmation still held is not listed yet, so the pull view agrees with the events shown.
      mistakes: () => [...(pacer ? checker.mistakes().filter((e) => !pacer.holds(e)) : checker.mistakes()), ...auxiliaryMistakes],
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

const wordOrder = (a: WordRef, b: WordRef) => a.surah - b.surah || a.ayah - b.ayah || a.word - b.word;

/** refinements.pace: how many of the latest pushes the pacer takes the next push's length from (the longest). */
const PACE_PUSHES = 8;

/**
 * refinements.pace: shows the words of a recitation in the rhythm they were said, on the audio clock (milliseconds of audio
 * pushed). The model answers for a window of audio at a time, so the push that receives an answer can complete two
 * words said a few hundred milliseconds apart, and would show them at once. Each locked position that moves to a later
 * word is given a due time on the audio clock:
 *
 * - when it arrives in the same push as the position before it, or while that one is still held: that position's due
 *   time plus the evidence time between the two (how far apart they were said);
 * - otherwise (it is the first position of its push, behind the last, or an ayah's first word): the push it arrives in.
 *
 * A position is shown at the last push that ends by its due time, taking the next push to be as long as the longest of
 * the last `PACE_PUSHES` pushes: it is never shown before it arrives, and never later than due unless the next push is
 * longer than all of those (pushes that grow: a word due inside the longer push shows when it ends). With pushes longer
 * than the gap between two words, nothing is held, and short pushes among long ones hold no more than the long ones
 * do. An ayah's first word is never held: whatever is held before it is shown with it. Events are never reordered:
 * anything that arrives behind a held position (checker events included) waits with it. Held events wait for a push
 * or the take's end: an app that stops pushing audio without calling `end()` does not see them.
 */
export function createPacer(opensAyah: (word: WordRef) => boolean) {
  const queue: { e: EngineEvent; due: number }[] = [];
  /** The events in `queue`, to tell a held event from a shown one. */
  const held = new Set<EngineEvent>();
  /** The last locked position given a due time: its word, evidence time and due time (ms), and the push it came in. */
  let last: { word: WordRef; timeMs: number; due: number; push: number } | null = null;
  let push = 0;
  /** The lengths of the latest pushes (ms), at most `PACE_PUSHES`. */
  const recent: number[] = [];
  /** The audio clock at the end of this push, and at the end of the next one (as long as the longest recent push). */
  let now = 0;
  let next = 0;
  const due = (from: number) => {
    const out: EngineEvent[] = [];
    while (queue.length && queue[0].due < from) {
      const { e } = queue.shift()!;
      held.delete(e);
      out.push(e);
    }
    return out;
  };
  return {
    /** A push of `pushMs` of audio that brings the clock to `clockMs`: the held events due before the next push ends. */
    tick(clockMs: number, pushMs: number): EngineEvent[] {
      push++;
      now = clockMs;
      recent.push(pushMs);
      if (recent.length > PACE_PUSHES) recent.shift();
      next = clockMs + Math.max(...recent);
      return due(next);
    },
    /** This push's events, in order: the ones to show now. */
    admit(events: EngineEvent[]): EngineEvent[] {
      const out: EngineEvent[] = [];
      for (const e of events) {
        let at = now;
        const word = e.type === "position" && e.state === "locked" ? e.observed[0]?.word : undefined;
        if (word && e.type === "position") {
          const forward = !last || wordOrder(word, last.word) > 0;
          if (forward && opensAyah(word)) out.push(...due(Infinity));
          else if (forward && last && (last.push === push || queue.length)) at = Math.max(now, last.due + Math.max(0, e.timeMs - last.timeMs));
          last = { word, timeMs: e.timeMs, due: at, push };
        }
        if (!queue.length && at < next) out.push(e);
        else {
          queue.push({ e, due: at });
          held.add(e);
        }
      }
      return out;
    },
    /** Everything held, now (the take's end, or evidence pushed without audio). */
    flush: (): EngineEvent[] => due(Infinity),
    /** Whether event `e` is held: admitted, and not yet shown. */
    holds: (e: EngineEvent): boolean => held.has(e),
  };
}


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
    /**
     * The frame decoding has reached with nothing heard since the last cell cut, given `decoded` (as `ready`'s): up to
     * the first phoneme still held, if any.
     */
    quietTo(decoded?: number): number | undefined {
      return decoded === undefined ? undefined : held.length ? Math.min(decoded, held[0].start) : decoded;
    },
    rest(): PhonemeEvidence[] {
      const out = this.ready();
      if (held.length || post?.ids.length) out.push(cut(held.length, Infinity));
      return out;
    },
  };
}
