/**
 * `ZipformerSession` — the default recognition path of `@tilawa/core`.
 *
 * Streaming phoneme pipeline: 16 kHz PCM -> Kaldi fbank (80 mel) -> streaming
 * Zipformer2-CTC (251 tajweed-phoneme tokens) -> greedy CTC -> whole-Quran
 * n-gram search + per-surah online DP tracker -> per-word verdicts -> the
 * SDK's verse events (`emission.ts`).
 *
 * Runtime-agnostic: ONNX arrives through the same injection seam as
 * `SessionRunner` — either a whole `ort`-like namespace plus model bytes, or an
 * already-created session plus that runtime's `Tensor` constructor (the shape
 * React Native needs, where `InferenceSession.create` takes a file path).
 */
import { QuranDB } from "../quran-db.js";
import type { QuranVerse, SurroundingVerse, WorkerOutbound } from "../types.js";
import { SURROUNDING_CONTEXT } from "../types.js";
import {
  accumulateSnapshot,
  ayahConfidence,
  ayahKey,
  ayahMeetsGate,
  bridgeGapAyahs,
  buildFinalSequence,
  FALLBACK_MAX_DISTANCE,
  GAP_MAX_WORDS,
  mergeTallies,
  MIN_WORD_FRACTION,
  newlyEligibleAyahs,
  shouldRunFallback,
  snapshotTallies,
  wordProgressFromCursor,
  type AyahTally,
  type BridgedAyahTally,
  type EmissionVerdict,
} from "./emission.js";
import { DEFAULT_CONFIG, SAMPLE_RATE, type EngineConfig } from "./config.js";
import { GreedyCtcDecoder } from "./ctcDecoder.js";
import { KaldiFbank } from "./fbank.js";
import { QuranCorpus } from "./corpus.js";
import { QuranIndex, stripPreambles } from "./search.js";
import { RecitationEngine } from "./engine.js";
import type { Tracker } from "./tracker.js";
import { BLANK_ID, TOKENS } from "./tokens.js";
import { costTable } from "./phonemeCost.js";
import { normalizedDistance } from "./alignment.js";
import type { EngineEvent, ExpectedPassage, FallbackHit, WordVerdict } from "./types.js";
import {
  ZipformerRunner,
  type OrtLike,
  type OrtSessionLike,
  type ZipformerIo,
} from "./zipformerRunner.js";
import DEFAULT_IO from "./zipformer-io.json" with { type: "json" };

import { CorrectionController, type CorrectionAction, type RecitationMode } from "./correction.js";
import { FramePosteriors } from "./posteriors.js";
import { A0W_SLIP_HEAD, EncoderFrames, slipThreshold, type SlipHead, type SlipSensitivity } from "./slipHead.js";

const TAIL_SECONDS = 2.0;
/** Mean heard ratio over an unmatched ayah's words at or above which the gap
 * counts as heard-but-unfollowed (`unclear_ayah`) rather than skipped. */
export const AYAH_HEARD_FRACTION = 0.5;
const SKIP_PERSIST_FRAMES = 12;

/** I/O manifest of the shipped `zipformer_interp_gentle_a05.int8.onnx`. */
export const DEFAULT_ZIPFORMER_IO = DEFAULT_IO as ZipformerIo;

/** Model bytes, or a loader that produces them (fetch, fs, asset bundle). */
export type ModelSource =
  | Uint8Array
  | ArrayBuffer
  | (() => Uint8Array | ArrayBuffer | Promise<Uint8Array | ArrayBuffer>);

/** Parsed `zipformer_quran.json`, or a loader for it. */
export type CorpusSource = unknown | (() => unknown | Promise<unknown>);

/** Display text: a ready `QuranDB`, raw `quran.json` rows, or a loader. */
export type QuranSource =
  | QuranDB
  | unknown[]
  | (() => unknown[] | QuranDB | Promise<unknown[] | QuranDB>);

interface EncodedAyah {
  surah: number;
  ayah: number;
  ids: Uint8Array;
}

async function resolveSource<T>(src: T | (() => T | Promise<T>)): Promise<T> {
  return typeof src === "function" ? (src as () => T | Promise<T>)() : src;
}

/**
 * Build the display-text `QuranDB` from raw `quran.json` rows. The Zipformer
 * path never reads the text-CTC fields, so they default to empty.
 */
export function displayQuranFromRaw(raw: unknown): QuranDB {
  if (!Array.isArray(raw)) throw new Error("quran.json must be an array");
  return new QuranDB(
    raw.map((row) => {
      const v = row as QuranVerse;
      return {
        ...v,
        phonemes: v.phonemes ?? "",
        phonemes_joined: v.phonemes_joined ?? "",
        phoneme_words: v.phoneme_words ?? [],
      };
    }),
  );
}

async function resolveQuranDb(src: QuranSource | undefined): Promise<QuranDB> {
  if (!src) return displayQuranFromRaw([]);
  const resolved = await resolveSource(src as QuranDB | unknown[]);
  return resolved instanceof QuranDB ? resolved : displayQuranFromRaw(resolved);
}

function surroundingVerses(db: QuranDB, surah: number, ayah: number): SurroundingVerse[] {
  return db.getSurah(surah)
    .filter((v) => Math.abs(v.ayah - ayah) <= SURROUNDING_CONTEXT)
    .map((v) => ({
      surah: v.surah,
      ayah: v.ayah,
      text: v.text_uthmani,
      is_current: v.ayah === ayah,
    }));
}

export interface ZipformerSessionOptions {
  /**
   * The ONNX runtime namespace (`onnxruntime-web`, `onnxruntime-node`,
   * `onnxruntime-react-native`). Needed with {@link model}; skip it when you
   * pass {@link session} + {@link Tensor} yourself.
   */
  ort?: unknown;
  /** Model bytes or a loader. Required unless {@link session} is given. */
  model?: ModelSource;
  /**
   * An already-created inference session. Use this on React Native, where
   * `InferenceSession.create()` takes a model *path*, not bytes.
   */
  session?: OrtSessionLike;
  /** That runtime's `Tensor` constructor. Required with {@link session}. */
  Tensor?: OrtLike["Tensor"];
  /**
   * Execution providers for `InferenceSession.create`. When omitted and you
   * pass {@link ort} + {@link model}: `["wasm"]` under onnxruntime-web,
   * `["cpu"]` under onnxruntime-node.
   */
  executionProviders?: string[];
  /** I/O manifest. Defaults to the bundled {@link DEFAULT_ZIPFORMER_IO}. */
  io?: ZipformerIo | (() => ZipformerIo | Promise<ZipformerIo>);
  /**
   * Phoneme corpus — parsed `zipformer_quran.json` or a loader for it. Not
   * bundled (5.5 MB, NPL-1.2 derivative); see the SDK README for where to get
   * it.
   */
  corpus: CorpusSource;
  /** Display text for `verse_match` events. Optional; omit for an empty DB. */
  quran?: QuranSource;
  /** Engine knobs, merged onto {@link DEFAULT_CONFIG}. */
  config?: Partial<EngineConfig>;
  /** Silence appended by `stop()` to flush the CTC tail. Default 2.0 s. */
  tailSeconds?: number;
  /** Fraction of an ayah's words that must land to emit it. Default 0.5. */
  minWordFraction?: number;
  /** Never relocate off the surah we locked onto. Default false. */
  stayOnSurah?: boolean;
  /** Whole-ayah search over the transcript when nothing was emitted. Default true. */
  enableFallback?: boolean;
  /** Max normalized distance for that fallback to count. Default 0.5. */
  fallbackMaxDistance?: number;
  /** Let {@link verses} bridge a single short skipped ayah. Default false. */
  allowGaps?: boolean;
  /** Longest ayah (in words) `allowGaps` may bridge. Default 3. */
  gapMaxWords?: number;
  /** Streaming events, in the same order `feed()`/`stop()` return them. */
  onEvent?: (msg: WorkerOutbound) => void;
  /** Emit `debug` messages for engine events. Default false. */
  debug?: boolean;
  /**
   * Slip head over encoder frames. Off by default. `true` or `"strict"` uses
   * the no-extra-false-flag cutoff; `"high"` uses the sensitivity point.
   * If the model does not return encoder frames, the head stays off.
   */
  slipHead?: boolean | SlipSensitivity;
}

export class ZipformerSession {
  private readonly corpus: QuranCorpus;
  private readonly index: QuranIndex;
  private readonly table = costTable();
  private readonly ayahIds: EncodedAyah[] = [];
  private readonly quranDb: QuranDB;
  private readonly runner: ZipformerRunner;
  private readonly cfg: EngineConfig;
  private readonly tailSeconds: number;
  private readonly minWordFraction: number;
  private readonly stayOnSurah: boolean;
  private readonly enableFallback: boolean;
  private readonly fallbackMaxDistance: number;
  private readonly allowGaps: boolean;
  private readonly gapMaxWords: number;
  private readonly onEvent: ((msg: WorkerOutbound) => void) | null;
  private fbank = new KaldiFbank();
  private decoder = new GreedyCtcDecoder(TOKENS, BLANK_ID);
  private readonly posteriors: FramePosteriors;
  private readonly encoderFrames = new EncoderFrames();
  private readonly slipWeights: SlipHead = A0W_SLIP_HEAD;
  /** Requested sensitivity. False until the caller turns the head on. */
  private slipMode: false | SlipSensitivity = false;
  /** Flips off, silently, when a producing chunk has no encoder frames. */
  private slipLive = true;
  private engine: RecitationEngine;
  private accumulated = new Map<string, AyahTally>();
  private emitted = new Set<string>();
  private transcriptParts: string[] = [];
  private lastCursor: { surah: number; ayah: number; word: number } | null = null;
  /** Last `verse_match` emitted by the current tracker lock. Cleared on
   * locate / relocate / lost so an ayah gap across a jump never flags. */
  private lastMatch: { surah: number; ayah: number } | null = null;
  private ayahIssuesRaised = new Set<string>();
  readonly correction = new CorrectionController();
  private practiceEngine: RecitationEngine | null = null;
  private expected: ExpectedPassage | null = null;
  private skipCandidate: { key: string; frame: number } | null = null;
  private stopping = false;
  lastFallback: FallbackHit | null = null;
  debugEnabled = false;

  private constructor(
    runner: ZipformerRunner,
    corpusJson: unknown,
    quranDb: QuranDB,
    opts: ZipformerSessionOptions,
  ) {
    this.runner = runner;
    this.quranDb = quranDb;
    this.cfg = { ...DEFAULT_CONFIG, ...opts.config };
    this.tailSeconds = opts.tailSeconds ?? TAIL_SECONDS;
    this.minWordFraction = opts.minWordFraction ?? MIN_WORD_FRACTION;
    this.stayOnSurah = opts.stayOnSurah ?? false;
    this.enableFallback = opts.enableFallback ?? true;
    this.fallbackMaxDistance = opts.fallbackMaxDistance ?? FALLBACK_MAX_DISTANCE;
    this.allowGaps = opts.allowGaps ?? false;
    this.gapMaxWords = opts.gapMaxWords ?? GAP_MAX_WORDS;
    this.onEvent = opts.onEvent ?? null;
    this.debugEnabled = opts.debug ?? false;
    this.slipMode = opts.slipHead === true || opts.slipHead === "strict" ? "strict"
      : opts.slipHead === "high" ? "high" : false;
    this.posteriors = new FramePosteriors(runner.io.vocabSize);
    this.corpus = new QuranCorpus(corpusJson);
    this.index = new QuranIndex(this.corpus, this.cfg);
    for (const s of this.corpus.surahs) {
      for (let a = 1; a <= s.ayahCount; a++) {
        const first = this.corpus.ayahFirstWord(s.n, a);
        const end = first + this.corpus.ayahWordCount(s.n, a);
        this.ayahIds.push({
          surah: s.n,
          ayah: a,
          ids: this.table.encode(this.corpus.text.slice(this.corpus.wordStart[first], this.corpus.wordStart[end])),
        });
      }
    }
    this.engine = this.makeEngine();
  }

  static async create(opts: ZipformerSessionOptions): Promise<ZipformerSession> {
    const io = opts.io ? await resolveSource(opts.io) : DEFAULT_ZIPFORMER_IO;
    const corpusJson = await resolveSource(opts.corpus);
    const quranDb = await resolveQuranDb(opts.quran);

    let runner: ZipformerRunner;
    if (opts.session) {
      if (!opts.Tensor) {
        throw new Error("ZipformerSession: `session` also needs the runtime's `Tensor`");
      }
      runner = ZipformerRunner.fromSession(opts.session, io, opts.Tensor);
    } else {
      if (!opts.ort || !opts.model) {
        throw new Error("ZipformerSession: pass `ort` + `model`, or `session` + `Tensor`");
      }
      const loaded = await resolveSource(opts.model);
      const bytes = loaded instanceof Uint8Array ? loaded : new Uint8Array(loaded);
      runner = await ZipformerRunner.create(
        opts.ort,
        bytes,
        io,
        opts.executionProviders,
      );
    }
    return new ZipformerSession(runner, corpusJson, quranDb, opts);
  }

  /** Every ayah the tracker has scored so far, in the order it first saw them. */
  get tallies(): AyahTally[] {
    return [...this.accumulated.values()].sort((a, b) => a.firstSeen - b.firstSeen);
  }

  /**
   * The ayahs that clear the emission gate — i.e. the ones `verse_match` was
   * emitted for. Bridges one short skipped ayah when `allowGaps` is set.
   */
  get verses(): BridgedAyahTally[] {
    const gated = this.tallies.filter((t) => ayahMeetsGate(t, this.minWordFraction));
    if (!this.allowGaps) return gated;
    return bridgeGapAyahs(gated, this.tallies, this.gapMaxWords);
  }

  /** Raw phoneme transcript accumulated since the last `reset()`. */
  get transcript(): string {
    return this.transcriptParts.join("");
  }

  /** `"searching"` until the engine locks onto a position, then `"tracking"`. */
  get engineState(): string {
    return this.engine.state;
  }

  /** Effective engine config (defaults merged with the constructor overrides). */
  get config(): EngineConfig {
    return this.cfg;
  }

  /** Latest per-word acoustic verdicts of the active tracker (main or practice
   * engine). Empty before the recitation is located. Diagnostic use only. */
  verdicts(): WordVerdict[] {
    const engine = this.practiceEngine ?? this.engine;
    return engine.tracer?.verdicts(false) ?? [];
  }

  /** Drop all state — new recitation, same model and corpus. */
  reset(): WorkerOutbound[] {
    this.correction.reset();
    this.practiceEngine = null;
    this.accumulated = new Map();
    this.emitted = new Set();
    this.transcriptParts = [];
    this.lastCursor = null;
    this.lastMatch = null;
    this.ayahIssuesRaised = new Set();
    this.skipCandidate = null;
    this.lastFallback = null;
    this.resetDecoder();
    this.engine = this.makeEngine();
    return [];
  }

  /** Push one chunk of mono 16 kHz float32 PCM. Any size; 480 ms works well. */
  async feed(samples: Float32Array): Promise<WorkerOutbound[]> {
    if (this.correction.state.phase === "error" || this.correction.state.phase === "corrected") return [];
    return this.dispatch(await this.feedSamples(samples));
  }

  /**
   * Correction mode: the passage the reciter is about to read (e.g. the ayahs
   * on screen). The tracker then locks onto it at once, stays inside it, and
   * cannot be pulled into a similar passage. Ignored in tracking mode. Kept
   * across {@link reset}; pass null to clear.
   */
  setExpected(passage: ExpectedPassage | null): void {
    this.expected = passage ? { ...passage } : null;
    this.engine.setExpected(this.expected);
  }

  setMode(mode: RecitationMode): WorkerOutbound[] {
    const out = this.correction.state.phase !== 'idle' ? this.correct('close') : [];
    this.correction.setMode(mode);
    this.attachPosteriors(this.engine);
    if (this.practiceEngine) this.attachPosteriors(this.practiceEngine);
    return out;
  }

  /**
   * Turn the slip head on or off. `"strict"` is the no-extra-false-flag
   * cutoff; `"high"` is the sensitivity point. A model without encoder
   * frames keeps the head off either way.
   */
  setSlipHead(mode: false | SlipSensitivity): void {
    this.slipMode = mode;
    this.slipLive = true;
    this.attachPosteriors(this.engine);
    if (this.practiceEngine) this.attachPosteriors(this.practiceEngine);
  }

  correct(action: CorrectionAction): WorkerOutbound[] {
    if (!this.correction.act(action)) return [];
    const state = this.correction.state;
    this.resetDecoder();
    if (state.phase === 'retrying') {
      this.practiceEngine = new RecitationEngine(this.corpus, this.index, this.cfg);
      this.practiceEngine.setStayOnSurah(true);
      this.attachPosteriors(this.practiceEngine);
      this.practiceEngine.track(state.issue!.surah, state.issue!.ayah, 0);
    } else {
      this.practiceEngine = null;
      if (state.phase === 'idle' && state.resume) {
        // Keep verse history, but discard pre-practice acoustic context.
        this.engine = this.makeEngine();
        this.engine.track(state.resume.surah, state.resume.ayah, state.resume.word);
        this.lastCursor = { ...state.resume };
      }
    }
    return this.dispatch([this.correctionMessage()]);
  }

  private correctionMessage(): WorkerOutbound {
    const issue = this.correction.state.issue;
    return { type: 'correction', state: { ...this.correction.state },
      totalWords: issue ? this.wordCount(issue.surah, issue.ayah) : 0 };
  }

  /** Alias of {@link stop} — end of audio, flush the tail, emit the sequence. */
  async flush(): Promise<WorkerOutbound[]> {
    return this.stop();
  }

  async stop(): Promise<WorkerOutbound[]> {
    if (this.correction.state.phase !== "idle") return [];
    this.stopping = true;
    try { return await this.finish(); }
    finally { this.stopping = false; }
  }

  private async finish(): Promise<WorkerOutbound[]> {
    const out: WorkerOutbound[] = [];
    const silence = new Float32Array(Math.round(this.tailSeconds * SAMPLE_RATE));
    out.push(...await this.feedSamples(silence));

    const frames = this.fbank.inputFinished();
    if (frames.length) {
      out.push(...await this.runFrames(frames));
    }
    const flushed = this.decoder.flush();
    if (flushed.length) {
      out.push(...this.consumeTokens(flushed));
    }
    out.push(...this.settleCorrection());

    this.dumpTallies();
    const live = newlyEligibleAyahs(this.accumulated, this.emitted, this.minWordFraction);
    for (const t of live) {
      this.emitted.add(ayahKey(t));
      out.push(this.toVerseMatch(t));
    }
    for (const t of live) out.push(...this.checkAyahGap(t));

    let fallback: FallbackHit | null = null;
    if (this.enableFallback && shouldRunFallback([...this.emitted])) {
      fallback = this.fallbackSearch(this.transcript);
      this.lastFallback = fallback;
      if (fallback) {
        const words = this.corpus.ayahWordCount(fallback.surah, fallback.ayah);
        const tally: AyahTally = {
          surah: fallback.surah,
          ayah: fallback.ayah,
          ok: words,
          unsure: 0,
          wrong: 0,
          skipped: 0,
          pending: 0,
          words,
          firstSeen: this.accumulated.size,
        };
        this.accumulated.set(ayahKey(tally), tally);
        if (!this.emitted.has(ayahKey(tally))) {
          this.emitted.add(ayahKey(tally));
          out.push(this.toVerseMatch(tally));
        }
        if (this.debugEnabled) {
          out.push({
            type: "debug",
            event: "fallback",
            at: Date.now(),
            data: { ...fallback },
          });
        }
      }
    }

    const seq = buildFinalSequence([...this.accumulated.values()], fallback, this.minWordFraction);
    out.push({
      type: "final_sequence",
      verses: seq.verses,
      confidence: Math.round(seq.confidence * 100) / 100,
    });
    out.push({
      type: "raw_transcript",
      text: this.transcript,
      confidence: seq.confidence,
    });
    return this.dispatch(out);
  }

  private makeEngine(): RecitationEngine {
    const engine = new RecitationEngine(this.corpus, this.index, this.cfg);
    engine.setStayOnSurah(this.stayOnSurah);
    engine.startSearch();
    engine.onBeforeRelocate = () => this.dumpTallies();
    this.attachPosteriors(engine);
    engine.setExpected(this.expected);
    return engine;
  }

  /** GOP scoring costs a few CTC Viterbi passes per word; only correction mode reads it.
   * The slip head is the same: tracking never sees `slip`. */
  private attachPosteriors(engine: RecitationEngine): void {
    const correction = this.correction.mode === "correction";
    engine.setPosteriors(correction ? this.posteriors : null);
    engine.setCorrection(correction);
    const slipOn = correction && this.slipMode !== false && this.slipLive;
    engine.setSlip(slipOn ? this.encoderFrames : null, slipOn ? this.slipWeights : null);
    const thr = slipOn ? slipThreshold(this.slipWeights, this.slipMode as SlipSensitivity) : Infinity;
    if (this.correction.thresholds.slipFlag !== thr) {
      this.correction.thresholds = { ...this.correction.thresholds, slipFlag: thr };
    }
  }

  private resetDecoder(): void {
    this.fbank.reset();
    this.decoder.reset();
    this.runner.reset();
    this.posteriors.clear(0);
    this.encoderFrames.clear(0);
    this.slipLive = true;
  }

  wordCount = (surah: number, ayah: number): number =>
    this.corpus.ayahWordCount(surah, ayah);

  private dumpTallies(): void {
    const tracer = this.engine.tracer;
    if (!tracer) return;
    const snap = snapshotTallies(tracer.verdicts(true) as EmissionVerdict[], this.wordCount);
    accumulateSnapshot(this.accumulated, snap);
  }

  private currentSnapshot(): Map<string, AyahTally> {
    const tracer = this.engine.tracer;
    if (!tracer) return new Map();
    return snapshotTallies(tracer.verdicts(true) as EmissionVerdict[], this.wordCount);
  }

  private dispatch(messages: WorkerOutbound[]): WorkerOutbound[] {
    if (this.onEvent) for (const msg of messages) this.onEvent(msg);
    return messages;
  }

  private async feedSamples(samples: Float32Array): Promise<WorkerOutbound[]> {
    const frames = this.fbank.acceptWaveform(samples);
    return this.runFrames(frames);
  }

  private async runFrames(frames: Float32Array[]): Promise<WorkerOutbound[]> {
    if (!frames.length) return [];
    const { logProbs, frames: out, encoder, encoderFrames } = await this.runner.accept(frames);
    if (out === 0) return [];
    this.posteriors.push(logProbs, out, this.decoder.framesDecoded);
    if (this.slipMode && this.slipLive) {
      if (encoder && encoderFrames) {
        this.encoderFrames.push(encoder, encoderFrames, this.decoder.framesDecoded);
      } else {
        // The graph has no encoder-frame output. Leave the rules as they were.
        this.slipLive = false;
        this.encoderFrames.clear(0);
        this.attachPosteriors(this.engine);
        if (this.practiceEngine) this.attachPosteriors(this.practiceEngine);
      }
    }
    const tokens = this.decoder.consume(logProbs, out, this.runner.io.vocabSize);
    return this.consumeTokens(tokens);
  }

  private consumeTokens(tokens: Array<{ sym: string; frame: number; margin: number }>): WorkerOutbound[] {
    const out: WorkerOutbound[] = [];
    if (this.practiceEngine) {
      this.practiceEngine.feed(tokens, this.decoder.framesDecoded);
      const engine = this.practiceEngine;
      const settled = !tokens.length && engine.tracker?.heard.length
        ? this.decoder.framesDecoded - engine.tracker.heard[engine.tracker.heard.length - 1]!.frame >= this.cfg.settleFrames : false;
      if (engine.tracer && engine.tracker && !engine.tracker.lost
        && (engine.tracker.costRate(this.cfg.holdWindow) ?? 0) < this.cfg.holdRate) {
        const raised = this.correction.observe(engine.tracer.verdicts(Boolean(settled)),
          this.correction.state.resume!, this.decoder.framesDecoded);
        out.push(...this.noteMessages());
        if (raised) out.push(this.correctionMessage());
      } else this.correction.clearEvidence();
      return out;
    }
    for (const t of tokens) this.transcriptParts.push(t.sym);
    for (const ev of this.engine.feed(tokens, this.decoder.framesDecoded)) {
      out.push(...this.handle(ev));
    }
    out.push(...this.emitNewMatches());
    const tracker = this.engine.tracker;
    if (!this.stopping && this.engine.tracer && tracker && !tracker.lost && this.lastCursor
      && (tracker.costRate(this.cfg.holdWindow) ?? 0) < this.cfg.holdRate) {
      const last = tracker.heard[tracker.heard.length - 1];
      const settled = !!last && this.decoder.framesDecoded - last.frame >= this.cfg.settleFrames;
      const raised = this.correction.observe(this.engine.tracer.verdicts(settled), this.lastCursor, this.decoder.framesDecoded);
      out.push(...this.noteMessages());
      if (raised) {
        // Retain main-session coverage before a practice exit replaces its tracker.
        this.dumpTallies();
        out.push(this.correctionMessage());
      } else out.push(...this.checkSkippedAyah(false));
    } else {
      this.correction.clearEvidence();
      this.skipCandidate = null;
    }
    if (tokens.length) {
      out.push({
        type: "raw_transcript",
        text: this.transcript,
        confidence: 1,
      });
    }
    return out;
  }

  private handle(ev: EngineEvent): WorkerOutbound[] {
    const out: WorkerOutbound[] = [];
    switch (ev.type) {
      case "cursor": {
        if (ev.surah != null && ev.ayah != null && ev.word != null) {
          this.lastCursor = { surah: ev.surah, ayah: ev.ayah, word: ev.word };
          out.push(this.wordProgress());
        }
        break;
      }
      case "verdicts": {
        if (this.lastCursor) out.push(this.wordProgress());
        break;
      }
      case "lost": {
        // Transient: the tracker keeps its place. Losing and recovering inside
        // one surah is exactly the unclear-ayah case, so the match chain stays.
        this.correction.clearEvidence();
        break;
      }
      case "relocated": {
        this.correction.clearEvidence();
        this.lastMatch = null;
        break;
      }
      case "located": {
        this.correction.clearEvidence();
        this.lastMatch = null;
        if (ev.surah != null && ev.ayah != null) {
          out.push({
            type: "verse_candidate",
            candidates: [{
              surah: ev.surah,
              ayah: ev.ayah,
              confidence: 0.5,
              rank: 0,
              source: "discovery",
            }],
            stable: false,
            final_flush: false,
          });
        }
        break;
      }
      case "idle":
      case "completed": {
        if (ev.type === "completed" || ev.reason === "silent") out.push(...this.settleCorrection());
        this.correction.clearEvidence();
        this.lastMatch = null;
        this.dumpTallies();
        out.push(...this.emitNewMatches(this.accumulated));
        this.engine.startSearch();
        this.resetDecoder();
        this.lastCursor = null;
        break;
      }
      default:
        break;
    }
    if (this.debugEnabled && ev.type !== "cursor" && ev.type !== "verdicts") {
      out.push({
        type: "debug",
        event: ev.type,
        at: Date.now(),
        data: ev as unknown as Record<string, unknown>,
      });
    }
    return out;
  }

  /** Correction mode: last word-level check before the main tracker is dropped. */
  private settleCorrection(): WorkerOutbound[] {
    if (this.practiceEngine) return [];
    const tracker = this.engine.tracer && this.engine.tracker;
    let verdicts: WordVerdict[];
    let cursor = this.lastCursor;
    let tracer = this.engine.tracer;
    if (!tracker) {
      tracer = this.engine.alignBuffer();
      if (!tracer) return [];
      verdicts = tracer.verdicts(true);
      const w = tracer.cursorWordIndex;
      cursor = { surah: this.corpus.wordSurah[w]!, ayah: this.corpus.wordAyah[w]!, word: this.corpus.wordInAyah[w]! };
    } else {
      if (tracker.lost || !cursor) return [];
      verdicts = tracer!.verdicts(true);
    }
    const raised = this.correction.settle(verdicts, cursor);
    const notes = this.noteMessages();
    if (!raised) return [...notes, ...this.checkSkippedAyah(true, tracer!.tracker)];
    this.dumpTallies();
    return [...notes, this.correctionMessage()];
  }

  /**
   * Correction mode with an expected passage: raise `possible_skipped_ayah`
   * when the engine reads the audio on the current ayah as the next one. Live
   * checks must hold for 12 frames; `atSettle` raises at once.
   */
  private checkSkippedAyah(atSettle: boolean, tracker?: Tracker): WorkerOutbound[] {
    if (this.correction.mode !== "correction" || !this.expected || !this.lastCursor && !atSettle) return [];
    const hit = this.engine.skippedAyah(tracker ?? this.engine.tracker);
    const key = hit ? ayahKey(hit) : null;
    const frame = this.decoder.framesDecoded;
    if (!hit || !key || this.ayahIssuesRaised.has(key)) {
      this.skipCandidate = null;
      return [];
    }
    if (!atSettle) {
      if (!this.skipCandidate || this.skipCandidate.key !== key || frame < this.skipCandidate.frame) {
        this.skipCandidate = { key, frame };
        return [];
      }
      if (frame - this.skipCandidate.frame < SKIP_PERSIST_FRAMES) return [];
    }
    this.skipCandidate = null;
    const words = this.wordCount(hit.surah, hit.ayah);
    const issue = {
      surah: hit.surah, ayah: hit.ayah, word: 0,
      wordIndex: this.corpus.ayahFirstWord(hit.surah, hit.ayah),
      kind: "possible_skipped_ayah" as const,
      words,
    };
    const cursor = this.lastCursor ?? { surah: hit.surah, ayah: hit.ayah, word: 0 };
    if (!this.correction.raise(issue, cursor)) return [];
    this.ayahIssuesRaised.add(key);
    this.dumpTallies();
    return [this.correctionMessage()];
  }

  private noteMessages(): WorkerOutbound[] {
    return this.correction.takeNotes().map(issue => ({ type: 'correction_note' as const, issue }));
  }

  private wordProgress(): WorkerOutbound {
    const cursor = this.lastCursor!;
    const tracer = this.engine.tracer;
    const verdicts = (tracer ? tracer.verdicts(true) : []) as EmissionVerdict[];
    return wordProgressFromCursor(cursor, verdicts, this.wordCount(cursor.surah, cursor.ayah));
  }

  private emitNewMatches(source?: Map<string, AyahTally>): WorkerOutbound[] {
    const tallies = source ?? mergeTallies(this.accumulated, this.currentSnapshot());
    const out: WorkerOutbound[] = [];
    const batch = newlyEligibleAyahs(tallies, this.emitted, this.minWordFraction);
    for (const t of batch) {
      this.emitted.add(ayahKey(t));
      out.push(this.toVerseMatch(t));
    }
    for (const t of batch) out.push(...this.checkAyahGap(t));
    return out;
  }

  /**
   * Correction mode only. `t` (ayah N+2) was just matched; if the previous
   * match of this tracker lock was ayah N of the same surah and N+1 was never
   * matched, raise one ayah-level issue for N+1. `possible_skipped_ayah` when
   * nothing of N+1 was heard, `unclear_ayah` when it was heard but not followed.
   * Word-level rules are untouched; this only covers the whole-ayah hole they
   * cannot see (no clear neighbours inside the ayah).
   */
  private checkAyahGap(t: AyahTally): WorkerOutbound[] {
    const prev = this.lastMatch;
    this.lastMatch = { surah: t.surah, ayah: t.ayah };
    if ((this.stopping && !this.expected) || this.correction.mode !== "correction" || !prev || !this.lastCursor) return [];
    if (prev.surah !== t.surah || t.ayah !== prev.ayah + 2) return [];
    const surah = t.surah;
    const ayah = t.ayah - 1;
    const key = ayahKey({ surah, ayah });
    if (this.emitted.has(key) || this.ayahIssuesRaised.has(key)) return [];
    const words = this.wordCount(surah, ayah);
    // How much of the ayah's expected audio the aligner actually heard. A real
    // skip leaves most words `skipped` (heardRatio 0) and lends only a little
    // of the next ayah's onset to the first words; a heard-but-unfollowed ayah
    // has `wrong` words with heardRatio near 1.
    const gapVerdicts = (this.engine.tracer?.verdicts(true) ?? []).filter((v) => v.surah === surah && v.ayah === ayah);
    const heardFraction = gapVerdicts.reduce((sum, v) => sum + Math.min(1, Math.max(0, v.heardRatio || 0)), 0) / Math.max(1, words);
    const kind = heardFraction >= AYAH_HEARD_FRACTION ? "unclear_ayah" as const : "possible_skipped_ayah" as const;
    const issue = {
      surah, ayah, word: 0,
      wordIndex: this.corpus.ayahFirstWord(surah, ayah),
      kind,
      words,
    };
    if (!this.correction.raise(issue, this.lastCursor)) return [];
    this.ayahIssuesRaised.add(key);
    this.dumpTallies();
    return [this.correctionMessage()];
  }

  private toVerseMatch(t: AyahTally): WorkerOutbound {
    const verse = this.quranDb.getVerse(t.surah, t.ayah);
    return {
      type: "verse_match",
      surah: t.surah,
      ayah: t.ayah,
      verse_text: verse?.text_uthmani ?? "",
      surah_name: verse?.surah_name ?? "",
      confidence: Math.round(ayahConfidence(t) * 100) / 100,
      surrounding_verses: surroundingVerses(this.quranDb, t.surah, t.ayah),
    };
  }

  private fallbackSearch(text: string): FallbackHit | null {
    if (!text) return null;
    const stripped = stripPreambles(text, this.table);
    const rest = text.slice(stripped.offset);
    if (stripped.basmala && rest.length < this.cfg.searchMinChars) {
      return { surah: 1, ayah: 1, distance: 0, how: "basmala" };
    }
    const q = this.table.encode(rest.length >= 3 ? rest : text);
    let best: FallbackHit | null = null;
    for (const a of this.ayahIds) {
      if (a.ids.length > 2.5 * q.length + 8 || q.length > 2.5 * a.ids.length + 8) continue;
      const d = normalizedDistance(q, a.ids, this.table);
      if (!best || d < best.distance) best = { surah: a.surah, ayah: a.ayah, distance: d, how: "whole-ayah" };
    }
    if (!best || best.distance > this.fallbackMaxDistance) return null;
    return best;
  }
}

/**
 * Create the default (Zipformer) recognition session.
 *
 * ```ts
 * import * as ort from "onnxruntime-node";
 * const session = await createZipformerSession({
 *   ort,
 *   model: () => readFile("zipformer_interp_gentle_a05.int8.onnx"),
 *   corpus: async () => JSON.parse(await readFile("zipformer_quran.json", "utf8")),
 *   quran: async () => JSON.parse(await readFile("quran.json", "utf8")),
 *   onEvent: (msg) => console.log(msg.type),
 * });
 * // executionProviders default to ["wasm"] under onnxruntime-web, ["cpu"] under node
 * await session.feed(pcm16k);
 * const final = await session.stop();
 * ```
 */
export function createZipformerSession(
  opts: ZipformerSessionOptions,
): Promise<ZipformerSession> {
  return ZipformerSession.create(opts);
}
