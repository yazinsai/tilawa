import { BUFFER_CAP, DEFAULT_CONFIG, type EngineConfig } from "./config.js";
import type { QuranCorpus } from "./corpus.js";
import { expandTokens } from "./ctcDecoder.js";
import { alignSemiGlobal } from "./alignment.js";
import { preamblePending, stripPreambles, type QuranIndex } from "./search.js";
import type { FramePosteriors } from "./posteriors.js";
import type { EncoderFrames, SlipHead } from "./slipHead.js";
import { Tracker } from "./tracker.js";
import { VerdictTracer } from "./verdicts.js";
import type {
  CtcToken,
  EngineEvent,
  EngineState,
  ExpectedPassage,
  HeardChar,
  SearchHint,
  SearchHit,
  WordVerdict,
} from "./types.js";

export class RecitationEngine {
  readonly corpus: QuranCorpus;
  readonly index: QuranIndex;
  readonly cfg: EngineConfig;
  state: EngineState = "searching";
  tracker: Tracker | null = null;
  tracer: VerdictTracer | null = null;
  framesDecoded = 0;
  heardTotal = 0;

  private buffer: HeardChar[] = [];
  private hint: SearchHint | null = null;
  private stay = false;
  private searchStartFrame = 0;
  private lastSearchFrame = 0;
  private lastSearchHeard = 0;
  private lastRelocateFrame = 0;
  private lastProgressFrame = 0;
  private lastCharFrame = 0;
  private locateFailedEmitted = false;
  private lostEmitted = false;
  private completedEmitted = false;
  private struggles = 0;
  private relocateCandidate: { surah: number; ayah: number } | null = null;
  private lastCursorWord = -1;
  private lastStates = new Map<number, string>();
  private prevSettled = false;
  private lastStruggleChars = 0;
  onBeforeRelocate: (() => void) | null = null;
  private posteriors: FramePosteriors | null = null;
  private encoder: EncoderFrames | null = null;
  private slipHead: SlipHead | null = null;
  private correction = false;
  private expected: { surah: number; ayah: number; firstWord: number; endWord: number } | null = null;
  private expectedLocked = false;

  constructor(corpus: QuranCorpus, index: QuranIndex, cfg: EngineConfig = DEFAULT_CONFIG) {
    this.corpus = corpus;
    this.index = index;
    this.cfg = cfg;
  }

  setHint(hint: SearchHint | null): void {
    this.hint = hint;
  }

  /** Frame posteriors for GOP scoring of verdicts (correction mode); null disables. */
  setPosteriors(posteriors: FramePosteriors | null): void {
    this.posteriors = posteriors;
    if (this.tracer) this.tracer.posteriors = posteriors;
  }

  /** Encoder frames for the slip head. Null leaves `WordVerdict.slip` unset. */
  setSlip(encoder: EncoderFrames | null, head: SlipHead | null): void {
    this.encoder = encoder;
    this.slipHead = head;
    if (this.tracer) {
      this.tracer.encoder = encoder;
      this.tracer.slipHead = head;
    }
  }

  /** Correction mode: the back-fill and stop-time alignment knobs apply. */
  setCorrection(on: boolean): void {
    this.correction = on;
    if (this.tracer) this.tracer.anchorAyahEnd = on ? this.cfg.anchorAyahEnd : 0;
  }

  /**
   * Correction mode: the passage the reciter was asked to read. The first lock
   * goes straight to its first word (after any isti'adha / basmala), later
   * searches only lock inside it, the tracker pays `outsideJumpCost` to jump
   * out of it, and nothing relocates to another surah. Null clears it.
   */
  setExpected(p: ExpectedPassage | null): void {
    this.expectedLocked = false;
    if (!p || !this.corpus.hasAyah(p.surah, p.ayah)) {
      this.expected = null;
      return;
    }
    const end = Math.max(p.ayah, Math.min(p.ayahEnd ?? p.ayah, this.corpus.surahs[p.surah - 1]!.ayahCount));
    this.expected = {
      surah: p.surah,
      ayah: p.ayah,
      firstWord: this.corpus.ayahFirstWord(p.surah, p.ayah),
      endWord: this.corpus.ayahFirstWord(p.surah, end) + this.corpus.ayahWordCount(p.surah, end),
    };
  }

  private get window(): { surah: number; ayah: number; firstWord: number; endWord: number } | null {
    return this.correction ? this.expected : null;
  }

  setStayOnSurah(stay: boolean): void {
    this.stay = stay;
  }

  startSearch(): void {
    this.state = "searching";
    this.tracker = null;
    this.tracer = null;
    this.buffer = [];
    this.heardTotal = 0;
    this.searchStartFrame = this.framesDecoded;
    this.lastSearchFrame = this.framesDecoded;
    this.lastSearchHeard = 0;
    this.lastRelocateFrame = this.framesDecoded;
    this.lastProgressFrame = this.framesDecoded;
    this.lastCharFrame = this.framesDecoded;
    this.locateFailedEmitted = false;
    this.lostEmitted = false;
    this.completedEmitted = false;
    this.struggles = 0;
    this.relocateCandidate = null;
    this.lastCursorWord = -1;
    this.lastStates.clear();
    this.prevSettled = false;
    this.lastStruggleChars = 0;
  }

  track(surah: number, ayah: number, word = 0): EngineEvent[] {
    const wordIndex = this.corpus.wordIndex(surah, ayah, word);
    return this.lock(wordIndex, [], "located");
  }

  feed(tokens: readonly CtcToken[], framesDecoded: number): EngineEvent[] {
    if (framesDecoded < this.framesDecoded) {
      this.searchStartFrame = framesDecoded;
      this.lastSearchFrame = framesDecoded;
      this.lastRelocateFrame = framesDecoded;
      this.lastProgressFrame = framesDecoded;
      this.lastCharFrame = framesDecoded;
    }
    this.framesDecoded = framesDecoded;
    const chars = expandTokens(tokens);
    if (chars.length) {
      this.lastCharFrame = framesDecoded;
      this.heardTotal += chars.length;
      this.buffer.push(...chars);
      if (this.buffer.length > BUFFER_CAP) {
        this.buffer.splice(0, this.buffer.length - BUFFER_CAP);
      }
    }
    if (this.state === "searching") return this.feedSearching();
    return this.feedTracking(chars);
  }

  lock(
    wordIndex: number,
    replay: readonly HeardChar[],
    how: "located" | "relocated",
    from?: { surah: number; ayah: number },
  ): EngineEvent[] {
    if (how === "relocated") this.onBeforeRelocate?.();
    const surah = this.corpus.wordSurah[wordIndex]!;
    const ayah = this.corpus.wordAyah[wordIndex]!;
    const word = this.corpus.wordInAyah[wordIndex]!;
    const prev = this.tracker
      ? {
          surah: this.tracker.surah,
          ayah: this.corpus.wordAyah[this.tracker.cursorWordIndex]!,
        }
      : from;
    const win = this.window;
    this.tracker = new Tracker(this.corpus, this.index.table, surah, wordIndex, this.cfg,
      win && win.surah === surah ? win : null);
    this.tracer = new VerdictTracer(this.tracker, this.index.table, this.cfg);
    this.tracer.anchorAyahEnd = this.correction ? this.cfg.anchorAyahEnd : 0;
    if (win) this.expectedLocked = true;
    this.tracer.posteriors = this.posteriors;
    this.tracer.encoder = this.encoder;
    this.tracer.slipHead = this.slipHead;
    this.state = "tracking";
    this.lostEmitted = false;
    this.completedEmitted = false;
    this.struggles = 0;
    this.relocateCandidate = null;
    this.lastCursorWord = -1;
    this.lastStates.clear();
    this.prevSettled = false;
    this.lastRelocateFrame = this.framesDecoded;
    this.lastStruggleChars = this.heardTotal;
    const events: EngineEvent[] = [];
    if (how === "relocated" && prev) {
      events.push({
        type: "relocated",
        from: prev,
        to: { surah, ayah, word },
      });
    } else {
      events.push({
        type: "located",
        surah,
        ayah,
        word,
        replayed: replay.length,
      });
    }
    if (replay.length) this.tracker.feed(replay);
    events.push(...this.trackingEvents(false));
    return events;
  }

  private feedSearching(): EngineEvent[] {
    const win = this.window;
    if (win && !this.expectedLocked) {
      const start = this.expectedStart();
      if (start >= 0) return this.lock(win.firstWord, this.buffer.slice(start), "located");
    }
    const events: EngineEvent[] = [];
    const minC = this.cfg.searchMinChars;
    const due =
      this.buffer.length >= minC &&
      (this.heardTotal - this.lastSearchHeard >= this.cfg.searchEveryChars ||
        (this.heardTotal - this.lastSearchHeard > 0 &&
          this.framesDecoded - this.lastSearchFrame >= this.cfg.searchEveryFrames));
    if (due) {
      this.lastSearchFrame = this.framesDecoded;
      this.lastSearchHeard = this.heardTotal;
      const qLen = Math.min(this.cfg.searchQueryChars, this.buffer.length);
      const qStart = this.buffer.length - qLen;
      const query = this.buffer.slice(qStart).map((c) => c.ch).join("");
      const result = this.index.search(query, win ?? this.hint);
      const inWindow = !win || (result.hits[0] && result.hits[0].wordIndex >= win.firstWord
        && result.hits[0].wordIndex < win.endWord);
      if (result.decisive && result.hits[0] && inWindow) {
        const hit = result.hits[0];
        const at = qStart + hit.queryStart;
        const back = this.correction ? this.backfill(hit, at) : null;
        if (back) return this.lock(back.wordIndex, this.buffer.slice(back.from), "located");
        return this.lock(hit.wordIndex, this.buffer.slice(at), "located");
      }
    }
    if (
      !this.locateFailedEmitted &&
      this.framesDecoded - this.searchStartFrame >= this.cfg.locateFailedFrames
    ) {
      this.locateFailedEmitted = true;
      events.push({ type: "locateFailed" });
    }
    return events;
  }

  private feedTracking(chars: readonly HeardChar[]): EngineEvent[] {
    if (!this.tracker || !this.tracer) return [];
    if (chars.length) this.tracker.feed(chars);
    const events = this.trackingEvents(chars.length > 0);
    if (this.tracker.lost) {
      if (!this.lostEmitted) {
        this.lostEmitted = true;
        events.push({ type: "lost" });
      }
    } else {
      this.lostEmitted = false;
    }

    if (this.framesDecoded - this.lastRelocateFrame >= this.cfg.relocateEveryFrames) {
      this.lastRelocateFrame = this.framesDecoded;
      const heardSinceTick = this.heardTotal - this.lastStruggleChars;
      this.lastStruggleChars = this.heardTotal;
      if (this.stay || this.window) {
        this.struggles = 0;
      } else {
        const moved = this.maybeRelocate();
        if (moved) return [...events, ...moved];
        if (heardSinceTick > 0) {
          const held = this.isHeld();
          this.struggles = this.tracker.lost || held ? this.struggles + 1 : 0;
          if (this.cfg.maxStruggles > 0 && this.struggles >= this.cfg.maxStruggles) {
            events.push({ type: "idle", reason: "lost" });
            this.struggles = 0;
            this.lastProgressFrame = this.framesDecoded;
          }
        }
      }
    }

    if (this.framesDecoded - this.lastProgressFrame >= this.cfg.idleFrames) {
      events.push({ type: "idle", reason: "silent" });
      this.lastProgressFrame = this.framesDecoded;
    }
    return events;
  }

  private maybeRelocate(): EngineEvent[] | null {
    if (!this.tracker || this.buffer.length < this.cfg.searchMinChars) return null;
    const qLen = Math.min(this.cfg.relocateQueryChars, this.buffer.length);
    const qStart = this.buffer.length - qLen;
    const query = this.buffer.slice(qStart).map((c) => c.ch).join("");
    const result = this.index.search(query, null, 1);
    const hit = result.hits[0];
    const rate = this.tracker.costRate();
    const candidate = hit ? { surah: hit.surah, ayah: hit.ayah } : null;
    const agrees =
      !!candidate &&
      !!this.relocateCandidate &&
      candidate.surah === this.relocateCandidate.surah &&
      candidate.ayah === this.relocateCandidate.ayah;
    this.relocateCandidate = candidate;
    if (
      hit &&
      rate !== null &&
      rate >= this.cfg.lostRate &&
      hit.surah !== this.tracker.surah &&
      hit.distance <= this.cfg.relocateMaxDistance &&
      hit.distance + this.cfg.relocateRateMargin <= rate &&
      agrees
    ) {
      const from = {
        surah: this.tracker.surah,
        ayah: this.corpus.wordAyah[this.tracker.cursorWordIndex]!,
      };
      const replay = this.buffer.slice(qStart + hit.queryStart);
      return this.lock(hit.wordIndex, replay, "relocated", from);
    }
    return null;
  }

  /** Buffer offset where the expected passage starts: past any isti'adha /
   * basmala, once enough has been heard to rule a still-growing one out. -1
   * while waiting. A passage that itself opens with the basmala keeps it. */
  private expectedStart(): number {
    const win = this.window!;
    if (this.buffer.length < this.cfg.searchMinChars) return -1;
    const text = this.buffer.map((c) => c.ch).join("");
    const opensWithBasmala = win.ayah === 1 && win.surah === 1;
    const offset = opensWithBasmala ? 0 : stripPreambles(text, this.index.table).offset;
    const rest = text.slice(offset);
    if (rest.length < this.cfg.searchMinChars) return -1;
    if (!opensWithBasmala && preamblePending(rest, this.index.table)) return -1;
    return offset;
  }

  /**
   * Search locks only once a query is decisive, often several words into the
   * ayah, and the words before the hit never get a verdict. Start at the
   * ayah's first word instead and replay as many earlier heard chars as those
   * words should take (`backfillRatio` per expected char), never reaching
   * back into an isti'adha / basmala. `at` is the hit's offset in the buffer.
   */
  private backfill(hit: SearchHit, at: number): { wordIndex: number; from: number } | null {
    const ratio = this.cfg.backfillRatio;
    if (!(ratio > 0)) return null;
    const first = this.corpus.ayahFirstWord(hit.surah, hit.ayah);
    const before = this.corpus.wordStart[hit.wordIndex]! - this.corpus.wordStart[first]!;
    if (before <= 0) return null;
    const floor = stripPreambles(this.buffer.slice(0, at).map((c) => c.ch).join(""), this.index.table).offset;
    const from = Math.max(floor, at - Math.ceil(before * ratio));
    return from < at ? { wordIndex: first, from } : null;
  }

  /**
   * Correction mode, end of a take that never locked: align the buffered
   * chars to the best search hit (decisive or not) at or under
   * `stopAlignDistance`, for one final word check. The result is a detached
   * tracer; it never feeds verse tallies.
   */
  alignBuffer(): VerdictTracer | null {
    const maxD = this.cfg.stopAlignDistance;
    const win = this.window;
    if (!this.correction || this.state !== "searching" || !(maxD > 0 || win)) return null;
    if (this.buffer.length < this.cfg.searchMinChars) return null;
    const qLen = Math.min(this.cfg.searchQueryChars, this.buffer.length);
    const qStart = this.buffer.length - qLen;
    const query = this.buffer.slice(qStart).map((c) => c.ch).join("");
    if (win && !this.expectedLocked) {
      const text = this.buffer.map((c) => c.ch).join("");
      const from = win.ayah === 1 && win.surah === 1 ? 0 : stripPreambles(text, this.index.table).offset;
      return this.detachedTracer(win.surah, win.firstWord, this.buffer.slice(from));
    }
    const hit = this.index.search(query, win ?? this.hint).hits[0];
    if (!hit || !(hit.distance <= (maxD > 0 ? maxD : this.cfg.searchDecisiveDistance))) return null;
    if (win && (hit.wordIndex < win.firstWord || hit.wordIndex >= win.endWord)) return null;
    const at = qStart + hit.queryStart;
    const back = this.backfill(hit, at) ?? { wordIndex: hit.wordIndex, from: at };
    return this.detachedTracer(hit.surah, back.wordIndex, this.buffer.slice(back.from));
  }

  /**
   * Correction mode with an expected passage: the interior ayah A the tracker
   * is in, when the heard chars it aligned to A read as ayah A+1 instead (the
   * reciter skipped A and the aligner bent A+1's audio onto A's text). Uses
   * `tracker` when given (a detached stop-time tracker), else the live one.
   */
  skippedAyah(tracker: Tracker | null = this.tracker): { surah: number; ayah: number } | null {
    const win = this.window;
    const min = this.cfg.skipMinChars;
    if (!win || !tracker || !(min > 0) || tracker.surah !== win.surah) return null;
    const c = this.corpus;
    const n = tracker.heard.length;
    const ayahOf = (g: number) => {
      const cell = tracker.trail[g]!;
      return c.wordAyah[tracker.firstWord + (cell <= 0 ? 0 : tracker.localWordOfPos[Math.min(cell, tracker.len) - 1]!)]!;
    };
    if (n < min) return null;
    const a = ayahOf(n - 1);
    const firstAyah = c.wordAyah[win.firstWord]!;
    const lastAyah = c.wordAyah[win.endWord - 1]!;
    if (a <= firstAyah || a >= lastAyah) return null;
    let from = n - 1;
    while (from > 0 && ayahOf(from - 1) === a) from--;
    if (from === 0 || ayahOf(from - 1) !== a - 1 || n - from < min) return null;
    const table = this.index.table;
    const ids = table.encode(tracker.heard.slice(from).map((h) => h.ch).join(""));
    const text = (ayah: number) => {
      const w0 = c.ayahFirstWord(win.surah, ayah);
      return table.encode(c.text.slice(c.wordStart[w0]!, c.wordStart[w0 + c.ayahWordCount(win.surah, ayah)]!));
    };
    const here = text(a);
    const next = text(a + 1);
    const dHere = alignSemiGlobal(ids, here, 0, here.length, table).distance;
    const fit = alignSemiGlobal(ids, next, 0, next.length, table);
    if (fit.distance > this.cfg.skipMaxDistance || dHere - fit.distance < this.cfg.skipMargin) return null;
    // The A+1 reading must account for the start of what was heard: audio on
    // A followed by A+1 (A itself decoded badly) is not a skip.
    if (fit.queryStart > this.cfg.skipMaxHead * ids.length) return null;
    return { surah: win.surah, ayah: a };
  }

  private detachedTracer(surah: number, wordIndex: number, chars: readonly HeardChar[]): VerdictTracer {
    const win = this.window;
    const tracker = new Tracker(this.corpus, this.index.table, surah, wordIndex, this.cfg,
      win && win.surah === surah ? win : null);
    tracker.feed(chars);
    const tracer = new VerdictTracer(tracker, this.index.table, this.cfg);
    tracer.posteriors = this.posteriors;
    tracer.anchorAyahEnd = this.cfg.anchorAyahEnd;
    return tracer;
  }

  private isHeld(): boolean {
    if (!this.tracker) return false;
    const rate = this.tracker.costRate(this.cfg.holdWindow);
    return rate !== null && rate >= this.cfg.holdRate;
  }

  private settled(): boolean {
    return this.framesDecoded - this.lastCharFrame >= this.cfg.settleFrames;
  }

  private trackingEvents(gotChars: boolean): EngineEvent[] {
    if (!this.tracker || !this.tracer) return [];
    if (this.isHeld()) return [];
    const events: EngineEvent[] = [];
    const cursorIdx = this.tracker.cursorWordIndex;
    if (cursorIdx !== this.lastCursorWord) {
      this.lastCursorWord = cursorIdx;
      this.lastProgressFrame = this.framesDecoded;
      events.push({
        type: "cursor",
        surah: this.corpus.wordSurah[cursorIdx]!,
        ayah: this.corpus.wordAyah[cursorIdx]!,
        word: this.corpus.wordInAyah[cursorIdx]!,
        wordIndex: cursorIdx,
      });
    }
    const settled = this.settled();
    const silenceSettle = !gotChars && settled && !this.prevSettled;
    const vs = this.tracer.verdicts(settled);
    const changes: WordVerdict[] = [];
    const refreshPending = gotChars && this.prevSettled;
    for (const v of vs) {
      const key = `${v.state}:${v.distance}:${v.heardRatio}:${v.margin}`;
      const prev = this.lastStates.get(v.wordIndex);
      if (prev === key) continue;
      const wasPending = prev?.startsWith("pending:") ?? false;
      if (wasPending && v.state === "pending" && !refreshPending) continue;
      changes.push(v);
      this.lastStates.set(v.wordIndex, key);
      if (v.state !== "pending" && !silenceSettle) {
        this.lastProgressFrame = this.framesDecoded;
      }
    }
    if (changes.length) events.push({ type: "verdicts", changes });
    for (const key of this.lastStates.keys()) {
      if (!vs.some((v) => v.wordIndex === key)) this.lastStates.delete(key);
    }
    this.prevSettled = settled;
    if (!this.completedEmitted && this.tracker.reachedEnd) {
      const lastW = this.tracker.endWord - 1;
      const lastV = vs.find((v) => v.wordIndex === lastW);
      if (lastV && lastV.state !== "pending") {
        this.completedEmitted = true;
        events.push({ type: "completed", surah: this.tracker.surah });
      }
    }
    return events;
  }
}
