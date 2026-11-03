import type { AlignmentOp } from "./align.js";
import { PosteriorFrames } from "./ctc.js";
import { MID_WORD_START, UNIT, alignSubstring, collapseHeld, insertionCost, isHeld } from "./fuzzy.js";
import { EVENT_SCHEMA_VERSION } from "./events.js";
import type { EngineEvent, PositionCandidate, PositionEvent, TraceCandidate, TrackerState, TrackerTrace, WordRef } from "./events.js";
import { EvidenceError } from "./evidence.js";
import type { DecodedPhoneme, PhonemeEvidence } from "./evidence.js";
import { createLocalSearch } from "./local.js";
import type { LocalPosition, LocalSearch } from "./local.js";
import type { AyahRef, PhraseAsset, ReferenceIndex } from "./reference.js";
import { letterSkeleton } from "./parallel-phrases.js";
import { continuationDecision, longestSpanAt, spanId, slotSpans } from "./slot-spans.js";
import type { SlotSpan } from "./slot-spans.js";

export interface TrackerOptions {
  mode?: "free" | "memorization";
  /** Grams occurring more often than this carry too little location information to vote. */
  maxGramOccurrences?: number;
  maxCandidates?: number;
  /** Fit: a candidate fits only with PER at most this. Default 0.35. */
  maxPer?: number;
  /** Fit: and weighted cost per observed phoneme at most this. Default {@link MISFIT_MAX_COST}. */
  maxCost?: number;
  /** Evidence: `locked` needs at least this many observed phonemes. Default 10. */
  minPhonemes?: number;
  /** Evidence: and a span of at least this many words. Default 2. */
  minWords?: number;
  /**
   * Separation: `locked` needs the best distinct candidate to trail by `max(minSeparation, separationScale / n)`
   * plain edits for `n` observed phonemes, so short clips need a wider margin. Defaults {@link COMMIT_MIN_SEPARATION}
   * and {@link COMMIT_SEPARATION_SCALE}.
   */
  minSeparation?: number;
  separationScale?: number;
  /** `locked` also needs the top candidate's confidence at least this; a unique but poor fit stays `acquiring`. Default 0.08. */
  minLockConfidence?: number;
  /** Bonus for the intended or filename ayah (full) and its neighbours (half). Bounded so it can't override fit or separation. */
  prior?: { surah: number; ayah: number; weight: number };
  /** Full prior bonus, uniformly, for any ayah inside one of the inclusive ranges (a surah or a page); `ayah: Infinity` ends at the surah's last ayah. Same cap as `prior`. */
  priorRange?: { ranges: { from: AyahRef; to: AyahRef }[]; weight: number };
  /**
   * Once locked, a distant location must hold a global lock for this many consecutive observed phonemes before the
   * tracker jumps there. Default 8.
   */
  jumpPhonemes?: number;
  /**
   * Non-Quran phrases to recognise and skip in a stream. A basmala outside 1:1 and 27:30 is skipped too, and then
   * favours surah openings. Skipping only acts on earlier pushes, so a single-push clip is unaffected.
   */
  phrases?: PhraseAsset;
  /** Attach shortlist, state transition and alignment path to every position event. */
  trace?: boolean;
  /**
   * Streaming: before the first lock, run global search once per this much audio on a fixed grid of evidence time
   * rather than once per push. How often a stream is searched decides how short a prefix can lock, so with this set
   * the events don't depend on how the caller chunks evidence. Default: once per push (offline locate of a whole clip).
   */
  acquireMs?: number;
}

export interface Tracker {
  push(evidence: PhonemeEvidence): EngineEvent[];
  end(): EngineEvent[];
}

interface Match {
  offset: number;
  length: number;
  /** Weighted cost in `UNIT`s. */
  cost: number;
  per: number;
  votes: number;
  path: AlignmentOp[];
  basmala?: boolean;
  /** The decoded alignment cost, kept when posteriors replace `cost` for ranking. */
  decodedCost?: number;
  /** With `basmala`: observed phonemes before this index are the prefix, which costs `prefixCost`. */
  prefixEnd?: number;
  prefixCost?: number;
}

const MAX_BASMALA_PER = 0.25;
/** The largest prior bonus, in `UNIT`s; below `minSeparation` so a prior alone never separates. */
const MAX_PRIOR_BONUS = UNIT / 2;
/** Nats of CTC likelihood, relative to the decoded sequence, that cost as much as one plain edit. */
const NATS_PER_EDIT = 3;
/** Posteriors are rejected when fewer decoded phonemes than this appear in the top-k over their frames. */
const MIN_LISTED = 0.5;
/**
 * While acquiring, at most this many earlier phonemes are kept when a new chunk arrives. The chunk itself is always
 * searched whole, so a single-push clip is unaffected; streamed acquisitions longer than this depend on chunking.
 */
const MAX_HISTORY = 96;
/** Local recent cost, in plain edits per phoneme, above which global search runs alongside local tracking. */
const WEAK_COST = 0.3;
/** Phonemes before the weakness was noticed that the global search also sees: recent cost lags the deviation. */
const LOOKBACK = 6;
/** Global search while weak or after `lost` looks at only the latest this many phonemes. */
const WINDOW = 48;
/** While weak: hold the anchor once the window (lookback included) holds this many phonemes, nothing fits globally and local cost exceeds `maxCost`. `lost` follows only when that search locks off the anchor. */
const LOST_AFTER = 24;
/** A global lock within this many ayahs of the local bookmark is the local search's to follow, not a jump. */
const NEAR_AYAHS = 2;
/** Phonemes by which recent cost lags a deviation: half the local search's recent-cost window. */
const WEAK_LAG = 8;
/** A new local run shows once its last this many phonemes cost at most `CONFIRM_COST` plain edits each on average. */
const CONFIRM = 3;
const CONFIRM_COST = 0.34;
/**
 * After a skipped takbir, al-Fatiha's opening gets this bonus: in prayer each rak'ah restarts with it, and its al-hamd
 * opening is shared with five other passages, so the half-edit `prior` bonus couldn't separate it.
 */
const FATIHA_BONUS = 2 * UNIT;
/** A phrase is heard when it aligns to the latest phonemes at most this weighted cost per phrase phoneme. */
const PHRASE_MAX_COST = 0.3;
/** While locked, a heard phrase is skipped only if local search fit it this badly, per phoneme: else it's Quran text. */
const PHRASE_LOCAL_COST = 0.25;
/** Phonemes kept for phrase detection; longer than any phrase as recited. */
const HEARD = 64;
/**
 * A phrase in progress holds off locking once this many of its phonemes, and this share of it, are heard. Shorter
 * prefixes ("Allahu" of the takbir) are everyday Quran text.
 */
const PHRASE_PREFIX = 12;
const PHRASE_PREFIX_SHARE = 0.4;
/** A restart in progress holds off locking once this many phonemes restate the window's start. */
const RESTART_PREFIX = 6;
/** Repeated blocks this long, at most this weighted cost per phoneme apart, collapse to one copy in a stream. */
const MIN_REPEAT = 8;
const MAX_REPEAT = 40;
const REPEAT_MAX_COST = 0.15;

const ayahCompare = (a: AyahRef, b: AyahRef) => a.surah - b.surah || a.ayah - b.ayah;

/** Local recent cost above this, in plain edits per phoneme, is the misfit that starts a hold. */
export const MISFIT_MAX_COST = 0.35;
/** Commit separation, in plain edits: `max(COMMIT_MIN_SEPARATION, COMMIT_SEPARATION_SCALE / n)`. */
export const COMMIT_MIN_SEPARATION = 1;
export const COMMIT_SEPARATION_SCALE = 30;

/** The lock's commit margin for `n` observed phonemes, in plain edits. */
export function commitMargin(n: number, minSeparation = COMMIT_MIN_SEPARATION, separationScale = COMMIT_SEPARATION_SCALE): number {
  return Math.max(minSeparation, separationScale / Math.max(1, n));
}

export interface IssueDropFit {
  n: number;
  /** Weighted cost per observed phoneme, in plain edits. */
  unedited: number;
  edited: number;
  /** Plain edits by which the edited anchor beats the written one. */
  gain: number;
  /** {@link commitMargin} for this span. */
  margin: number;
  explains: boolean;
}

/**
 * Cost of aligning every heard phoneme to a substring of `reference`, starting at its first phoneme, in `UNIT`s.
 * The same weighted word costs as local tracking (`subCost`, `insertionCost`, `deletionCost`).
 */
export function anchorFitCost(reference: string, heard: string): number | null {
  if (!reference || !heard) return null;
  const fit = alignSubstring(reference, 0, reference.length, heard, (offset) => offset === 0, { strictStart: true });
  return fit ? fit.cost : null;
}

/**
 * Whether applying an issue's edit to the anchor text explains a misfit. `written` is the anchor from the issue
 * through the drop; `edited` is that text with the issue's own edit applied; `heard` is the phonemes between them.
 * The edited fit must beat the unedited fit by the commit margin and fall below the misfit cost. No new threshold.
 */
export function issueDropFit(
  written: string,
  edited: string,
  heard: string,
  maxCost = MISFIT_MAX_COST,
  minSeparation = COMMIT_MIN_SEPARATION,
  separationScale = COMMIT_SEPARATION_SCALE,
): IssueDropFit | null {
  const uneditedCost = anchorFitCost(written, heard);
  const editedCost = anchorFitCost(edited, heard);
  if (uneditedCost === null || editedCost === null) return null;
  const n = heard.length;
  const gain = (uneditedCost - editedCost) / UNIT;
  const margin = commitMargin(n, minSeparation, separationScale);
  const unedited = uneditedCost / UNIT / n;
  const editedPer = editedCost / UNIT / n;
  return { n, unedited, edited: editedPer, gain, margin, explains: gain >= margin && editedPer < maxCost };
}

/**
 * An added word's heard phones deleted, the phones after them kept. `heard` is the issue through the drop;
 * `kept` is `heard` without the inserted span. The margin is the commit margin of the full span, so a short
 * tail is not asked to clear the margin a longer one would. Same word costs as {@link issueDropFit}.
 */
export function insertionDropFit(
  reference: string,
  heard: string,
  kept: string,
  maxCost = MISFIT_MAX_COST,
  minSeparation = COMMIT_MIN_SEPARATION,
  separationScale = COMMIT_SEPARATION_SCALE,
): IssueDropFit | null {
  if (!reference || !heard || !kept) return null;
  const uneditedCost = anchorFitCost(reference, heard);
  const editedCost = anchorFitCost(reference, kept);
  if (uneditedCost === null || editedCost === null) return null;
  const n = heard.length;
  const gain = (uneditedCost - editedCost) / UNIT;
  const margin = commitMargin(n, minSeparation, separationScale);
  const unedited = uneditedCost / UNIT / n;
  const editedPer = editedCost / UNIT / n;
  return { n, unedited, edited: editedPer, gain, margin, explains: gain >= margin && editedPer < maxCost };
}

export interface DestinationWord {
  /** Phonemes of the destination word this span aligned to. */
  destination: string;
  /** Heard phonemes the destination alignment gave that word. */
  heard: string;
}

/**
 * Heard words that fit a relocation's destination better than the anchor continuation with exactly one
 * inserted heard word, by the commit margin. A word the insertion hypothesis also fits, such as the anchor's
 * own next word, counts zero. `total` is the winning hypothesis's cost, in `UNIT`s, so callers can prefer the
 * anchor start the hypothesis actually explains.
 */
export function relocationSupport(
  anchorWords: readonly string[],
  heard: readonly DestinationWord[],
  minSeparation = COMMIT_MIN_SEPARATION,
  separationScale = COMMIT_SEPARATION_SCALE,
): { count: number; total: number } | null {
  const words = heard.filter((h) => h.heard.length > 0 && h.destination.length > 0);
  if (!words.length || !anchorWords.length) return null;
  let best: { count: number; total: number } | null = null;
  for (let ins = 0; ins < words.length; ins++) {
    let ai = 0;
    let total = 0;
    let count = 0;
    let placed = true;
    for (let i = 0; i < words.length; i++) {
      const h = words[i];
      const dest = anchorFitCost(h.destination, h.heard);
      let hyp: number | null = null;
      if (i === ins) {
        let c = 0;
        for (const ch of h.heard) c += insertionCost("", 0, ch);
        hyp = c;
      } else if (ai < anchorWords.length) {
        hyp = anchorFitCost(anchorWords[ai], h.heard);
        ai++;
      }
      if (dest === null || hyp === null) {
        placed = false;
        break;
      }
      // The commit margin widens on a short span. A word the destination reads exactly is already that sure, so it
      // counts once it leads by the margin's floor; a partial or imperfect reading has to clear the wide margin, which
      // is what keeps a shared word heard in part from voting.
      const gain = (hyp - dest) / UNIT;
      const margin = commitMargin(h.heard.length, minSeparation, separationScale);
      if (gain >= margin || (dest === 0 && gain >= minSeparation)) count++;
      total += hyp;
    }
    if (placed && (!best || total < best.total)) best = { count, total };
  }
  return best;
}

/** What the tracker is fed from one chunk of listener evidence, in the product and in every bench alike: phonemes only. */
export function trackerInput(evidence: PhonemeEvidence): PhonemeEvidence {
  return { ...evidence, posteriors: undefined };
}

export function createTracker(index: ReferenceIndex, options: TrackerOptions = {}): Tracker {
  const maxGramOccurrences = options.maxGramOccurrences ?? 5000;
  const maxCandidates = options.maxCandidates ?? 64;
  const maxPer = options.maxPer ?? 0.35;
  const maxCost = options.maxCost ?? MISFIT_MAX_COST;
  const minPhonemes = options.minPhonemes ?? 10;
  const minWords = options.minWords ?? 2;
  const minSeparation = options.minSeparation ?? COMMIT_MIN_SEPARATION;
  const separationScale = options.separationScale ?? COMMIT_SEPARATION_SCALE;
  const minLockConfidence = options.minLockConfidence ?? 0.08;
  const prior = options.prior;
  const priorRange = options.priorRange;
  const jumpPhonemes = options.jumpPhonemes ?? 8;
  const memorization = options.mode === "memorization";
  const wordStarts = new Set(index.words.map((w) => w.start));
  const isWordStart = (offset: number) => wordStarts.has(offset);
  const fatiha = index.words.filter((w) => w.surah === 1 && w.ayah === 1);
  const basmala = index.phonemes.slice(fatiha[0].start, fatiha.at(-1)!.end);
  const fatihaAyah2 = index.words.find((w) => w.surah === 1 && w.ayah === 2)!.start;
  const fatihaRabbi = index.words.findIndex((w) => w.surah === 1 && w.ayah === 2 && w.word === 3);

  let observed = "";
  let observedConf: number[] = [];
  const observedFrames: { start: number; end: number }[] = [];
  const tokenIds = new Map([...index.tokens].map((t, i) => [t, i + 1]));
  /** Only the first acquisition reranks with posteriors; dropped at lock, so later global searches are decoded-only. */
  let posteriors: PosteriorFrames | null = null;
  let decodedOnly = false;
  let withPosteriors = false;
  let frameMs: number | null = null;
  let lastFrame = 0;
  let last: PositionEvent | null = null;
  /** Set once locked: from then on the tracker follows the reciter locally, phoneme by phoneme. */
  let local: LocalSearch | null = null;
  let lockedOn: Match | null = null;
  /** The event that reported the lock, kept with its trace until a local event replaces it. */
  let lockEvent: PositionEvent | null = null;
  let emittedSinceLock = false;
  /** Local fit has degraded: global search over the latest phonemes runs alongside the local search. */
  let weak = false;
  /** A distant global lock seen while weak, and for how many consecutive phonemes it has held. */
  let pending: { m: Match; held: number } | null = null;
  /** After a misfit: global search over a rolling window, one phoneme at a time, until it locks again. */
  let reacquiring = false;
  /**
   * Lock kept while that search runs. A misfit used to emit `lost` before any destination had committed, and the
   * checker then dropped the anchor and the tokens heard in the gap. `lost` is emitted only when a lock commits off
   * this anchor, or the session ends still uncommitted. The held event carries `hold: "start"` once. The checker
   * keeps an open issue only when that issue's edit explains the drop, and downgrades the rest; tokens after the
   * signal still enter the window.
   */
  let holding: PositionEvent | null = null;
  /** Set at the first lock. The onset veto applies only to that lock. */
  let everLocked = false;
  /** Slot-differing span the lock is inside, until its continuation picks a member. */
  let twin: { span: SlotSpan; memberStart: number; endPhone: number; buf: string } | null = null;
  /**
   * A lock onto the other member of a parallel span (or an exact copy of the next ayah). The phones heard until
   * the in-order continuation commits are what a retro rebind is scored on.
   */
  let excursion: { home: number; dest: number; length: number; entry: number; phones: string; trimmed: boolean } | null = null;
  /**
   * Holding and continuation resolution run only on the first lock, and only until that lock's position leaves
   * every slot-differing span. A word outside those spans is unique to its location. Any later `startLocal`
   * (a jump, a reacquisition, the re-anchor after a copy change) closes it.
   */
  let arm = false;
  /** Spans already resolved or given up at the lost timeout. Not watched again. */
  const twinSettled = new Set<string>();
  const allSpans = slotSpans(index);
  /** Every phoneme until the first lock, untrimmed, so a first-lock veto can see back to the speech onset. */
  let raw = "";
  let rawConf: number[] = [];
  /** Phones of `raw` consumed by a leading non-Quran phrase the streaming matcher already cut. */
  let onsetCut = 0;
  /** Memorization mode: the expected bookmark held in place after the reciter jumped away from it. */
  let frozen: WordRef | null = null;
  /** The expected bookmark from before local fit last turned weak. */
  let steady: WordRef | null = null;
  /**
   * The expected bookmark after each recent phoneme. Recent cost lags a deviation by several phonemes, during which
   * local skips can already have moved the bookmark; the one from before that lag is what weakness should freeze.
   */
  const expectedHistory: (WordRef | null)[] = [];
  /** The local search's cost for each of the last `CONFIRM` phonemes. */
  const stepCosts: number[] = [];

  const phraseList = options.phrases ? [...options.phrases.phrases, { name: "basmala", text: "", p: basmala }] : [];
  /** The latest phonemes, with the local search's cost for each and its state before each, for phrase detection. */
  let heard = "";
  let heardCost: number[] = [];
  let heardState: unknown[] = [];
  /** While acquiring: observed phonemes up to here end with a phrase, to drop at the next push. */
  let phraseCut = 0;
  /** A basmala was skipped in an earlier push (`surahStart`) or this one (`basmalaHeard`): favour surah openings. */
  let surahStart = false;
  let basmalaHeard = false;
  /** A takbir was skipped: in prayer al-Fatiha comes next (`fatihaNext`), promoted from `takbirHeard` like the basmala. */
  let fatihaNext = false;
  /** Re-acquiring after a basmala ended a locked passage: where it ended, and phonemes since. */
  let afterBasmala: { from: WordRef; held: number } | null = null;
  let takbirHeard = false;
  /** A phrase was just skipped while locked: a boundary, so the next distant lock needn't wait for persistence. */
  let afterPhrase = false;
  let pushes = 0;
  const validate = (e: PhonemeEvidence) => {
    if (e.inventory !== index.inventoryId) {
      throw new EvidenceError(`Evidence inventory "${e.inventory}" does not match index inventory "${index.inventoryId}"`);
    }
    if (frameMs !== null && e.frameMs !== frameMs) {
      throw new EvidenceError(`Frame stride changed from ${frameMs} ms to ${e.frameMs} ms within a session`);
    }
    for (const ph of e.phonemes) {
      if (!index.tokens.has(ph.p)) throw new EvidenceError(`Phoneme "${ph.p}" is not in inventory "${index.inventoryId}"`);
    }
    if (e.posteriors ? decodedOnly : withPosteriors) throw new EvidenceError("Every chunk in a session must carry posteriors, or none");
    const p = e.posteriors;
    if (!p) return;
    if (p.ids.length !== p.nFrames * p.k || p.probs.length !== p.nFrames * p.k) {
      throw new EvidenceError(`Posterior rows hold ${p.ids.length} ids and ${p.probs.length} probs, expected nFrames x k = ${p.nFrames * p.k}`);
    }
    let listed = 0;
    for (const ph of e.phonemes) {
      if (ph.start < p.startFrame || ph.end >= p.startFrame + p.nFrames) {
        throw new EvidenceError(`Phoneme frames ${ph.start}..${ph.end} fall outside posterior frames ${p.startFrame}..${p.startFrame + p.nFrames - 1}; check the frame stride`);
      }
      const id = tokenIds.get(ph.p)!;
      let found = false;
      for (let f = ph.start; f <= ph.end && !found; f++) {
        for (let c = 0; c < p.k; c++) if (p.ids[(f - p.startFrame) * p.k + c] === id) found = true;
      }
      if (found) listed++;
    }
    if (e.phonemes.length && listed < MIN_LISTED * e.phonemes.length) {
      throw new EvidenceError(`Only ${listed} of ${e.phonemes.length} decoded phonemes appear in their frames' top-k; posterior ids don't follow the inventory order`);
    }
  };

  const exactScan = (query: string): Match[] => {
    const found: Match[] = [];
    for (let at = index.phonemes.indexOf(query); at !== -1 && found.length < maxCandidates; at = index.phonemes.indexOf(query, at + 1)) {
      const path: AlignmentOp[] = [...query].map((p, k) => ({ op: "match", ref: p, refAt: k, obsAt: k }));
      found.push({ offset: at, length: query.length, cost: 0, per: 0, votes: 0, path });
    }
    return found;
  };

  const fuzzyMatches = (query: string, queryConf: number[]): Match[] => {
    if (query.length < index.gramSize) return exactScan(query);
    const votes = new Map<number, number>();
    for (let q = 0; q + index.gramSize <= query.length; q++) {
      const hits = index.grams.get(query.slice(q, q + index.gramSize));
      if (!hits || hits.length > maxGramOccurrences) continue;
      for (const pos of hits) {
        const diagonal = Math.max(0, pos - q);
        votes.set(diagonal, (votes.get(diagonal) ?? 0) + 1);
      }
    }
    // Every gram was too common to vote; fall back to a direct scan.
    if (votes.size === 0) return exactScan(query);

    // Learner insertions and deletions shift the diagonal, so nearby diagonals vote for one location.
    const slack = Math.max(4, Math.ceil(query.length * 0.25));
    // Bins are centred on vote peaks with a fixed width: chaining nearby diagonals lets common grams join one bin
    // across the whole Quran on long clips.
    const diagonals = [...votes.keys()].sort((a, b) => a - b);
    const prefix = [0];
    for (const d of diagonals) prefix.push(prefix.at(-1)! + votes.get(d)!);
    const firstAtLeast = (x: number) => {
      let a = 0;
      let b = diagonals.length;
      while (a < b) {
        const mid = (a + b) >> 1;
        if (diagonals[mid] < x) a = mid + 1;
        else b = mid;
      }
      return a;
    };
    const windowed = diagonals.map((d, k) => {
      const a = firstAtLeast(d - slack);
      const b = firstAtLeast(d + slack + 1);
      return { k, lo: diagonals[a], hi: diagonals[b - 1], votes: prefix[b] - prefix[a] };
    });
    windowed.sort((x, y) => y.votes - x.votes || diagonals[x.k] - diagonals[y.k]);
    const bins: { lo: number; hi: number; votes: number; centre: number }[] = [];
    for (const c of windowed) {
      if (bins.length >= maxCandidates) break;
      const centre = diagonals[c.k];
      if (bins.some((b) => Math.abs(b.centre - centre) <= slack)) continue;
      bins.push({ lo: c.lo, hi: c.hi, votes: c.votes, centre });
    }

    const byStart = new Map<number, Match>();
    for (const bin of bins) {
      const lo = Math.max(0, bin.lo - slack);
      const hi = Math.min(index.phonemes.length, bin.hi + query.length + slack);
      const a = alignSubstring(index.phonemes, lo, hi, query, isWordStart, { conf: queryConf });
      if (!a || a.per > maxPer) continue;
      const prev = byStart.get(a.start);
      if (prev && prev.cost <= a.cost) continue;
      byStart.set(a.start, { offset: a.start, length: a.end - a.start, cost: a.cost, per: a.per, votes: bin.votes, path: a.path });
    }
    return [...byStart.values()].sort((a, b) => a.cost - b.cost || a.offset - b.offset);
  };

  /** Matches for the evidence, plus matches for what follows a leading basmala, which the index only holds as 1:1. */
  const locate = (): Match[] => {
    const plain = fuzzyMatches(observed, observedConf);
    const prefix = alignSubstring(observed, 0, observed.length, basmala, (o) => o === 0, { strictStart: true });
    if (!prefix || prefix.per > MAX_BASMALA_PER || observed.length - prefix.end < index.gramSize) return plain;
    const k = prefix.end;
    const after = fuzzyMatches(observed.slice(k), observedConf.slice(k))
      // Basmala then 1:2 is just 1:1–1:2, which the plain search already found.
      .filter((m) => m.offset !== fatihaAyah2)
      .map((m) => ({
        ...m,
        cost: m.cost + prefix.cost,
        basmala: true,
        prefixEnd: k,
        prefixCost: prefix.cost,
        path: m.path.map((o) => (o.op === "del" ? o : { ...o, obsAt: o.obsAt + k })),
      }));
    const byStart = new Map<number, Match>();
    for (const m of [...plain, ...after]) if (!byStart.has(m.offset) || byStart.get(m.offset)!.cost > m.cost) byStart.set(m.offset, m);
    return [...byStart.values()].sort((a, b) => a.cost - b.cost || a.offset - b.offset);
  };

  const toIds = (tokens: string[]) => collapseHeld(tokens).map((t) => tokenIds.get(t)!);
  /**
   * Reranking cost: the CTC likelihood, over every frame of the evidence, of the candidate's alignment read as labels
   * (reference tokens where it matches, substitutes or deletes; observed tokens where it inserts), relative to the
   * decoded sequence. Insertions, a mid-word start and a basmala prefix add their decoded costs on top.
   */
  const posteriorCost = (frames: PosteriorFrames, m: Match, self: number): number => {
    const from = observedFrames[0].start;
    const to = observedFrames.at(-1)!.end + 1;
    const labels: string[] = [];
    const prefixEnd = m.basmala ? m.prefixEnd! : 0;
    for (let j = 0; j < prefixEnd; j++) labels.push(observed[j]);
    let edits = (m.basmala ? m.prefixCost! : 0) + (isWordStart(m.offset) ? 0 : MID_WORD_START);
    const token = (o: AlignmentOp | undefined) => (!o ? undefined : o.op === "ins" ? o.obs : o.ref);
    m.path.forEach((o, i) => {
      if (o.op === "ins") {
        // An inserted held sound beside the same sound only lengthens it; the collapse drops it, so don't charge it.
        const lengthens = isHeld(o.obs) && (labels.at(-1) === o.obs || token(m.path[i + 1]) === o.obs);
        if (!lengthens) edits += UNIT * observedConf[o.obsAt];
        labels.push(o.obs);
      } else labels.push(o.ref);
    });
    const nll = frames.nll(toIds(labels), from, to);
    if (!Number.isFinite(nll) || !Number.isFinite(self)) return m.cost;
    return Math.max(0, edits + ((nll - self) * UNIT) / NATS_PER_EDIT);
  };

  const ref = (i: number): WordRef => {
    const w = index.words[i];
    return { surah: w.surah, ayah: w.ayah, word: w.word };
  };

  const normCost = (m: Match) => m.cost / UNIT / observed.length;
  /** Fit takes the better of the two costs: posteriors can rescue a mangled decode, but CTC charges held-sound length in full. */
  const fitCost = (m: Match) => Math.min(m.cost, m.decodedCost ?? m.cost) / UNIT / observed.length;

  const firstWord = (m: Match) => index.wordAt(m.offset);
  const lastWord = (m: Match) => index.wordAt(m.offset + Math.max(0, m.length - 1));

  const toCandidate = (m: Match, confidence: number): PositionCandidate => {
    const end = ref(lastWord(m));
    return { word: end, span: { start: ref(firstWord(m)), end }, confidence, cost: normCost(m), per: m.per, ...(m.basmala ? { prefix: "basmala" as const } : {}) };
  };

  const priorBonus = (m: Match): number => {
    const first = index.words[firstWord(m)];
    const opening = first.word === 1 && (first.ayah === 1 || (first.surah === 1 && first.ayah === 2));
    const nextSurah = afterBasmala && opening && first.ayah === 1 && first.surah === afterBasmala.from.surah + 1 ? MAX_PRIOR_BONUS : 0;
    const openingBonus = (surahStart && opening ? MAX_PRIOR_BONUS : 0) + nextSurah;
    // Only once the span reaches "rabbi": "al-hamdu lillahi" alone is also how 6:1, 18:1, 34:1 and 35:1 open.
    if (fatihaNext && first.surah === 1 && first.word === 1 && first.ayah <= 2 && lastWord(m) >= fatihaRabbi) return FATIHA_BONUS;
    let bonus = openingBonus;
    if (priorRange) {
      const inside = priorRange.ranges.some(({ from, to }) => {
        for (let i = firstWord(m); i <= lastWord(m); i++) {
          const w = index.words[i];
          if (ayahCompare(w, from) >= 0 && ayahCompare(w, to) <= 0) return true;
        }
        return false;
      });
      if (inside) bonus = Math.max(bonus, MAX_PRIOR_BONUS * Math.min(1, Math.max(0, priorRange.weight)));
    }
    if (!prior) return bonus;
    let best = 0;
    for (let i = firstWord(m); i <= lastWord(m); i++) {
      const w = index.words[i];
      if (w.surah !== prior.surah) continue;
      const d = Math.abs(w.ayah - prior.ayah);
      best = Math.max(best, d === 0 ? 1 : d === 1 ? 0.5 : 0);
    }
    return Math.max(bonus, best * MAX_PRIOR_BONUS * Math.min(1, Math.max(0, prior.weight)));
  };

  /** Overlapping or touching ranges describe one location, e.g. shifted spans of the same ayah. */
  const sameLocation = (a: Match, b: Match) => a.offset <= b.offset + b.length && b.offset <= a.offset + a.length;

  /** Phoneme offset where the ayah containing `offset` starts. */
  const ayahOrigin = (offset: number): number => {
    let i = index.wordAt(Math.max(0, Math.min(offset, index.phonemes.length - 1)));
    const w = index.words[i];
    while (i > 0) {
      const prev = index.words[i - 1];
      if (prev.surah !== w.surah || prev.ayah !== w.ayah) break;
      i--;
    }
    return index.words[i].start;
  };
  /** Ayah start immediately before an ayah start, or -1 at the beginning of the mushaf. */
  const previousAyahOrigin = (origin: number): number => {
    const i = index.wordAt(origin);
    if (i <= 0) return -1;
    return ayahOrigin(index.words[i - 1].start);
  };

  /**
   * Heard phonemes from the first Quran onset. Phrases the streaming matcher already cut are dropped via
   * `onsetCut`; anything still leading is stripped with the same phrase alignment.
   */
  const onsetSpeech = (): { text: string; conf: number[] } => {
    let text = raw.slice(onsetCut);
    let conf = rawConf.slice(onsetCut);
    if (!phraseList.length) return { text, conf };
    for (let guard = 0; guard < 6 && text.length; guard++) {
      let cut = 0;
      for (const phrase of phraseList) {
        const n = phrase.p.length;
        if (n < 2 || text.length < 2) continue;
        const hi = Math.min(text.length, Math.ceil(n * 1.6) + 4);
        // Same alignment as `hear`: the phrase is the query, any reference offset may start it, and the cost cap is
        // PHRASE_MAX_COST. Only a match that begins at the onset is a leading phrase.
        const a = alignSubstring(text, 0, hi, phrase.p, () => true);
        if (!a || a.start !== 0 || a.end <= cut || a.cost / UNIT / n > PHRASE_MAX_COST) continue;
        const heardEnd = a.path.filter((o) => o.op === "match" && o.obsAt >= n - 2).length;
        if (heardEnd < 2) continue;
        cut = a.end;
      }
      if (!cut) break;
      text = text.slice(cut);
      conf = conf.slice(cut);
    }
    return { text, conf };
  };

  /**
   * Alignment of every onset phone against the contiguous ayahs ending at this candidate. The run walks back an ayah
   * at a time until it covers that speech, so a look-alike tail has to match the heard opening with its own
   * preceding ayah.
   */
  const contiguousRunCost = (m: Match, speech: string, conf: number[]): number => {
    const refEnd = Math.min(index.phonemes.length, m.offset + Math.max(m.length, 0));
    if (refEnd <= 0 || !speech) return Infinity;
    let lo = ayahOrigin(Math.min(m.offset, refEnd - 1));
    for (let guard = 0; refEnd - lo < speech.length && guard < 30; guard++) {
      const prev = previousAyahOrigin(lo);
      if (prev < 0 || prev >= lo) break;
      lo = prev;
    }
    const a = alignSubstring(index.phonemes, lo, refEnd, speech, isWordStart, {
      strictStart: true,
      ...(conf.length === speech.length ? { conf } : {}),
    });
    return a ? a.cost : Infinity;
  };

  /** Global search over the observed phonemes: the event it supports, the lock if any, and its trace. */
  const assess = () => {
    const frames = posteriors;
    const self = frames ? frames.nll(toIds([...observed]), observedFrames[0].start, observedFrames.at(-1)!.end + 1) : 0;
    const matches = frames ? locate().map((m) => ({ ...m, cost: posteriorCost(frames, m, self), decodedCost: m.cost })) : locate();
    const scored = matches.map((m) => ({ m, score: m.cost - priorBonus(m) })).sort((a, b) => a.score - b.score || a.m.offset - b.m.offset);
    const distinct: typeof scored = [];
    for (const c of scored) if (!distinct.some((k) => sameLocation(k.m, c.m))) distinct.push(c);
    const fitting = distinct.filter((c) => c.m.per <= maxPer && fitCost(c.m) <= maxCost);

    const n = observed.length;
    const required = UNIT * Math.max(minSeparation, separationScale / n);
    const decide = (pool: typeof fitting) => {
      // How far each candidate leads the best other fitting candidate; negative when it trails.
      const lead = (c: (typeof pool)[number]) => Math.min(...pool.filter((o) => o !== c).map((o) => o.score - c.score), Infinity);
      const top = pool[0];
      const separated = top !== undefined && lead(top) >= required;
      const terms = (c: (typeof pool)[number]) => ({
        fit: Math.max(0, 1 - c.m.per / maxPer) * Math.max(0, 1 - fitCost(c.m) / maxCost),
        separation: Math.min(1, Math.max(0, 0.5 + lead(c) / (2 * required))),
      });
      const confidence = (c: (typeof pool)[number]) => {
        const t = terms(c);
        return t.fit * t.separation;
      };
      const enough =
        n >= minPhonemes && top !== undefined && lastWord(top.m) - firstWord(top.m) + 1 >= minWords && confidence(top) >= minLockConfidence;
      const state: TrackerState = !top ? "lost" : !separated ? "ambiguous" : !enough ? "acquiring" : "locked";
      // Ambiguous lists every candidate within the margin of the best; otherwise only the best.
      const shown = !top ? [] : separated ? [top] : pool.filter((c) => c.score - top.score < required);
      const candidates = shown.map((c) => toCandidate(c.m, confidence(c)));
      const event: PositionEvent = {
        type: "position",
        v: EVENT_SCHEMA_VERSION,
        timeMs: lastFrame * (frameMs ?? 0),
        referenceVersion: index.version,
        expected: state === "locked" ? candidates[0].word : null,
        observed: candidates,
        state,
      };
      const trace = (from: TrackerState | null): TrackerTrace => ({
        shortlist: scored.map(({ m }) => ({
          span: toCandidate(m, 0).span,
          offset: m.offset,
          length: m.length,
          votes: m.votes,
          cost: normCost(m),
          per: m.per,
        })),
        transition: { from, to: state },
        path: scored[0]?.m.path ?? null,
        decision: top ? { ...terms(top), leadEdits: lead(top) / UNIT, requiredEdits: required / UNIT } : null,
      });
      return { event, lock: state === "locked" ? top.m : null, trace, top: top && n >= minPhonemes && confidence(top) > 0 ? top.m : null, state };
    };

    let decision = decide(fitting);
    // First lock only. A suffix window can prefer a look-alike whose own ayah matches the tail. Score each
    // candidate on the contiguous ayahs back to the speech onset, and drop the commit when that winner is beaten
    // by the same margin that would commit a lock. A winner that is not beaten is left exactly as ranked.
    // An opening that already starts on an ayah's first word is the clean case: the heard onset is that ayah, and
    // the candidate stays. The veto is for a tail match, whose own preceding ayah has to cover the onset.
    const ayahStart = decision.state === "locked" && index.words[firstWord(fitting[0].m)].word === 1;
    if (decision.state === "locked" && !everLocked && fitting.length > 1 && !ayahStart) {
      const speech = onsetSpeech();
      if (speech.text.length >= minPhonemes) {
        const costs = fitting.map((c) => contiguousRunCost(c.m, speech.text, speech.conf));
        const best = Math.min(...costs);
        if (costs[0] - best >= required) {
          const kept = fitting.filter((_, i) => costs[i] - best < required);
          if (kept.length) decision = decide(kept);
        }
      }
    }
    return { event: decision.event, lock: decision.lock, trace: decision.trace, top: decision.top };
  };

  const marked = (event: PositionEvent): PositionEvent => (arm ? { ...event, spanHold: true } : event);
  const emit = (event: PositionEvent, trace: (from: TrackerState | null) => TrackerTrace): PositionEvent => {
    const from = last?.state ?? null;
    const out = marked(event);
    last = out;
    return options.trace ? { ...out, trace: trace(from) } : out;
  };
  const bare = (event: PositionEvent) => {
    const { spanHold: _, timeMs: __, trace: ___, ...rest } = event;
    return JSON.stringify(rest);
  };
  const unchanged = (event: PositionEvent) => !!last && bare(last) === bare(event);

  const evaluate = (final: boolean): PositionEvent[] => {
    if (!observed) return [];
    const a = assessOutsidePhrases();
    if (a.lock && !everLocked && longestSpanAt(allSpans, lastWord(a.lock)) !== null) arm = true;
    if (unchanged(a.event) && !final) return [];
    if (a.lock && !final) lockedOn = a.lock;
    return [emit(a.event, a.trace)];
  };

  const quietTrace = (from: TrackerState | null, to: TrackerState, shortlist: TraceCandidate[] = []): TrackerTrace => ({
    shortlist,
    transition: { from, to },
    path: null,
    decision: null,
  });

  const observe = (ph: DecodedPhoneme) => {
    observedFrames.push({ start: ph.start, end: ph.end });
    observed += ph.p;
    observedConf.push(ph.conf);
    if (!everLocked) {
      raw += ph.p;
      for (let i = 0; i < ph.p.length; i++) rawConf.push(ph.conf);
    }
    lastFrame = Math.max(lastFrame, ph.end + 1);
    if (excursion) {
      excursion.phones += ph.p;
      if (excursion.phones.length > 480) {
        excursion.phones = excursion.phones.slice(-480);
        excursion.trimmed = true;
      }
    }
  };
  /** Keep only the latest `n` observed phonemes. */
  const keepLast = (n: number) => {
    const drop = observed.length - n;
    if (drop <= 0) return;
    observed = observed.slice(drop);
    observedConf = observedConf.slice(drop);
    observedFrames.splice(0, drop);
  };

  /**
   * Record one phoneme for phrase detection, with the local search's state before it and its cost; returns how many
   * of the latest phonemes form a whole phrase that ends here, and which.
   */
  const hear = (p: string, state: unknown, cost: number): { name: string; length: number } | null => {
    if (!phraseList.length) return null;
    heard += p;
    heardCost.push(cost);
    heardState.push(state);
    if (heard.length > HEARD) {
      heard = heard.slice(1);
      heardCost = heardCost.slice(1);
      heardState = heardState.slice(1);
    }
    for (const phrase of phraseList) {
      const n = phrase.p.length;
      const lo = Math.max(0, heard.length - Math.ceil(n * 1.6) - 4);
      const a = alignSubstring(heard, lo, heard.length, phrase.p, () => true);
      if (!a || a.end < heard.length - 1 || a.cost / UNIT / n > PHRASE_MAX_COST) continue;
      // The phrase must be finished: its last two phonemes heard, not just cheap to leave out.
      const heardEnd = a.path.filter((o) => o.op === "match" && o.obsAt >= n - 2).length;
      if (heardEnd < 2) continue;
      return { name: phrase.name, length: heard.length - a.start };
    }
    return null;
  };
  /** The latest phonemes are a phrase still being said: hold off locking, a prefix like "a'udhu billahi" is Quran text too. */
  const phraseInProgress = (): boolean => {
    for (const phrase of phraseList) {
      const n = phrase.p.length;
      for (let k = Math.min(n - 2, heard.length); k >= Math.max(PHRASE_PREFIX, PHRASE_PREFIX_SHARE * n); k -= 2) {
        const a = alignSubstring(phrase.p, 0, n - 2, heard.slice(-k), (o) => o === 0, { strictStart: true });
        if (a && a.cost / UNIT / k <= PHRASE_MAX_COST / 2) return true;
      }
    }
    return false;
  };
  /** The window's latest phonemes restate its start: a restart still being said, not yet a repeat to collapse. */
  const repeatInProgress = (): boolean => {
    for (let k = Math.min(MAX_REPEAT, observed.length >> 1); k >= RESTART_PREFIX; k--) {
      const a = alignSubstring(observed, 0, observed.length - k, observed.slice(-k), (o) => o === 0, { strictStart: true });
      if (a && a.cost / UNIT / k <= REPEAT_MAX_COST) return true;
    }
    return false;
  };
  /** Assess, but never lock while a phrase or a repeat is in progress. */
  const assessOutsidePhrases = () => {
    const a = assess();
    if (!a.lock || !phraseList.length || (!phraseInProgress() && !repeatInProgress())) return a;
    const event: PositionEvent = { ...a.event, state: "acquiring", expected: null };
    return { ...a, event, lock: null, top: null, trace: (from: TrackerState | null) => ({ ...a.trace(from), transition: { from, to: "acquiring" as const } }) };
  };
  /**
   * An immediately repeated block at the end of the window ("Alif Lam Mim" twice) is a repetition, not two ayahs: keep
   * one copy, so global search doesn't prefer passages that happen to continue like the repeat.
   */
  const collapseRepeat = () => {
    for (let n = Math.min(MAX_REPEAT, observed.length >> 1); n >= MIN_REPEAT; n--) {
      const tail = observed.slice(-n);
      const before = observed.slice(-2 * n, -n);
      const a = alignSubstring(before, 0, n, tail, (o) => o === 0, { strictStart: true });
      if (a && a.end >= n - 1 && a.cost / UNIT / n <= REPEAT_MAX_COST) {
        const at = observed.length - 2 * n;
        const collapsed = observed.slice(0, at) + tail;
        const conf = [...observedConf.slice(0, at), ...observedConf.slice(-n)];
        // Quran text can itself contain a near-repeat (من أجر إن أجري). Removing it would manufacture evidence
        // for a different passage. Keep the whole phrase whenever it fits the reference at least as well.
        const cost = (text: string, confidence: number[]) => Math.min(...fuzzyMatches(text, confidence).map((m) => m.cost));
        if (cost(observed, observedConf) <= cost(collapsed, conf)) continue;
        observed = collapsed;
        observedConf = [...observedConf.slice(0, at), ...observedConf.slice(-n)];
        observedFrames.splice(at, n);
        return;
      }
    }
  };
  const forget = () => {
    heard = "";
    heardCost = [];
    heardState = [];
  };

  /** Drop the latest `n` observed phonemes. */
  const dropLast = (n: number) => {
    const keep = Math.max(0, observed.length - n);
    observed = observed.slice(0, keep);
    observedConf = observedConf.slice(0, keep);
    observedFrames.length = keep;
  };

  const startLocal = (m: Match) => {
    local = createLocalSearch(index, { offset: m.offset + m.length, origin: m.offset, edits: Math.round(m.per * m.length) });
    weak = false;
    pending = null;
    reacquiring = false;
    holding = null;
    if (everLocked) arm = false;
    everLocked = true;
    raw = "";
    rawConf = [];
    onsetCut = 0;
    surahStart = false;
    basmalaHeard = false;
    fatihaNext = false;
    takbirHeard = false;
    afterBasmala = null;
    afterPhrase = false;
    twin = null;
    keepLast(0);
  };

  /** Memorization holds the expected bookmark where a jump left it; free mode follows the reciter. */
  const expectedFor = (word: WordRef): WordRef => frozen ?? word;
  /** Neither a jump nor a passage change happened, so the bookmark stays where local tracking last trusted it. */
  const expectedWhileLost = () => frozen ?? steady;

  const near = (m: Match, offset: number) => {
    const a = index.words[lastWord(m)];
    const b = index.words[index.wordAt(Math.max(0, offset - 1))];
    return a.surah === b.surah && Math.abs(a.ayah - b.ayah) <= NEAR_AYAHS;
  };
  /** Same passage the local search would keep: not a relocation. */
  const nearWord = (a: WordRef, b: WordRef) => a.surah === b.surah && Math.abs(a.ayah - b.ayah) <= NEAR_AYAHS;

  const spokenStep = (wi: number, dir: -1 | 1): number => {
    if (wi < 0) return -1;
    for (let k = wi + dir; k >= 0 && k < index.words.length; k += dir) {
      if (index.words[k].end > index.words[k].start) return k;
    }
    return -1;
  };
  /** Heard words of a destination match: reference word, and the phonemes the alignment heard for it. */
  const destinationWords = (m: Match): DestinationWord[] => {
    const groups: { wi: number; heard: string }[] = [];
    let cur: { wi: number; heard: string } | null = null;
    let lead = "";
    for (const op of m.path) {
      if (op.op === "ins") {
        if (cur) cur.heard += op.obs;
        else lead += op.obs;
        continue;
      }
      const wi = index.wordAt(Math.max(0, Math.min(index.phonemes.length - 1, m.offset + op.refAt)));
      if (!cur || cur.wi !== wi) {
        cur = { wi, heard: lead };
        lead = "";
        groups.push(cur);
      }
      if (op.op === "match") cur.heard += op.ref;
      else if (op.op === "sub") cur.heard += op.obs;
    }
    if (lead) {
      if (groups.length) groups[0].heard = lead + groups[0].heard;
      else groups.push({ wi: -1, heard: lead });
    }
    return groups.filter((g) => g.heard.length > 0 && g.wi >= 0).map((g) => ({
      destination: index.phonemes.slice(index.words[g.wi].start, index.words[g.wi].end),
      heard: g.heard,
    }));
  };
  const anchorWordsFrom = (wi: number, n: number): string[] => {
    const out: string[] = [];
    for (let k = wi; k < index.words.length && out.length < n; k++) {
      const w = index.words[k];
      if (w.end > w.start) out.push(index.phonemes.slice(w.start, w.end));
    }
    return out;
  };
  /**
   * A distant lock relocates only when enough heard words fit it better than the anchor plus one inserted word.
   * The anchor start is whichever of the cursor and its neighbours that hypothesis explains cheapest: the cursor
   * sits on the word before a shared word as often as on the shared word itself.
   */
  const relocationCommits = (m: Match, anchorWi: number): boolean => {
    const heard = destinationWords(m);
    if (heard.length < minWords) return false;
    const seeds = [anchorWi, spokenStep(anchorWi, -1), spokenStep(anchorWi, 1), spokenStep(spokenStep(anchorWi, 1), 1)].filter((i) => i >= 0);
    let best: { count: number; total: number } | null = null;
    for (const seed of seeds) {
      const fit = relocationSupport(anchorWordsFrom(seed, heard.length + 1), heard, minSeparation, separationScale);
      if (fit && (!best || fit.total < best.total)) best = fit;
    }
    return !!best && best.count >= minWords;
  };

  /**
   * The anchor stays locked and relocation search continues. One event tells the checker to downgrade issues open
   * now; `last` stays the plain lock, so later re-emits of the anchor compare unchanged.
   */
  const beginHold = (): PositionEvent => {
    if (twin) {
      twinSettled.add(spanId(twin.span));
      twin = null;
    }
    if (memorization) frozen ??= steady;
    const anchor = last!;
    holding = anchor;
    local = null;
    weak = false;
    pending = null;
    reacquiring = true;
    const event: PositionEvent = { ...anchor, timeMs: lastFrame * (frameMs ?? 0), hold: "start" };
    return options.trace ? { ...event, trace: quietTrace(anchor.state, "locked") } : event;
  };

  const lostEvent = (timeMs: number, trace: (from: TrackerState | null) => TrackerTrace): PositionEvent =>
    emit(
      { type: "position", v: EVENT_SCHEMA_VERSION, timeMs, referenceVersion: index.version, expected: expectedWhileLost(), observed: [], state: "lost" },
      trace,
    );

  /** Next word that has its own phonemes. Assimilated words are skipped. */
  const spoken = (wi: number) => {
    let k = wi;
    while (k < index.words.length && index.words[k].end === index.words[k].start) k++;
    return k;
  };

  const ayahStartOf = (wi: number) => {
    let k = wi;
    const w = index.words[wi];
    while (k > 0 && index.words[k - 1].surah === w.surah && index.words[k - 1].ayah === w.ayah) k--;
    return k;
  };
  const ayahEndOf = (wi: number) => {
    let k = wi + 1;
    const w = index.words[wi];
    while (k < index.words.length && index.words[k].surah === w.surah && index.words[k].ayah === w.ayah) k++;
    return k;
  };
  const phonesBetween = (from: number, to: number) => {
    if (from < 0 || to <= from || to > index.words.length) return "";
    return index.phonemes.slice(index.words[from].start, index.words[to - 1].end);
  };
  const skeleton = (from: number, to: number) => {
    let s = "";
    for (let i = from; i < to; i++) s += `${letterSkeleton(index.wordText?.[i] ?? "")}\0`;
    return s;
  };
  /**
   * A jump off the in-order ayah onto a parallel copy. Closed when a later lock is that copy's in-order
   * continuation and the skipped-over text fits the excursion's phones no worse than the copy does.
   * Exact copies tie. A one-word variant pays that word's edits, so the heard copy is the cheaper one.
   */
  const beginExcursion = (fromWi: number, m: Match) => {
    const destWi = firstWord(m);
    if (fromWi < 0 || destWi < 0) return;
    if (excursion && destWi >= excursion.dest && destWi < excursion.dest + excursion.length) return;
    const at = lastWord(m);
    const span = longestSpanAt(allSpans, destWi);
    if (span) {
      const dest = span.starts.find((st) => destWi >= st && destWi < st + span.length);
      const home = span.starts.find((st) => fromWi >= st && fromWi < st + span.length);
      if (dest !== undefined && home !== undefined && dest !== home) {
        excursion = { home, dest, length: span.length, entry: Math.max(0, Math.min(span.length - 1, at - dest)), phones: "", trimmed: false };
        return;
      }
    }
    const homeLo = fromWi < index.words.length ? ayahEndOf(fromWi) : -1;
    const destLo = destWi < index.words.length ? ayahStartOf(destWi) : -1;
    if (homeLo >= 0 && homeLo < index.words.length && destLo >= 0 && destLo !== homeLo) {
      const homeHi = ayahEndOf(homeLo);
      const destHi = ayahEndOf(destLo);
      const length = homeHi - homeLo;
      if (length > 0 && destHi - destLo === length && skeleton(homeLo, homeHi) === skeleton(destLo, destHi)) {
        excursion = { home: homeLo, dest: destLo, length, entry: Math.max(0, Math.min(length - 1, at - destLo)), phones: "", trimmed: false };
        return;
      }
    }
    excursion = null;
  };
  /** Retro locked event, or null when this lock does not close the excursion. */
  const finishExcursion = (destWi: number): PositionEvent | null => {
    const ex = excursion;
    if (!ex || ex.trimmed || destWi < 0) return null;
    if (destWi >= ex.dest && destWi < ex.dest + ex.length) return null;
    const cont = spoken(ex.home + ex.length);
    // Landing here is the commit: the search already preferred this continuation by the commit margin.
    if (destWi < cont || destWi >= cont + 4) return null;
    const from = ex.entry;
    const homeText = phonesBetween(ex.home + from, ex.home + ex.length);
    const destText = phonesBetween(ex.dest + from, ex.dest + ex.length);
    const homeCost = anchorFitCost(homeText, ex.phones);
    const destCost = anchorFitCost(destText, ex.phones);
    // Exact copies tie. A one-word variant pays that word, so the copy that was heard stays cheaper.
    if (homeCost === null || destCost === null || homeCost > destCost) return null;
    const end = ex.home + ex.length - 1;
    const event: PositionEvent = {
      type: "position",
      v: EVENT_SCHEMA_VERSION,
      timeMs: lastFrame * (frameMs ?? 0),
      referenceVersion: index.version,
      expected: expectedFor(ref(ex.home)),
      observed: [{ word: ref(end), span: { start: ref(ex.home), end: ref(end) }, confidence: 1, cost: 0, per: 0 }],
      state: "locked",
      relocated: { from: ref(ex.dest), to: ref(ex.home) },
    };
    return emit(event, (fromState) => quietTrace(fromState, "locked"));
  };
  const takeJump = (fromWi: number, m: Match): PositionEvent | null => {
    const closing = finishExcursion(firstWord(m));
    if (closing) {
      excursion = null;
      return closing;
    }
    beginExcursion(fromWi, m);
    return null;
  };

  /**
   * While the lock sits in a slot-differing span, score heard phones against each member's next word. The current
   * copy wins when its own continuation fits. Another copy wins only when the heard word fits that continuation and
   * beats every other member by the commit margin. Held slot accusations then follow that copy.
   */
  const considerTwin = (p: string, pos: LocalPosition): PositionEvent | null => {
    const wi = index.wordAt(Math.max(0, pos.offset - 1));
    const outside = longestSpanAt(allSpans, wi) === null;
    if (!twin && !arm) return null;
    if (!twin) {
      if (outside) {
        arm = false;
        return null;
      }
      const span = longestSpanAt(allSpans, wi);
      if (!span || twinSettled.has(spanId(span))) return null;
      const memberStart = span.starts.find((st) => wi >= st && wi < st + span.length);
      if (memberStart === undefined) return null;
      const endWi = memberStart + span.length;
      if (endWi >= index.words.length) return null;
      twin = { span, memberStart, endPhone: index.words[endWi].start, buf: "" };
    }
    // A clean phone still inside the span belongs to the shared text, not to the continuation.
    if (pos.offset <= twin.endPhone && pos.stepCost < 1) twin.buf = "";
    else {
      twin.buf += p;
      if (twin.buf.length > 96) twin.buf = twin.buf.slice(-96);
    }
    const span = twin.span;
    const candidates = span.starts.map((st) => {
      const w = spoken(st + span.length);
      const word = index.words[w];
      return { id: st, phonemes: word ? index.phonemes.slice(word.start, word.end) : "" };
    });
    const pick = continuationDecision(twin.buf, candidates, twin.memberStart, (n) => commitMargin(n, minSeparation, separationScale), maxCost);
    // The continuation settled the copy. A later global return must not rebind it again.
    if (pick) excursion = null;
    // Past the first continuation word with no unique other copy: keep the locked one and let the window move on.
    // The continuation word itself sits outside the span; the open span still gets to judge it.
    if (!pick && wi < twin.memberStart + twin.span.length + 4) return null;
    if (!pick) {
      const fromWi = twin.memberStart;
      twinSettled.add(spanId(twin.span));
      twin = null;
      if (outside) arm = false;
      const word = ref(wi);
      emittedSinceLock = true;
      return emit({
        type: "position",
        v: EVENT_SCHEMA_VERSION,
        timeMs: lastFrame * (frameMs ?? 0),
        referenceVersion: index.version,
        expected: expectedFor(word),
        observed: [{ word, span: { start: ref(index.wordAt(pos.origin)), end: word }, confidence: 1, cost: pos.recentCost, per: 0 }],
        state: "locked",
        relocated: { from: ref(fromWi), to: ref(fromWi) },
      }, (from) => quietTrace(from, "locked"));
    }
    const fromWi = twin.memberStart;
    const toWi = pick.id;
    twinSettled.add(spanId(span));
    twin = null;
    if (outside) arm = false;
    const at = toWi === fromWi ? wi : spoken(toWi + span.length);
    const origin = toWi === fromWi ? index.wordAt(pos.origin) : toWi;
    const word = ref(at);
    const event: PositionEvent = {
      type: "position",
      v: EVENT_SCHEMA_VERSION,
      timeMs: lastFrame * (frameMs ?? 0),
      referenceVersion: index.version,
      expected: expectedFor(word),
      observed: [{
        word,
        span: { start: ref(origin), end: word },
        confidence: 1,
        cost: pos.recentCost,
        per: 0,
      }],
      state: "locked",
      relocated: { from: ref(fromWi), to: ref(toWi) },
    };
    emittedSinceLock = true;
    if (toWi !== fromWi) {
      const dest = index.words[at];
      if (!dest) return null;
      stepCosts.length = 0;
      startLocal({ offset: dest.end, length: 0, cost: 0, per: 0, votes: 0, path: [] });
    }
    return emit(event, (from) => quietTrace(from, "locked"));
  };

  /** The local search's position as an event, or null when its word and run start haven't changed. */
  const localEvent = (shortlist: TraceCandidate[]): PositionEvent | null => {
    const pos = local!.position();
    const word = ref(index.wordAt(pos.offset - 1));
    const length = pos.offset - pos.origin;
    const per = length ? pos.edits / length : 0;
    const confidence = Math.max(0, 1 - per / maxPer) * Math.max(0, 1 - pos.recentCost / maxCost);
    const candidate: PositionCandidate = { word, span: { start: ref(index.wordAt(pos.origin)), end: word }, confidence, cost: pos.recentCost, per };
    const prev = last!.observed[0];
    if (last!.state === "locked" && JSON.stringify([prev.word, prev.span.start]) === JSON.stringify([word, candidate.span.start])) return null;
    const newRun = JSON.stringify(prev.span.start) !== JSON.stringify(candidate.span.start);
    // A weak local search jumps around chasing evidence that may belong elsewhere; only its current run may advance.
    if (weak && newRun) return null;
    // A skip or restart shows once its landing fits: at a jump's onset, local skips chase audio that belongs elsewhere.
    if (newRun && stepCosts.length >= CONFIRM && stepCosts.slice(-CONFIRM).reduce((a, b) => a + b, 0) / CONFIRM > CONFIRM_COST) return null;
    const event: PositionEvent = {
      type: "position",
      v: EVENT_SCHEMA_VERSION,
      timeMs: lastFrame * (frameMs ?? 0),
      referenceVersion: index.version,
      expected: expectedFor(word),
      observed: [candidate],
      state: "locked",
    };
    emittedSinceLock = true;
    return emit(event, (from) => quietTrace(from, "locked", shortlist));
  };

  /**
   * Global search over the latest phonemes, until it locks somewhere. While `holding`, the anchor stays locked:
   * non-locked assessments are not emitted. A lock off the anchor emits `lost` and then the new place; a lock on
   * the same passage emits only that lock, so tokens heard in the gap stay in the checker's window.
   */
  const reacquire = (): PositionEvent[] => {
    keepLast(WINDOW);
    if (phraseList.length) collapseRepeat();
    const a = assessOutsidePhrases();
    // After a basmala ends a surah, the next surah is the expected continuation: its opening on top is enough.
    // Al-Fatiha excepted: in prayer any surah follows it.
    if (!a.lock && afterBasmala && afterBasmala.from.surah !== 1 && a.top) {
      const first = index.words[firstWord(a.top)];
      if (first.surah === afterBasmala.from.surah + 1 && first.ayah === 1) {
        a.lock = a.top;
        const c = toCandidate(a.top, a.event.observed[0]?.confidence ?? 0);
        a.event = { ...a.event, state: "locked", observed: [c], expected: c.word };
      }
    }
    if (a.lock) {
      const anchorWord = holding?.observed[0]?.word;
      const dest = a.event.observed[0]?.word;
      const release = !!(holding && anchorWord && dest && !nearWord(anchorWord, dest));
      const anchorWi = anchorWord ? index.words.findIndex((w) => w.surah === anchorWord.surah && w.ayah === anchorWord.ayah && w.word === anchorWord.word) : -1;
      // A lookalike of the anchor plus one inserted word has not earned the move. Keep searching.
      if (!(release && anchorWi >= 0 && !relocationCommits(a.lock, anchorWi))) {
        if (memorization && afterBasmala) {
          const first = index.words[firstWord(a.lock)];
          if (!(first.surah === afterBasmala.from.surah + 1 && first.ayah === 1)) frozen ??= steady;
        }
        afterBasmala = null;
        const released = release ? [lostEvent(a.event.timeMs, (from) => quietTrace(from, "lost"))] : [];
        const fromWi = anchorWi >= 0 ? anchorWi : -1;
        const rebind = fromWi >= 0 ? takeJump(fromWi, a.lock) : null;
        startLocal(a.lock);
        return [...released, ...(rebind ? [rebind] : []), emit({ ...a.event, expected: expectedFor(a.event.observed[0].word) }, a.trace)];
      }
    }
    // Between surahs the last position stays on screen, unless nothing locks for a while.
    if (afterBasmala && ++afterBasmala.held < WINDOW) return [];
    // Junk that fits nowhere would otherwise favour passages with text before the recited phrase to absorb it.
    if (a.event.state === "lost") keepLast(LOOKBACK);
    if (holding) {
      const event: PositionEvent = { ...holding, timeMs: lastFrame * (frameMs ?? 0) };
      return unchanged(event) ? [] : [emit(event, (from) => quietTrace(from, "locked"))];
    }
    const event = { ...a.event, expected: expectedWhileLost() };
    return unchanged(event) ? [] : [emit(event, a.trace)];
  };

  /**
   * Phoneme by phoneme after the first lock. Local tracking emits one event per change of word or run start. When its
   * fit degrades, global search over the latest phonemes runs alongside: a distant global lock that holds for
   * `jumpPhonemes` commits a jump, and nothing fitting anywhere for long enough holds the anchor while the search
   * continues. `lost` is emitted when that search locks off the anchor, or when the session ends still there.
   */
  const follow = (evidence: PhonemeEvidence): EngineEvent[] => {
    const events: EngineEvent[] = [];
    for (const ph of evidence.phonemes) {
      observe(ph);
      if (!local) {
        const heardPhrase = hear(ph.p, null, NaN);
        if (heardPhrase) {
          forget();
          dropLast(heardPhrase.length);
          if (heardPhrase.name === "basmala") surahStart = true;
          if (heardPhrase.name === "takbir") fatihaNext = true;
          continue;
        }
        events.push(...reacquire());
        continue;
      }
      const before = local.save();
      local.step(ph.p);
      stepCosts.push(local.position().stepCost);
      if (stepCosts.length > CONFIRM) stepCosts.shift();
      const heardPhrase = hear(ph.p, before, local.position().stepCost);
      if (heardPhrase) {
        const from = heardCost.length - heardPhrase.length;
        const cost = heardCost.slice(from).reduce((a, b) => a + b, 0) / heardPhrase.length;
        if (cost >= PHRASE_LOCAL_COST) {
          // Not Quran text where we are: rewind the local search to before the phrase, as if it was never heard.
          if (heardPhrase.name === "basmala") {
            // A surah starts, not necessarily the next one: hand over to global search, favouring openings.
            afterBasmala = { from: last!.observed[0].word, held: 0 };
            if (memorization) steady = last!.expected;
            local = null;
            reacquiring = true;
            surahStart = true;
            forget();
            keepLast(0);
            continue;
          }
          local.restore(heardState[from]);
          if (heardPhrase.name === "takbir") fatihaNext = true;
          afterPhrase = true;
          forget();
          // The phrase is a boundary: global search restarts after it.
          keepLast(0);
          const e = localEvent([]);
          if (e) events.push(e);
          continue;
        }
      }
      const pos = local.position();
      const relocated = considerTwin(ph.p, pos);
      if (relocated) {
        events.push(relocated);
        continue;
      }
      if (!weak && pos.recentCost > WEAK_COST) {
        weak = true;
        steady = expectedHistory[0] ?? last!.expected;
      } else if (weak && pos.recentCost <= WEAK_COST) {
        weak = false;
        pending = null;
        afterPhrase = false;
      }
      let shortlist: TraceCandidate[] = [];
      if (!weak) keepLast(LOOKBACK);
      else {
        keepLast(WINDOW);
        const a = assessOutsidePhrases();
        if (options.trace) shortlist = a.trace(null).shortlist;
        if (a.lock && !near(a.lock, pos.offset)) {
          pending = { m: a.lock, held: pending && sameLocation(pending.m, a.lock) ? pending.held + 1 : 1 };
        } else pending = null;
        if (pending && pending.held >= (afterPhrase ? 1 : jumpPhonemes) && relocationCommits(pending.m, index.wordAt(Math.max(0, pos.offset - 1)))) {
          if (memorization) frozen ??= steady;
          const fromWi = index.wordAt(Math.max(0, pos.offset - 1));
          const rebind = takeJump(fromWi, pending.m);
          if (rebind) events.push(rebind);
          startLocal(pending.m);
        } else if (!pending && a.event.state === "lost" && pos.recentCost > maxCost && observed.length >= LOST_AFTER) {
          // Nothing has committed elsewhere. Keep this anchor until the search locks off it, or the session ends.
          events.push(beginHold());
          continue;
        }
      }
      const e = localEvent(shortlist);
      if (e) events.push(e);
      expectedHistory.push(last!.expected);
      if (expectedHistory.length > WEAK_LAG) expectedHistory.shift();
    }
    return events;
  };

  /**
   * Before a new chunk: drop all but `MAX_HISTORY` phonemes of earlier chunks, or all but `LOOKBACK` after `lost`, and
   * the posterior rows before what's kept. Done here rather than after evaluating, so `end` sees what the last push saw.
   */
  const trimHistory = () => {
    if (phraseCut) {
      if (!everLocked) onsetCut += phraseCut;
      keepLast(observed.length - phraseCut);
      phraseCut = 0;
    }
    if (basmalaHeard) surahStart = true;
    if (takbirHeard) fatihaNext = true;
    const keep = last?.state === "lost" ? LOOKBACK : MAX_HISTORY;
    if (observed.length <= keep) return;
    keepLast(keep);
    if (observedFrames.length) posteriors?.dropBefore(observedFrames[0].start);
  };

  /**
   * Leading junk (a takbir, a cough) costs every candidate the same insertions, which can hold a stream in
   * `acquiring` long after the recitation itself is clear. So when the history doesn't lock, try its latest phonemes.
   */
  const evaluateSuffix = (fresh: number): PositionEvent[] => {
    const keep = Math.max(WINDOW, fresh);
    if (observed.length <= keep) return [];
    const saved = { observed, observedConf, frames: [...observedFrames] };
    keepLast(keep);
    const a = assessOutsidePhrases();
    if (a.lock) {
      lockedOn = a.lock;
      if (!everLocked && longestSpanAt(allSpans, lastWord(a.lock)) !== null) arm = true;
      return [emit(a.event, a.trace)];
    }
    observed = saved.observed;
    observedConf = saved.observedConf;
    observedFrames.splice(0, observedFrames.length, ...saved.frames);
    return [];
  };

  /** One acquisition step: a push's phonemes, or one `acquireMs` cell's, searched globally. */
  const acquire = (phonemes: DecodedPhoneme[]): EngineEvent[] => {
    trimHistory();
    for (const ph of phonemes) {
      observe(ph);
      const heardPhrase = hear(ph.p, null, NaN);
      if (heardPhrase) {
        forget();
        phraseCut = observed.length;
        basmalaHeard ||= heardPhrase.name === "basmala";
        takbirHeard ||= heardPhrase.name === "takbir";
      }
    }
    if (pushes++ > 0 && phraseList.length) collapseRepeat();
    const events: EngineEvent[] = evaluate(false);
    if (!lockedOn) events.push(...evaluateSuffix(phonemes.length));
    if (lockedOn) {
      lockEvent = events.at(-1) as PositionEvent;
      startLocal(lockedOn);
      posteriors = null;
    }
    return events;
  };

  const endLocal = (): PositionEvent[] => {
    const timeMs = lastFrame * (frameMs ?? 0);
    // The search never committed. The anchor was held; give it up now.
    if (holding) {
      holding = null;
      return [lostEvent(timeMs, (from) => quietTrace(from, "lost"))];
    }
    // Nothing followed the lock: its event, with the global search's trace, is still the answer.
    if (local && !emittedSinceLock) {
      const e = { ...lockEvent!, timeMs };
      return [e.trace ? { ...e, trace: { ...e.trace, transition: { from: "locked" as const, to: "locked" as const } } } : e];
    }
    const final = { ...last!, timeMs };
    return [options.trace ? { ...final, trace: quietTrace(final.state, final.state) } : final];
  };

  /** Phonemes of the current acquisition cell, held until a phoneme from a later cell shows the cell is complete. */
  let cell: DecodedPhoneme[] = [];
  let cellAt = 0;

  return {
    push(evidence) {
      validate(evidence);
      frameMs = evidence.frameMs;
      if (evidence.posteriors) withPosteriors = true;
      else decodedOnly = true;
      if (local || reacquiring) return follow(evidence);
      if (evidence.posteriors) (posteriors ??= new PosteriorFrames(index.tokens.size)).append(evidence.posteriors);
      if (!options.acquireMs) return acquire(evidence.phonemes);
      const events: EngineEvent[] = [];
      const phs = evidence.phonemes;
      for (let i = 0; i < phs.length; i++) {
        const at = Math.floor((phs[i].start * frameMs) / options.acquireMs);
        while (at > cellAt && !local) {
          events.push(...acquire(cell));
          cell = [];
          cellAt++;
        }
        if (local) return [...events, ...follow({ ...evidence, phonemes: phs.slice(i), posteriors: undefined })];
        cell.push(phs[i]);
      }
      const complete = evidence.decodedTo === undefined ? 0 : Math.floor((evidence.decodedTo * frameMs) / options.acquireMs);
      while (complete > cellAt && !local) {
        events.push(...acquire(cell));
        cell = [];
        cellAt++;
      }
      return events;
    },
    end() {
      if (!local && !reacquiring) {
        const events = cell.length ? acquire(cell) : [];
        cell = [];
        return [...events, ...(local ? endLocal() : evaluate(true))];
      }
      return endLocal();
    },
  };
}
