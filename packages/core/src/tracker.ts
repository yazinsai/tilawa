import type { AlignmentOp } from "./align.js";
import { PosteriorFrames } from "./ctc.js";
import { MID_WORD_START, UNIT, alignSubstring, collapseHeld, isHeld } from "./fuzzy.js";
import { EVENT_SCHEMA_VERSION } from "./events.js";
import type { EngineEvent, PositionCandidate, PositionEvent, TraceCandidate, TrackerState, TrackerTrace, WordRef } from "./events.js";
import { EvidenceError } from "./evidence.js";
import type { DecodedPhoneme, PhonemeEvidence } from "./evidence.js";
import { createLocalSearch } from "./local.js";
import type { LocalSearch } from "./local.js";
import type { PhraseAsset, ReferenceIndex } from "./reference.js";

export interface TrackerOptions {
  mode?: "free" | "memorization";
  /** Grams occurring more often than this carry too little location information to vote. */
  maxGramOccurrences?: number;
  maxCandidates?: number;
  /** Fit: a candidate fits only with PER at most this. Default 0.35. */
  maxPer?: number;
  /** Fit: and weighted cost per observed phoneme at most this. Default 0.35. */
  maxCost?: number;
  /** Evidence: `locked` needs at least this many observed phonemes. Default 10. */
  minPhonemes?: number;
  /** Evidence: and a span of at least this many words. Default 2. */
  minWords?: number;
  /**
   * Separation: `locked` needs the best distinct candidate to trail by `max(minSeparation, separationScale / n)`
   * plain edits for `n` observed phonemes, so short clips need a wider margin. Defaults 1 and 30.
   */
  minSeparation?: number;
  separationScale?: number;
  /** `locked` also needs the top candidate's confidence at least this; a unique but poor fit stays `acquiring`. Default 0.08. */
  minLockConfidence?: number;
  /** Bonus for the intended or filename ayah (full) and its neighbours (half). Bounded so it can't override fit or separation. */
  prior?: { surah: number; ayah: number; weight: number };
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
/** While weak: `lost` once the window (lookback included) holds this many phonemes, nothing fits globally and local cost exceeds `maxCost`. */
const LOST_AFTER = 24;
/** A global lock within this many ayahs of the local bookmark is the local search's to follow, not a jump. */
const NEAR_AYAHS = 2;
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

export function createTracker(index: ReferenceIndex, options: TrackerOptions = {}): Tracker {
  const maxGramOccurrences = options.maxGramOccurrences ?? 5000;
  const maxCandidates = options.maxCandidates ?? 64;
  const maxPer = options.maxPer ?? 0.35;
  const maxCost = options.maxCost ?? 0.35;
  const minPhonemes = options.minPhonemes ?? 10;
  const minWords = options.minWords ?? 2;
  const minSeparation = options.minSeparation ?? 1;
  const separationScale = options.separationScale ?? 30;
  const minLockConfidence = options.minLockConfidence ?? 0.08;
  const prior = options.prior;
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
  /** After `lost`: global search over a rolling window, one phoneme at a time, until it locks again. */
  let reacquiring = false;
  /** Memorization mode: the expected bookmark held in place after the reciter jumped away from it. */
  let frozen: WordRef | null = null;
  /** The expected bookmark when local fit last turned weak. */
  let steady: WordRef | null = null;

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
    const afterBasmala = surahStart && opening ? MAX_PRIOR_BONUS : 0;
    // Only once the span reaches "rabbi": "al-hamdu lillahi" alone is also how 6:1, 18:1, 34:1 and 35:1 open.
    if (fatihaNext && first.surah === 1 && first.word === 1 && first.ayah <= 2 && lastWord(m) >= fatihaRabbi) return FATIHA_BONUS;
    if (!prior) return afterBasmala;
    let best = 0;
    for (let i = firstWord(m); i <= lastWord(m); i++) {
      const w = index.words[i];
      if (w.surah !== prior.surah) continue;
      const d = Math.abs(w.ayah - prior.ayah);
      best = Math.max(best, d === 0 ? 1 : d === 1 ? 0.5 : 0);
    }
    return Math.max(afterBasmala, best * MAX_PRIOR_BONUS * Math.min(1, Math.max(0, prior.weight)));
  };

  /** Overlapping or touching ranges describe one location, e.g. shifted spans of the same ayah. */
  const sameLocation = (a: Match, b: Match) => a.offset <= b.offset + b.length && b.offset <= a.offset + a.length;

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
    // How far each candidate leads the best other fitting candidate; negative when it trails.
    const lead = (c: (typeof fitting)[number]) => Math.min(...fitting.filter((o) => o !== c).map((o) => o.score - c.score), Infinity);
    const top = fitting[0];
    const separated = top !== undefined && lead(top) >= required;
    const terms = (c: (typeof fitting)[number]) => ({
      fit: Math.max(0, 1 - c.m.per / maxPer) * Math.max(0, 1 - fitCost(c.m) / maxCost),
      separation: Math.min(1, Math.max(0, 0.5 + lead(c) / (2 * required))),
    });
    const confidence = (c: (typeof fitting)[number]) => {
      const t = terms(c);
      return t.fit * t.separation;
    };
    const enough =
      n >= minPhonemes && top !== undefined && lastWord(top.m) - firstWord(top.m) + 1 >= minWords && confidence(top) >= minLockConfidence;
    const state: TrackerState = !top ? "lost" : !separated ? "ambiguous" : !enough ? "acquiring" : "locked";

    // Ambiguous lists every candidate within the margin of the best; otherwise only the best.
    const shown = !top ? [] : separated ? [top] : fitting.filter((c) => c.score - top.score < required);
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
    return { event, lock: state === "locked" ? top.m : null, trace };
  };

  const emit = (event: PositionEvent, trace: (from: TrackerState | null) => TrackerTrace): PositionEvent => {
    const from = last?.state ?? null;
    last = event;
    return options.trace ? { ...event, trace: trace(from) } : event;
  };
  const unchanged = (event: PositionEvent) => !!last && JSON.stringify({ ...last, timeMs: 0 }) === JSON.stringify({ ...event, timeMs: 0 });

  const evaluate = (final: boolean): PositionEvent[] => {
    if (!observed) return [];
    const a = assessOutsidePhrases();
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
    lastFrame = Math.max(lastFrame, ph.end + 1);
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
    return { event, lock: null, trace: (from: TrackerState | null) => ({ ...a.trace(from), transition: { from, to: "acquiring" as const } }) };
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
        observed = observed.slice(0, at) + tail;
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
    surahStart = false;
    basmalaHeard = false;
    fatihaNext = false;
    takbirHeard = false;
    afterPhrase = false;
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
    // A weak local search jumps around chasing evidence that may belong elsewhere; only its current run may advance.
    if (weak && JSON.stringify(prev.span.start) !== JSON.stringify(candidate.span.start)) return null;
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

  const goLost = (a: ReturnType<typeof assess>): PositionEvent => {
    if (memorization) frozen ??= steady;
    local = null;
    weak = false;
    pending = null;
    reacquiring = true;
    return emit({ ...a.event, expected: expectedWhileLost() }, a.trace);
  };

  /** After `lost`: global search over the latest phonemes, until it locks somewhere. */
  const reacquire = (): PositionEvent[] => {
    keepLast(WINDOW);
    if (phraseList.length) collapseRepeat();
    const a = assessOutsidePhrases();
    if (a.lock) {
      startLocal(a.lock);
      return [emit({ ...a.event, expected: expectedFor(a.event.observed[0].word) }, a.trace)];
    }
    const event = { ...a.event, expected: expectedWhileLost() };
    // Junk that fits nowhere would otherwise favour passages with text before the recited phrase to absorb it.
    if (a.event.state === "lost") keepLast(LOOKBACK);
    return unchanged(event) ? [] : [emit(event, a.trace)];
  };

  /**
   * Phoneme by phoneme after the first lock. Local tracking emits one event per change of word or run start. When its
   * fit degrades, global search over the latest phonemes runs alongside: a distant global lock that holds for
   * `jumpPhonemes` commits a jump, and nothing fitting anywhere for long enough is `lost`.
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
      const heardPhrase = hear(ph.p, before, local.position().stepCost);
      if (heardPhrase) {
        const from = heardCost.length - heardPhrase.length;
        const cost = heardCost.slice(from).reduce((a, b) => a + b, 0) / heardPhrase.length;
        if (cost >= PHRASE_LOCAL_COST) {
          // Not Quran text where we are: rewind the local search to before the phrase, as if it was never heard.
          local.restore(heardState[from]);
          if (heardPhrase.name === "basmala") local.openSurah();
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
      if (!weak && pos.recentCost > WEAK_COST) {
        weak = true;
        steady = last!.expected;
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
        if (pending && pending.held >= (afterPhrase ? 1 : jumpPhonemes)) {
          if (memorization) frozen ??= steady;
          startLocal(pending.m);
        } else if (a.event.state === "lost" && pos.recentCost > maxCost && observed.length >= LOST_AFTER) {
          events.push(goLost(a));
          continue;
        }
      }
      const e = localEvent(shortlist);
      if (e) events.push(e);
    }
    return events;
  };

  /**
   * Before a new chunk: drop all but `MAX_HISTORY` phonemes of earlier chunks, or all but `LOOKBACK` after `lost`, and
   * the posterior rows before what's kept. Done here rather than after evaluating, so `end` sees what the last push saw.
   */
  const trimHistory = () => {
    if (phraseCut) {
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
      return [emit(a.event, a.trace)];
    }
    observed = saved.observed;
    observedConf = saved.observedConf;
    observedFrames.splice(0, observedFrames.length, ...saved.frames);
    return [];
  };

  return {
    push(evidence) {
      validate(evidence);
      frameMs = evidence.frameMs;
      if (evidence.posteriors) withPosteriors = true;
      else decodedOnly = true;
      if (local || reacquiring) return follow(evidence);
      trimHistory();
      if (evidence.posteriors) (posteriors ??= new PosteriorFrames(index.tokens.size)).append(evidence.posteriors);
      for (const ph of evidence.phonemes) {
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
      const events = evaluate(false);
      if (!lockedOn) events.push(...evaluateSuffix(evidence.phonemes.length));
      if (lockedOn) {
        lockEvent = events.at(-1)!;
        startLocal(lockedOn);
        posteriors = null;
      }
      return events;
    },
    end() {
      if (!local && !reacquiring) return evaluate(true);
      const timeMs = lastFrame * (frameMs ?? 0);
      // Nothing followed the lock: its event, with the global search's trace, is still the answer.
      if (local && !emittedSinceLock) {
        const e = { ...lockEvent!, timeMs };
        return [e.trace ? { ...e, trace: { ...e.trace, transition: { from: "locked" as const, to: "locked" as const } } } : e];
      }
      const final = { ...last!, timeMs };
      return [options.trace ? { ...final, trace: quietTrace(final.state, final.state) } : final];
    },
  };
}
