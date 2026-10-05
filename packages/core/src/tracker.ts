import type { AlignmentOp } from "./align.js";
import { PosteriorFrames } from "./ctc.js";
import { MID_WORD_START, UNIT, alignSubstring, collapseHeld, isHeld } from "./fuzzy.js";
import { EVENT_SCHEMA_VERSION } from "./events.js";
import type { EngineEvent, PositionCandidate, PositionEvent, TraceCandidate, TrackerState, TrackerTrace, WordRef } from "./events.js";
import { EvidenceError } from "./evidence.js";
import type { DecodedPhoneme, PhonemeEvidence } from "./evidence.js";
import { createLocalSearch } from "./local.js";
import type { LocalSearch } from "./local.js";
import type { ReferenceIndex } from "./reference.js";

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
const MAX_HISTORY = 512;
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
    const bins: { lo: number; hi: number; votes: number }[] = [];
    for (const diagonal of [...votes.keys()].sort((a, b) => a - b)) {
      const bin = bins.at(-1);
      if (bin && diagonal - bin.hi <= slack) {
        bin.hi = diagonal;
        bin.votes += votes.get(diagonal)!;
      } else bins.push({ lo: diagonal, hi: diagonal, votes: votes.get(diagonal)! });
    }
    bins.sort((a, b) => b.votes - a.votes || a.lo - b.lo);

    const byStart = new Map<number, Match>();
    for (const bin of bins.slice(0, maxCandidates)) {
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
    if (!prior) return 0;
    let best = 0;
    for (let i = firstWord(m); i <= lastWord(m); i++) {
      const w = index.words[i];
      if (w.surah !== prior.surah) continue;
      const d = Math.abs(w.ayah - prior.ayah);
      best = Math.max(best, d === 0 ? 1 : d === 1 ? 0.5 : 0);
    }
    return best * MAX_PRIOR_BONUS * Math.min(1, Math.max(0, prior.weight));
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
    const a = assess();
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

  const startLocal = (m: Match) => {
    local = createLocalSearch(index, { offset: m.offset + m.length, origin: m.offset, edits: Math.round(m.per * m.length) });
    weak = false;
    pending = null;
    reacquiring = false;
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
    const a = assess();
    if (a.lock) {
      startLocal(a.lock);
      return [emit({ ...a.event, expected: expectedFor(a.event.observed[0].word) }, a.trace)];
    }
    const event = { ...a.event, expected: expectedWhileLost() };
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
        events.push(...reacquire());
        continue;
      }
      local.step(ph.p);
      const pos = local.position();
      if (!weak && pos.recentCost > WEAK_COST) {
        weak = true;
        steady = last!.expected;
      } else if (weak && pos.recentCost <= WEAK_COST) {
        weak = false;
        pending = null;
      }
      let shortlist: TraceCandidate[] = [];
      if (!weak) keepLast(LOOKBACK);
      else {
        keepLast(WINDOW);
        const a = assess();
        if (options.trace) shortlist = a.trace(null).shortlist;
        if (a.lock && !near(a.lock, pos.offset)) {
          pending = { m: a.lock, held: pending && sameLocation(pending.m, a.lock) ? pending.held + 1 : 1 };
        } else pending = null;
        if (pending && pending.held >= jumpPhonemes) {
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

  /** Drop all but `MAX_HISTORY` phonemes of earlier chunks, and the posterior rows before what's kept. */
  const trimHistory = () => {
    const drop = observed.length - MAX_HISTORY;
    if (drop <= 0) return;
    keepLast(MAX_HISTORY);
    posteriors?.dropBefore(observedFrames[0].start);
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
      for (const ph of evidence.phonemes) observe(ph);
      const events = evaluate(false);
      if (lockedOn) {
        lockEvent = events[0];
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
