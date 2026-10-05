import type { AlignmentOp } from "./align.js";
import { UNIT, alignSubstring } from "./fuzzy.js";
import { EVENT_SCHEMA_VERSION } from "./events.js";
import type { EngineEvent, PositionCandidate, PositionEvent, TrackerState, TrackerTrace, WordRef } from "./events.js";
import { EvidenceError } from "./evidence.js";
import type { PhonemeEvidence } from "./evidence.js";
import type { ReferenceIndex } from "./reference.js";

export interface TrackerOptions {
  mode?: "free" | "memorization";
  /** Grams occurring more often than this carry too little location information to vote. */
  maxGramOccurrences?: number;
  maxCandidates?: number;
  /** Candidates whose phoneme error rate exceeds this are discarded; with none left the tracker is lost. */
  maxPer?: number;
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
}

export function createTracker(index: ReferenceIndex, options: TrackerOptions = {}): Tracker {
  const maxGramOccurrences = options.maxGramOccurrences ?? 5000;
  const maxCandidates = options.maxCandidates ?? 64;
  const maxPer = options.maxPer ?? 0.35;
  const wordStarts = new Set(index.words.map((w) => w.start));
  const isWordStart = (offset: number) => wordStarts.has(offset);

  let observed = "";
  let frameMs: number | null = null;
  let lastFrame = 0;
  let last: PositionEvent | null = null;

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
  };

  const exactScan = (query: string): Match[] => {
    const found: Match[] = [];
    for (let at = index.phonemes.indexOf(query); at !== -1 && found.length < maxCandidates; at = index.phonemes.indexOf(query, at + 1)) {
      const path: AlignmentOp[] = [...query].map((p, k) => ({ op: "match", ref: p, refAt: k, obsAt: k }));
      found.push({ offset: at, length: query.length, cost: 0, per: 0, votes: 0, path });
    }
    return found;
  };

  const fuzzyMatches = (query: string): Match[] => {
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
      const a = alignSubstring(index.phonemes, lo, hi, query, isWordStart);
      if (!a || a.per > maxPer) continue;
      const prev = byStart.get(a.start);
      if (prev && prev.cost <= a.cost) continue;
      byStart.set(a.start, { offset: a.start, length: a.end - a.start, cost: a.cost, per: a.per, votes: bin.votes, path: a.path });
    }
    return [...byStart.values()].sort((a, b) => a.cost - b.cost || a.offset - b.offset);
  };

  const ref = (i: number): WordRef => {
    const w = index.words[i];
    return { surah: w.surah, ayah: w.ayah, word: w.word };
  };

  const normCost = (m: Match) => m.cost / UNIT / observed.length;

  const toCandidate = (m: Match, confidence: number): PositionCandidate => {
    const end = ref(index.wordAt(m.offset + Math.max(0, m.length - 1)));
    return { word: end, span: { start: ref(index.wordAt(m.offset)), end }, confidence, cost: normCost(m), per: m.per };
  };

  const evaluate = (final: boolean): EngineEvent[] => {
    if (!observed) return [];
    const matches = fuzzyMatches(observed);
    // Anything closer to the best than one confusable substitution is indistinguishable from it.
    const tied = matches.filter((m) => m.cost - matches[0].cost < UNIT / 2);
    const state: TrackerState = tied.length === 0 ? "lost" : tied.length === 1 ? "locked" : "ambiguous";
    const confidence = tied.length ? 1 / tied.length : 0;
    const candidates = tied.map((m) => toCandidate(m, confidence));
    const event: PositionEvent = {
      type: "position",
      v: EVENT_SCHEMA_VERSION,
      timeMs: lastFrame * (frameMs ?? 0),
      referenceVersion: index.version,
      expected: state === "locked" ? candidates[0].word : null,
      observed: candidates,
      state,
    };
    const changed = !last || JSON.stringify({ ...last, timeMs: 0 }) !== JSON.stringify({ ...event, timeMs: 0 });
    if (!changed && !final) return [];
    const from = last?.state ?? null;
    last = event;
    if (!options.trace) return [event];
    const top = matches[0];
    const trace: TrackerTrace = {
      shortlist: matches.map((m) => ({
        span: toCandidate(m, 0).span,
        offset: m.offset,
        length: m.length,
        votes: m.votes,
        cost: normCost(m),
        per: m.per,
      })),
      transition: { from, to: state },
      path: top ? top.path : null,
    };
    return [{ ...event, trace }];
  };

  return {
    push(evidence) {
      validate(evidence);
      frameMs = evidence.frameMs;
      for (const ph of evidence.phonemes) {
        observed += ph.p;
        lastFrame = Math.max(lastFrame, ph.end + 1);
      }
      return evaluate(false);
    },
    end() {
      return evaluate(true);
    },
  };
}
