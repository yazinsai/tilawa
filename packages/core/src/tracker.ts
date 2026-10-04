import { EVENT_SCHEMA_VERSION } from "./events.js";
import type { EngineEvent, PositionCandidate, PositionEvent, TrackerState, WordRef } from "./events.js";
import { EvidenceError } from "./evidence.js";
import type { PhonemeEvidence } from "./evidence.js";
import type { ReferenceIndex } from "./reference.js";

export interface TrackerOptions {
  mode?: "free" | "memorization";
  /** Grams occurring more often than this carry too little location information to vote. */
  maxGramOccurrences?: number;
  maxCandidates?: number;
}

export interface Tracker {
  push(evidence: PhonemeEvidence): EngineEvent[];
  end(): EngineEvent[];
}

interface Match {
  offset: number;
  length: number;
  cost: number;
}

export function createTracker(index: ReferenceIndex, options: TrackerOptions = {}): Tracker {
  const maxGramOccurrences = options.maxGramOccurrences ?? 5000;
  const maxCandidates = options.maxCandidates ?? 64;

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

  const exactMatches = (query: string): Match[] => {
    const found: Match[] = [];
    if (query.length < index.gramSize) {
      for (let at = index.phonemes.indexOf(query); at !== -1 && found.length < maxCandidates; at = index.phonemes.indexOf(query, at + 1)) {
        found.push({ offset: at, length: query.length, cost: 0 });
      }
      return found;
    }
    const votes = new Map<number, number>();
    for (let q = 0; q + index.gramSize <= query.length; q++) {
      const hits = index.grams.get(query.slice(q, q + index.gramSize));
      if (!hits || hits.length > maxGramOccurrences) continue;
      for (const pos of hits) {
        const diagonal = pos - q;
        if (diagonal >= 0) votes.set(diagonal, (votes.get(diagonal) ?? 0) + 1);
      }
    }
    const ranked = [...votes.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
    for (const [offset] of ranked) {
      if (index.phonemes.startsWith(query, offset)) found.push({ offset, length: query.length, cost: 0 });
      if (found.length >= maxCandidates) break;
    }
    if (found.length === 0 && votes.size === 0) {
      // Every gram was too common to vote; fall back to a direct scan.
      for (let at = index.phonemes.indexOf(query); at !== -1 && found.length < maxCandidates; at = index.phonemes.indexOf(query, at + 1)) {
        found.push({ offset: at, length: query.length, cost: 0 });
      }
    }
    return found.sort((a, b) => a.cost - b.cost || a.offset - b.offset);
  };

  const ref = (i: number): WordRef => {
    const w = index.words[i];
    return { surah: w.surah, ayah: w.ayah, word: w.word };
  };

  const toCandidate = (m: Match, confidence: number): PositionCandidate => {
    const end = ref(index.wordAt(m.offset + m.length - 1));
    return { word: end, span: { start: ref(index.wordAt(m.offset)), end }, confidence, cost: m.cost };
  };

  const evaluate = (final: boolean): EngineEvent[] => {
    if (!observed) return [];
    const matches = exactMatches(observed);
    const state: TrackerState = matches.length === 0 ? "lost" : matches.length === 1 ? "locked" : "ambiguous";
    const confidence = state === "locked" ? 1 : matches.length ? 1 / matches.length : 0;
    const candidates = matches.map((m) => toCandidate(m, confidence));
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
    last = event;
    return [event];
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
