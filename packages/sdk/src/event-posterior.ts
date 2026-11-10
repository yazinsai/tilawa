import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence, PhraseAsset, ReferenceIndex, WrongWordEventEvidence, WrongWordRecheck, CheckerOptions } from "@tilawa/core";
import { insertionCells } from "./seam-insertion.js";
import { transcriptionLattice, warningEventMass } from "./event-lattice.js";
import type { EventMass, LatticeLimits } from "./event-lattice.js";

// Draft resource limits. Training-speaker calibration must freeze these before target diagnostics or scoring.
export const EVENT_LATTICE: LatticeLimits = { prefixes: 64, labels: 5, frames: 200 };
export const ENLARGED_EVENT_LATTICE: LatticeLimits = { prefixes: 256, labels: 10, frames: 200 };
export interface EventPosteriorResult { candidate: WrongWordRecheck; base: EventMass; enlarged: EventMass; veto: boolean }

export function eventPosteriorRecheck(index: ReferenceIndex, observe?: (result: EventPosteriorResult) => void,
  limits = EVENT_LATTICE, enlargedLimits = ENLARGED_EVENT_LATTICE): NonNullable<CheckerOptions["wrongWordRecheck"]> {
  const tokens = [...index.tokens];
  return (candidate, judge) => {
    if (candidate.rows.length > limits.frames) return false;
    const base = warningEventMass(transcriptionLattice(candidate.rows, tokens, candidate.fromFrame, limits), judge);
    const enlarged = warningEventMass(transcriptionLattice(candidate.rows, tokens, candidate.fromFrame, enlargedLimits), judge);
    const veto = base.quiet > base.warning && enlarged.quiet > enlarged.warning;
    observe?.({ candidate, base, enlarged, veto });
    return veto;
  };
}

/** Generate only listener veto metadata. Primary tokens, rows and tracker decisions are retained verbatim. */
export function addEventPosteriorEvidence(chunks: PhonemeEvidence[], index: ReferenceIndex, phrases?: PhraseAsset,
  observe?: (result: EventPosteriorResult) => void, limits = EVENT_LATTICE, enlargedLimits = ENLARGED_EVENT_LATTICE): PhonemeEvidence[] {
  const cells = insertionCells(chunks);
  if (!cells.length) return chunks;
  const metadata: WrongWordEventEvidence[] = [];
  const tracker = createTracker(index, { mode: "free", phrases, acquireMs: 1000 });
  const recheck = eventPosteriorRecheck(index, result => {
    observe?.(result);
    if (result.veto) metadata.push({ word: result.candidate.word, observed: result.candidate.observed,
      fromFrame: result.candidate.fromFrame, toFrame: result.candidate.toFrame,
      quiet: result.base.quiet, enlargedQuiet: result.enlarged.quiet });
  }, limits, enlargedLimits);
  const checker = createChecker(index, { phrases, wrongWordRecheck(candidate, judge) {
    recheck(candidate, judge);
    return false; // Discovery cannot change the state being sampled.
  } });
  for (const e of cells) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  if (!metadata.length) return chunks;
  const at = chunks.findIndex(c => c.phonemes.length);
  return chunks.map((c, i) => i !== at ? c : { ...c,
    phonemes: c.phonemes.map((p, j) => j ? p : { ...p, wrongWordEvents: metadata }) });
}
