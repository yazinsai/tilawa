import type { SequenceEvidenceCandidate } from "@tilawa/core";

/** Training/probe transport. The matched control never decides a production veto. */
export interface WaveguideComparison {
  candidate: SequenceEvidenceCandidate & { frameMs: number };
  scores: { physical?: { written: number | null; rival: number | null }; control?: { written: number | null; rival: number | null } };
}

/** Certificates cannot be reused for a different window or either different pronunciation. */
export function waveguideSupportsRival(candidate: SequenceEvidenceCandidate, frameMs: number, comparisons: WaveguideComparison[]): boolean {
  return !comparisons.some(({ candidate: c, scores }) => c.frameMs === frameMs &&
    c.word.surah === candidate.word.surah && c.word.ayah === candidate.word.ayah && c.word.word === candidate.word.word &&
    c.written === candidate.written && c.rival === candidate.rival && c.fromFrame === candidate.fromFrame && c.toFrame === candidate.toFrame &&
    typeof scores.physical?.written === "number" && Number.isFinite(scores.physical.written) &&
    typeof scores.physical.rival === "number" && Number.isFinite(scores.physical.rival) && scores.physical.written > scores.physical.rival);
}
