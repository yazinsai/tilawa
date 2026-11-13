import type { WordRef } from "./events.js";

/** Complete hypotheses on the same native-frame window used by primary support. */
export interface SequenceEvidenceCandidate {
  word: WordRef;
  written: string;
  rival: string;
  fromFrame: number;
  toFrame: number;
}

export interface VowelResidualEvidence extends SequenceEvidenceCandidate {
  residualMargin: number;
  originalMargin: number;
  shuffledMargin: number;
  consonantMargin: number;
}

export function vowelResidualVeto(c: VowelResidualEvidence): boolean {
  const consonants = (s: string) => [...s].filter(p => "ءبتثجحخدذرزسشصضطظعغفقكلمنهوي".includes(p)).join("");
  return [c.residualMargin, c.originalMargin, c.shuffledMargin, c.consonantMargin].every(Number.isFinite) &&
    c.residualMargin > 0 && c.originalMargin < 0 && c.shuffledMargin < 0 && c.consonantMargin > 0 &&
    typeof c.written === "string" && !!c.written && typeof c.rival === "string" && !!c.rival &&
    consonants(c.written) !== consonants(c.rival);
}

export function sequenceEvidenceKey(c: SequenceEvidenceCandidate): string {
  return JSON.stringify([c.word.surah, c.word.ayah, c.word.word, c.written, c.rival, c.fromFrame, c.toFrame]);
}

/** Auxiliary listener judgment over a complete seam, tied to one proposed spelling. */
export interface SeamInsertionEvidence {
  word: WordRef;
  observed: string;
  fromFrame: number;
  toFrame: number;
  marginal: number;
}

export interface DecodedPhoneme {
  /** One token from the evidence inventory. */
  p: string;
  /** First and last frame (inclusive) of the token's CTC spike run. */
  start: number;
  end: number;
  conf: number;
  /** Retained by token-preserving evidence serialization and rechunking. */
  seamInsertions?: SeamInsertionEvidence[];
  vowelResidualVetoes?: VowelResidualEvidence[];
}

/** Per-frame top-k phoneme posteriors. See docs/evidence-contract.md. */
export interface TopKPosteriors {
  k: number;
  /** Absolute frame index of the first row. */
  startFrame: number;
  nFrames: number;
  /** Row-major nFrames x k token ids; 0 is the CTC blank, 1..N index the inventory tokens. */
  ids: ArrayLike<number>;
  /** Row-major nFrames x k probabilities, descending within a row. */
  probs: ArrayLike<number>;
}

export interface PhonemeEvidence {
  /** Must equal the reference index inventory id. */
  inventory: string;
  frameMs: number;
  phonemes: DecodedPhoneme[];
  posteriors?: TopKPosteriors;
  /** Frames before this one are final: no later evidence has a phoneme starting earlier. Optional; streams set it. */
  decodedTo?: number;
}

export class EvidenceError extends Error {}
