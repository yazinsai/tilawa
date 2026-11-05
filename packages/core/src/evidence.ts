import type { WordRef } from "./events.js";

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
