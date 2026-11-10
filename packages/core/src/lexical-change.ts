import type { LexicalChangeEvidence } from "./evidence.js";
import type { WordRef } from "./events.js";

export interface WordLocalizationCandidate {
  word: WordRef;
  expected: string;
  observed: string;
  fromFrame: number;
  toFrame: number;
}

/** An ayah view from the unchanged alignment. The listener merges overlapping
 * views of one occurrence before requiring a complete context. */
export interface WordLocalizationContext {
  words: number[];
  coveredWords: number[];
  fromFrame: number;
  toFrame: number;
}

const key = (c: WordLocalizationCandidate) => JSON.stringify([
  c.word.surah, c.word.ayah, c.word.word, c.expected, c.observed, c.fromFrame, c.toFrame,
]);

/** A certificate can only veto its own occurrence. Conflicting calibration abstains;
 * repeated evidence keeps the greatest change score rather than accumulating votes. */
export class LexicalChangeCertificates {
  private values = new Map<string, LexicalChangeEvidence>();
  private conflicts = new Set<string>();

  add(c: LexicalChangeEvidence) {
    if (![c.word.surah, c.word.ayah, c.word.word].every(n => Number.isInteger(n) && n > 0) ||
      !Number.isInteger(c.fromFrame) || c.fromFrame < 0 || !Number.isInteger(c.toFrame) || c.toFrame <= c.fromFrame ||
      ![c.score, c.preservation].every(n => Number.isFinite(n) && n >= 0 && n <= 1) ||
      !c.expected || typeof c.observed !== "string") return;
    const k = key(c), previous = this.values.get(k);
    if (previous && previous.preservation !== c.preservation) this.conflicts.add(k);
    if (!previous || c.score > previous.score) this.values.set(k, c);
  }

  veto(c: WordLocalizationCandidate): boolean {
    const k = key(c), value = this.values.get(k);
    return !this.conflicts.has(k) && value !== undefined && value.score < value.preservation;
  }
}
