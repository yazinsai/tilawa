import type { ReferenceIndex } from "./reference.js";

/** Log multipliers of substitution, insertion, deletion, lexical jump, forward movement and repeat. */
export interface ReferenceEditModel { logScale: readonly number[] }
export const EDIT_FEATURES = 6;

/** A jump over the remainder of one word can express a lexical discrepancy. No extra edge is added. */
export function jumpFeature(index: ReferenceIndex, source: number, destination: number): number {
  if (destination < source) return 5;
  return index.wordAt(destination) === index.wordAt(source) + 1 ? 3 : 4;
}

export function editScale(model: ReferenceEditModel | null, feature: number): number {
  return model ? Math.exp(model.logScale[feature]) : 1;
}

/** Training-only copy: unchanged words retain their original ordinal, changed words have no correspondence. */
export function editedReference(
  index: ReferenceIndex, word: number, kind: "delete" | "insert" | "substitute", pronunciation = "",
): { index: ReferenceIndex; correspondence: number[] } {
  if (!index.words[word] || (kind !== "delete" && !pronunciation)) throw new Error("Invalid reference edit");
  let phonemes = "";
  const words: ReferenceIndex["words"] = [];
  const correspondence: number[] = [];
  const add = (source: number, p: string, mapped: number) => {
    const original = index.words[source];
    const start = phonemes.length;
    phonemes += p;
    words.push({ ...original, start, end: phonemes.length });
    correspondence.push(mapped);
  };
  index.words.forEach((w, i) => {
    if (i === word && kind === "insert") add(i, pronunciation, -1);
    if (i === word && kind === "delete") return;
    add(i, i === word && kind === "substitute" ? pronunciation : index.phonemes.slice(w.start, w.end),
      i === word && kind === "substitute" ? -1 : i);
  });
  const starts = words.map(w => w.start);
  const wordAt = (at: number) => {
    let lo = 0, hi = starts.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (starts[mid] <= at) lo = mid; else hi = mid - 1; }
    return lo;
  };
  return { index: { ...index, words, phonemes, wordAt, wordForms: undefined, isolatedWordForms: undefined,
    wordText: undefined, grams: new Map() }, correspondence };
}
