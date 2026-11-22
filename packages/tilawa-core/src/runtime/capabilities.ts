import type { ErrorClass } from "../engine/index.js";

export const MEMORIZATION_CLASSES: ErrorClass[] = ["wrong_word", "omitted_word", "added_word", "skipped_passage"];

export const ALL_CLASSES: ErrorClass[] = [
  ...MEMORIZATION_CLASSES,
  "short_vowel",
  "consonant_h_ha",
  "consonant_ain_hamza",
  "consonant_sad_sin",
  "consonant_dad_dal",
  "consonant_ta_ta",
  "consonant_qaf_kaf",
  "consonant_dhal_zay",
  "consonant_tha_sin",
  "missing_shadda",
  "madd_natural",
];

/** Every class the model's metadata doesn't list as supported is `unsupported`, so v1 can ship memorization-only. */
export function capabilities(supported: readonly ErrorClass[]) {
  const s = new Set(supported);
  return { supported: ALL_CLASSES.filter((c) => s.has(c)), unsupported: ALL_CLASSES.filter((c) => !s.has(c)) };
}
