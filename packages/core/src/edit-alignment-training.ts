import type { ReferenceIndex } from "./reference.js";
import type { WordRef } from "./events.js";

export interface EditTrainingLabel {
  split: string;
  scored: boolean;
  excluded?: string;
  kind: string;
  ayahs: string[];
  events: { errorClass: string; location: WordRef; words?: number }[];
  non_errors: { kind: string; location: WordRef }[];
  unsupported: unknown[];
}
const key = (w: WordRef) => `${w.surah}:${w.ayah}:${w.word}`;

/** Reconstruct spoken text; scripted mistakes never count as acoustic errors. */
export function editTrainingReference(index: ReferenceIndex, label: EditTrainingLabel): string | null {
  if (label.split !== "train") throw new Error("Edit alignment training refuses held-out speakers");
  if (!label.scored || label.excluded || label.unsupported.length
    || !["none", "skip_word", "skip_ayah", "repeat"].includes(label.kind)
    || label.events.some(e => !["omitted_word", "skipped_passage"].includes(e.errorClass))
    || label.non_errors.some(e => e.kind !== "repeat")) return null;
  const ayahs = new Set(label.ayahs), deleted = new Set<string>();
  for (const e of label.events) {
    const start = index.words.findIndex(w => key(w) === key(e.location));
    if (start < 0 || start + (e.words ?? 1) > index.words.length) throw new Error("Script word absent from reference");
    for (let k = 0; k < (e.words ?? 1); k++) deleted.add(key(index.words[start + k]));
  }
  const repeats = new Set(label.non_errors.map(e => key(e.location)));
  const words = index.words.map((w, i) => ({ w, i })).filter(({ w }) => ayahs.has(`${w.surah}:${w.ayah}`) && !deleted.has(key(w)));
  if (!words.length) return null;
  return words.map(({ w, i }, at) => {
    const phones = at === words.length - 1 ? index.isolatedWordForms?.[i] ?? index.phonemes.slice(w.start, w.end)
      : index.phonemes.slice(w.start, w.end);
    return repeats.has(key(w)) ? phones + phones : phones;
  }).join("");
}
