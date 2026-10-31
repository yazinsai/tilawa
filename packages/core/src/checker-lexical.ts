import { collapseHeld, deletionCost, insertionCost, subCost, UNIT } from "./fuzzy.js";
import type { ReferenceIndex } from "./reference.js";

const CONSONANTS = new Set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي");
const form = (s: string) => collapseHeld([...s]).join("");

/**
 * Align consonants independently of vowel placement. Keep the original positions for run costs: vowels between
 * identical consonants must not create a held run. Equal-cost paths choose the fewest genuinely hard edits.
 */
export function hardConsonantEdits(expected: string, observed: string): number {
  const a = form(expected), b = form(observed);
  const positions = (s: string) => [...s].flatMap((p, i) => CONSONANTS.has(p) ? [i] : []);
  const r = positions(a), o = positions(b);
  type Score = { cost: number; hard: number };
  const add = (s: Score, cost: number): Score => ({ cost: s.cost + cost, hard: s.hard + Number(cost >= UNIT) });
  const best = (...scores: Score[]) => scores.reduce((a, b) => a.cost < b.cost || (a.cost === b.cost && a.hard <= b.hard) ? a : b);
  let row: Score[] = [{ cost: 0, hard: 0 }];
  for (let j = 0; j < o.length; j++) row.push(add(row[j], insertionCost(a, r[0] ?? a.length, b[o[j]])));
  for (let i = 0; i < r.length; i++) {
    const next = [add(row[0], deletionCost(a, r[i]))];
    for (let j = 0; j < o.length; j++) next.push(best(
      add(row[j], subCost(a[r[i]], b[o[j]])),
      add(row[j + 1], deletionCost(a, r[i])),
      add(next[j], insertionCost(a, r[i + 1] ?? a.length, b[o[j]])),
    ));
    row = next;
  }
  return row[o.length].hard;
}

/** Shared by checker instances; only supported reference pronunciations enter the whole-word vocabulary. */
const cache = new WeakMap<ReferenceIndex, ReturnType<typeof build>>();
function build(index: ReferenceIndex) {
  const forms = index.words.map((w, i) => [...new Set([
    index.phonemes.slice(w.start, w.end), ...(index.wordForms?.[i] ?? []),
  ].filter(Boolean).map(form))]);
  const vocabulary = new Set(forms.flat());
  const matches = (wi: number, heard: string) => forms[wi].includes(form(heard));
  return {
    matches,
    alternative(wi: number, heard: string): boolean {
      const normalized = form(heard);
      return vocabulary.has(normalized) && !matches(wi, heard) &&
        forms[wi].length > 0 && forms[wi].every(expected => hardConsonantEdits(expected, normalized) >= 2);
    },
  };
}

export function lexicalWords(index: ReferenceIndex) {
  let words = cache.get(index);
  if (!words) { words = build(index); cache.set(index, words); }
  return words;
}
