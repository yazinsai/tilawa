import { collapseHeld } from "./fuzzy.js";
import { letterSkeleton } from "./parallel-phrases.js";
import type { ReferenceIndex } from "./reference.js";

const normalize = (s: string) => collapseHeld([...s]).join("");
const SUN = new Set("تثدذرزسشصضطظلن");
const PREFIXES: Readonly<Record<string, readonly string[]>> = {
  "": [""], و: ["وَ"], ف: ["فَ"], ب: ["بِ"], ك: ["كَ"], ل: ["لِ", "لَ"],
};
interface Form { prefix: string; stem: string; phones: string; onset: string; tail: string }
const cache = new WeakMap<ReferenceIndex, ReturnType<typeof build>>();

/** Surface segments attested in the mushaf, retaining both letter and pronunciation identity. */
function build(index: ReferenceIndex) {
  const forms = index.words.map((w, wi): Form[] => {
    const text = index.wordText?.[wi];
    if (!text) return [];
    const letters = letterSkeleton(text);
    const phones = [...new Set([index.phonemes.slice(w.start, w.end), ...(index.wordForms?.[wi] ?? [])]
      .filter(Boolean).map(normalize))];
    const out: Form[] = [];
    for (const prefix of ["", "ال", "و", "ف", "ب", "ك", "ل"]) {
      if (!letters.startsWith(prefix) || letters.length === prefix.length) continue;
      const stem = letters.slice(prefix.length);
      // The article's lam assimilates into a sun letter. Keep its extra consonant separate from the stem's onset.
      const article = SUN.has(stem[0]) ? stem[0] : "ل";
      const onsets = prefix === "ال" ? [article, "ءَ" + article] : PREFIXES[prefix];
      for (const p of phones) for (const onset of onsets) {
        if (p.startsWith(onset) && p.length > onset.length) {
          out.push({ prefix, stem, phones: p, onset, tail: p.slice(onset.length) });
        }
      }
    }
    return out;
  });
  return {
    /** The residue restores an exact source word whose proclitic differs from the word that borrowed its stem. */
    reattaches(source: number, bound: number, residue: string, heard: string): boolean {
      if (!residue || /[ؙۜ]/u.test(residue + heard)) return false;
      const joined = normalize(residue + heard), prefix = normalize(residue);
      if (forms[bound].some(f => f.phones === joined)) return false;
      return forms[source].some(a => a.prefix && a.phones === joined && a.onset.startsWith(prefix) &&
        forms[bound].some(b => a.prefix !== b.prefix && a.stem === b.stem && a.tail === b.tail));
    },
  };
}

export function procliticWords(index: ReferenceIndex) {
  let hit = cache.get(index);
  if (!hit) { hit = build(index); cache.set(index, hit); }
  return hit;
}
