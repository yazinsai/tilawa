import { align } from "./align.js";
import { collapseHeld } from "./fuzzy.js";
import type { ReferenceIndex } from "./reference.js";

/**
 * Manner a phone keeps on a phone band. Nasal, liquid (lateral or trill), obstruent (stop or fricative)
 * and glide. ن/م are one nasal and ف/ك are one obstruent, so those channel pairs are not a contradiction.
 * Ghunna marks take the nasal of ن/م. Qalqala is not a manner.
 */
export const MANNER: Readonly<Record<string, "nasal" | "liquid" | "obstruent" | "glide">> = {
  "ن": "nasal", "م": "nasal", "ں": "nasal", "۾": "nasal",
  "ر": "liquid", "ل": "liquid",
  "و": "glide", "ي": "glide",
  "ء": "obstruent", "ب": "obstruent", "ت": "obstruent", "ث": "obstruent", "ج": "obstruent", "ح": "obstruent",
  "خ": "obstruent", "د": "obstruent", "ذ": "obstruent", "ز": "obstruent", "س": "obstruent", "ش": "obstruent",
  "ص": "obstruent", "ض": "obstruent", "ط": "obstruent", "ظ": "obstruent", "ع": "obstruent", "غ": "obstruent",
  "ف": "obstruent", "ق": "obstruent", "ك": "obstruent", "ه": "obstruent",
};

/** Sukun and pause marks. They are not a manner; a word is the same word without them. */
const MARK = new Set(["ڇ", "ۜ", "ؙ"]);

/** Manner classes of the consonants, held runs collapsed. Vowels and madd drop out. */
export function mannerSkeleton(phones: string): string {
  const parts: string[] = [];
  for (const p of collapseHeld([...phones].filter((c) => !MARK.has(c)))) {
    const m = MANNER[p];
    if (m) parts.push(m);
  }
  return parts.join(".");
}

/**
 * Consonant substitutions in the checker's unit alignment whose manner class changes.
 * A repeated copy of the same written consonant against the same heard manner is one position.
 * Same-manner pairs, vowel alignments and insertions are not a change.
 */
export function mannerChanges(written: string, heard: string): number {
  let n = 0;
  let prev = "";
  for (const op of align(written, heard)) {
    if (op.op !== "sub") {
      prev = "";
      continue;
    }
    const rm = MANNER[op.ref];
    const om = MANNER[op.obs];
    if (!rm || !om || rm === om) {
      prev = "";
      continue;
    }
    const key = `${op.ref}>${om}`;
    if (key !== prev) n++;
    prev = key;
  }
  return n;
}

const cache = new WeakMap<ReferenceIndex, { outside(wi: number, heard: string): boolean }>();

/**
 * Manner skeletons of every attested mushaf pronunciation. `outside` is true when the heard word's
 * skeleton is one of them and is not the written word or a word within ±3.
 */
export function mannerWords(index: ReferenceIndex): { outside(wi: number, heard: string): boolean } {
  let hit = cache.get(index);
  if (hit) return hit;
  const forms = index.words.map((w, i) => [...new Set([
    index.phonemes.slice(w.start, w.end), ...(index.wordForms?.[i] ?? []),
  ].filter(Boolean))]);
  const own = forms.map((list) => [...new Set(list.map(mannerSkeleton).filter((s) => s.includes(".")))]);
  const attested = new Set<string>();
  for (const list of own) for (const s of list) attested.add(s);
  const content = (i: number) => i >= 0 && i < index.words.length && index.words[i]!.start !== index.words[i]!.end;
  const near = (wi: number) => {
    const ban = new Set<string>();
    const take = (i: number) => { for (const s of own[i] ?? []) ban.add(s); };
    if (content(wi)) take(wi);
    let before = 0;
    for (let i = wi - 1; i >= 0 && before < 3; i--) if (content(i)) { take(i); before++; }
    let after = 0;
    for (let i = wi + 1; i < index.words.length && after < 3; i++) if (content(i)) { take(i); after++; }
    return ban;
  };
  hit = {
    outside(wi: number, heard: string): boolean {
      const sk = mannerSkeleton(heard);
      return sk.includes(".") && attested.has(sk) && !near(wi).has(sk);
    },
  };
  cache.set(index, hit);
  return hit;
}
