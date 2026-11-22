import type { ReferenceIndex } from "./reference.js";
import { letterSkeleton } from "./parallel-phrases.js";

/** Madd letters in the reference inventory. A run of these at a word end is the long vowel. */
const MADD = new Set(["ا", "ۦ", "ۥ"]);
const CONSONANTS = new Set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي");

export const isMaddLetter = (p: string): boolean => MADD.has(p);

function trail(phonemes: string): number {
  let n = 0;
  while (n < phonemes.length && MADD.has(phonemes[phonemes.length - 1 - n]!)) n++;
  return n;
}

/**
 * Last consonant of the word once its final madd run is set aside. The duration of a connected madd is measured
 * from this consonant's onset to the next word's hamza.
 */
export function stemConsonant(phonemes: string, start: number, end: number): number {
  let i = end - 1;
  while (i >= start && MADD.has(phonemes[i]!)) i--;
  while (i >= start && !CONSONANTS.has(phonemes[i]!)) i--;
  return i;
}

export interface MaddDrop {
  /** The hamza-initial word that follows. */
  next: number;
  /** Phonemes of the shorter word at the parallel ayah. */
  alt: string;
}

export interface MaddSites {
  /** Word index to the hamza word after it. Connected madd munfasil, from the written phonemes. */
  munfasil: ReadonlyMap<number, number>;
  /**
   * Munfasil words whose final long vowel, dropped, is another mushaf word, and whose 4-word phrase with that
   * word occurs at another ayah. The parallel word itself does not end in a madd: a second spelling of the same
   * long vowel is not the shorter form.
   */
  drops: ReadonlyMap<number, MaddDrop>;
}

const cache = new WeakMap<ReferenceIndex, MaddSites>();

export function maddSites(index: ReferenceIndex): MaddSites {
  const hit = cache.get(index);
  if (hit) return hit;
  const text = index.wordText;
  const empty: MaddSites = { munfasil: new Map(), drops: new Map() };
  if (!text || text.length !== index.words.length) {
    cache.set(index, empty);
    return empty;
  }
  const skel = text.map(letterSkeleton);
  const phone = index.words.map((w) => index.phonemes.slice(w.start, w.end));
  const attested = new Set(skel);
  const grams = new Map<string, number[]>();
  for (let i = 0; i + 4 <= index.words.length; i++) {
    if (index.words[i]!.surah !== index.words[i + 3]!.surah) continue;
    const key = skel.slice(i, i + 4).join("\0");
    const list = grams.get(key);
    if (list) list.push(i);
    else grams.set(key, [i]);
  }
  const munfasil = new Map<number, number>();
  const drops = new Map<number, MaddDrop>();
  for (let i = 0; i < index.words.length - 1; i++) {
    const ph = phone[i]!;
    if (!trail(ph)) continue;
    const w = index.words[i]!;
    const n = index.words[i + 1]!;
    if (n.surah !== w.surah || n.ayah !== w.ayah || !phone[i + 1]!.startsWith("ء")) continue;
    munfasil.set(i, i + 1);
    const sk = skel[i] ?? "";
    const last = sk.at(-1);
    if (!last || !"اوي".includes(last)) continue;
    const short = sk.slice(0, -1);
    if (!short || !attested.has(short)) continue;
    if (i + 3 >= index.words.length || index.words[i + 3]!.surah !== w.surah) continue;
    const key = [short, ...skel.slice(i + 1, i + 4)].join("\0");
    let alt = "";
    for (const j of grams.get(key) ?? []) {
      const there = index.words[j]!;
      if (there.surah === w.surah && there.ayah === w.ayah) continue;
      const altPh = phone[j]!;
      if (!altPh || trail(altPh)) continue;
      alt = altPh;
      break;
    }
    if (alt) drops.set(i, { next: i + 1, alt });
  }
  const sites: MaddSites = { munfasil, drops };
  cache.set(index, sites);
  return sites;
}
