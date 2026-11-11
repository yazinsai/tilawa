import type { ReferenceIndex } from "./reference.js";

/**
 * Nats by which an attested parallel phrase must beat the written phrase on the CTC forward sum.
 * Correct unspliced surah-37 recitations by four readers outside the lab held-out set, decoded with the
 * shipped listener, never preferred the parallel (175 phrases, worst −8.0 nats). On phone audio the same
 * contrast moves up: fully clean practice spans stay negative (worst −0.7), and the worst correct span
 * away from a planted mistake is +2.9. Four nats sits above that tail and below the planted swaps.
 * Not a per-edit cost. One switch, per phrase.
 */
export const PHRASE_SWITCH = 4;

/**
 * Mean probability of the best non-blank token, on frames the blank does not own, below which the window is
 * not a heard phrase. A flat inventory sits near 0.02; 0.1 is a few times that.
 */
export const PHRASE_PEAK = 0.1;

/** Four words is the shortest phrase that still enters and leaves on a shared word around one interior difference. */
const PHRASE = 4;

/**
 * Vowels, madd and marks. Everything else in the inventory is a consonant, including the tajweed variants
 * (ڇ ں ۾) the checker's plain consonant set does not list. A parallel that keeps the same consonants is a
 * vowel or waqf ending, which the phrase graph does not switch.
 */
const MARKS = new Set("َُِ۪ۦۥاـؙۜ");

const DIACRITICS = /[\u0640\u064b-\u065f\u0670\u06d6-\u06ed]/g;

/** Letter skeleton of a mushaf word: diacritics off, alef and alif maqsura folded. */
export function letterSkeleton(word: string): string {
  return word.replace(DIACRITICS, "").replace(/[ٱأإآى]/g, (c) => (c === "ى" ? "ي" : "ا"));
}

function consonants(phonemes: string): string {
  let out = "";
  for (const c of phonemes) if (!MARKS.has(c)) out += c;
  return out;
}

/**
 * One attested substitute for the written phrase: the shared words keep the phonemes written here, and the
 * interior word takes the phonemes of a parallel occurrence. No inserted or restarted speech, only that word.
 */
export interface ParallelPath {
  /** Word index of the phrase's first word. */
  start: number;
  /** Word index of the interior word that differs. */
  word: number;
  /** Phonemes of the four written words. */
  written: string;
  /** The same phrase with `word` replaced by the parallel word's phonemes. */
  alt: string;
  /** Phonemes of the parallel word alone. */
  altWord: string;
}

const cache = new WeakMap<ReferenceIndex, ReadonlyMap<number, readonly ParallelPath[]>>();

/**
 * Every 4-word mushaf phrase whose interior word has another attested completion, keyed by that word.
 * Longer phrases with one interior difference contain one of these: the three shared words are a subset.
 * Paths do not cross a surah. Built once per index.
 */
export function parallelPaths(index: ReferenceIndex): ReadonlyMap<number, readonly ParallelPath[]> {
  let hit = cache.get(index);
  if (hit) return hit;
  const text = index.wordText;
  const map = new Map<number, ParallelPath[]>();
  if (!text || text.length !== index.words.length) {
    cache.set(index, map);
    return map;
  }
  const skel = text.map(letterSkeleton);
  const phone = index.words.map((w) => index.phonemes.slice(w.start, w.end));
  const groups = new Map<string, number[]>();
  for (let i = 0; i + PHRASE <= index.words.length; i++) {
    if (index.words[i].surah !== index.words[i + PHRASE - 1].surah) continue;
    if (phone.slice(i, i + PHRASE).some((p) => !p)) continue;
    for (let j = 1; j < PHRASE - 1; j++) {
      const key = `${j}\0${skel.slice(i, i + j).join("\0")}\0|\0${skel.slice(i + j + 1, i + PHRASE).join("\0")}`;
      const list = groups.get(key);
      if (list) list.push(i);
      else groups.set(key, [i]);
    }
  }
  const seen = new Set<string>();
  for (const [key, starts] of groups) {
    const j = Number(key.slice(0, key.indexOf("\0")));
    const by = new Map<string, Set<string>>();
    for (const s of starts) {
      const k = skel[s + j];
      const set = by.get(k);
      if (set) set.add(phone[s + j]);
      else by.set(k, new Set([phone[s + j]]));
    }
    if (by.size < 2) continue;
    for (const s of starts) {
      const mine = consonants(phone[s + j]);
      const written = phone.slice(s, s + PHRASE).join("");
      for (const [k, forms] of by) {
        if (k === skel[s + j]) continue;
        for (const altWord of forms) {
          if (consonants(altWord) === mine || altWord === phone[s + j]) continue;
          const id = `${s}\0${j}\0${altWord}`;
          if (seen.has(id)) continue;
          seen.add(id);
          const alt = phone.slice(s, s + j).join("") + altWord + phone.slice(s + j + 1, s + PHRASE).join("");
          const path: ParallelPath = { start: s, word: s + j, written, alt, altWord };
          const list = map.get(path.word);
          if (list) list.push(path);
          else map.set(path.word, [path]);
        }
      }
    }
  }
  cache.set(index, map);
  return map;
}
