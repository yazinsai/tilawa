import type { ReferenceIndex } from "./reference.js";
import { letterSkeleton } from "./parallel-phrases.js";

const cache = new WeakMap<ReferenceIndex, ReadonlyMap<number, readonly string[]>>();

/**
 * Pausal fillers of an ayah's final word after the identical two-word lead. The ayah boundary closes
 * the choice. Phones come from final occurrences, where the reference already has the pausal ending.
 * Mid-ayah words and indexes without mushaf text have no slots.
 */
export function formulaSlots(index: ReferenceIndex): ReadonlyMap<number, readonly string[]> {
  const hit = cache.get(index);
  if (hit) return hit;
  const slots = new Map<number, string[]>();
  const text = index.wordText;
  if (!text || text.length !== index.words.length) {
    cache.set(index, slots);
    return slots;
  }
  const skel = text.map(letterSkeleton);
  const groups = new Map<string, number[]>();
  for (let i = 2; i < index.words.length; i++) {
    const word = index.words[i]!;
    const next = index.words[i + 1];
    const lead = index.words[i - 2]!;
    if (next && next.surah === word.surah && next.ayah === word.ayah) continue;
    if (lead.surah !== word.surah || lead.ayah !== word.ayah) continue;
    if (index.words.slice(i - 2, i + 1).some((w) => w.start === w.end)) continue;
    const key = skel.slice(i - 2, i).join("\0");
    const list = groups.get(key);
    if (list) list.push(i);
    else groups.set(key, [i]);
  }
  for (const group of groups.values()) for (const i of group) {
    const forms = new Set<string>();
    for (const j of group) {
      if (skel[i] === skel[j]) continue;
      const w = index.words[j]!;
      forms.add(index.phonemes.slice(w.start, w.end));
    }
    if (forms.size) slots.set(i, [...forms]);
  }
  cache.set(index, slots);
  return slots;
}
