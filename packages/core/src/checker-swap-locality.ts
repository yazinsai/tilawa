import { collapseHeld } from "./fuzzy.js";
import { lexicalWords } from "./checker-lexical.js";
import { spanLogP, type Channel } from "./noisy-channel.js";
import { letterSkeleton, parallelPaths } from "./parallel-phrases.js";
import type { ReferenceIndex } from "./reference.js";

const MARKS = new Set("ڇؙۜ");
const form = (s: string) => collapseHeld([...s].filter(p => !MARKS.has(p))).join("");
const cache = new WeakMap<ReferenceIndex, ReturnType<typeof build>>();

function build(index: ReferenceIndex) {
  const lexical = lexicalWords(index), parallels = parallelPaths(index);
  const identities = new Map<string, Set<string>>();
  const surahs = new Map<number, Set<string>>();
  const text = index.wordText;
  if (text?.length === index.words.length) index.words.forEach((w, wi) => {
    const identity = letterSkeleton(text[wi]);
    const local = surahs.get(w.surah) ?? new Set<string>();
    local.add(identity);
    surahs.set(w.surah, local);
    for (const raw of [index.phonemes.slice(w.start, w.end), ...(index.wordForms?.[wi] ?? [])]) {
      const f = form(raw);
      if (!f) continue;
      const names = identities.get(f) ?? new Set<string>();
      names.add(identity);
      identities.set(f, names);
    }
  });
  const linked = (wi: number, rival: string) => {
    const names = identities.get(form(rival));
    if (!names?.size) return null;
    const local = surahs.get(index.words[wi].surah);
    const alternatives = new Set<string>();
    for (const path of parallels.get(wi) ?? []) {
      for (const name of identities.get(form(path.altWord)) ?? []) alternatives.add(name);
    }
    return {
      words: [...names],
      sameSurah: [...names].some(name => local?.has(name)),
      parallel: [...names].some(name => alternatives.has(name)),
    };
  };
  return {
    /**
     * A named lexical rival keeps its identity. Otherwise rank whole attested, non-neighbour words under the
     * folded listener channel. Locality is checked after ranking; it must not bias the rival selection.
     * Missing text, an empty channel or an unidentifiable rival abstains.
     */
    assess(wi: number, heard: string, channel: Channel, named?: string) {
      if (!identities.size) return null;
      let rival = named && identities.has(form(named)) ? form(named) : lexical.surplusWord(wi, heard);
      // The spelling test permits soft vowel differences and returns its first form. An exact attested
      // pronunciation identifies the heard word before those softer variants (e.g. different case endings).
      if (!named && rival && identities.has(form(heard))) rival = form(heard);
      if (!rival) {
        if (!channel.phones.length) return null;
        const phones = [...form(heard)];
        if (!phones.length) return null;
        let best = -Infinity;
        for (const candidate of lexical.attestedForms) {
          const score = spanLogP(phones, [...candidate], channel);
          if (score <= best || lexical.nearbyWord(wi, candidate)) continue;
          best = score;
          rival = candidate;
        }
      }
      if (!rival) return null;
      const link = linked(wi, rival);
      return link ? { rival, ...link, keep: link.sameSurah || link.parallel } : null;
    },
  };
}

export function swapLocality(index: ReferenceIndex) {
  let hit = cache.get(index);
  if (!hit) { hit = build(index); cache.set(index, hit); }
  return hit;
}
