import { collapseHeld } from "./fuzzy.js";
import { lexicalWords } from "./checker-lexical.js";
import { spanLogP, type Channel } from "./noisy-channel.js";
import type { ReferenceIndex } from "./reference.js";

const CONSONANTS = new Set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي");
const EDGE = new Set("ۦۥں۾ڇؙۜ");
/** The checker's existing waqf-edge strip, then the channel estimator's held-phone collapse. */
function phones(raw: string): string[] {
  let a = 0, b = raw.length;
  while (a < b && EDGE.has(raw[a]!)) a++;
  while (b > a && EDGE.has(raw[b - 1]!)) b--;
  return collapseHeld([...raw.slice(a, b)]);
}

const cache = new WeakMap<ReferenceIndex, readonly string[]>();
function vocabulary(index: ReferenceIndex): readonly string[] {
  let forms = cache.get(index);
  if (!forms) {
    // Only attested isolated pronunciations are pausal rivals. No generated phone strings.
    forms = [...new Set((index.isolatedWordForms ?? []).map(raw => phones(raw).join("")))]
      .filter(raw => [...raw].some(p => CONSONANTS.has(p)));
    cache.set(index, forms);
  }
  return forms;
}

/**
 * Evidence filter for an already manner-rekeyed accusation. Both hypotheses see exactly the same
 * stripped greedy tokens and folded channel. The caller keeps the accusation only on a strict win.
 */
export function pausalChannelChoice(
  index: ReferenceIndex, wi: number, heard: string, mannerOf: (raw: string) => string | null, channel: Channel,
): { writtenLogP: number; rivalLogP: number; rival: string | null } {
  const word = index.words[wi]!;
  const written = index.isolatedWordForms?.[wi] || index.phonemes.slice(word.start, word.end);
  const observed = phones(heard);
  const writtenLogP = spanLogP(observed, phones(written), channel);
  let rivalLogP = -Infinity, rival: string | null = null;
  const manner = mannerOf(heard);
  if (!manner || !observed.length || !channel.phones.length) return { writtenLogP, rivalLogP, rival };
  const lexical = lexicalWords(index);
  const own = new Set([written, index.phonemes.slice(word.start, word.end), ...(index.wordForms?.[wi] ?? [])]
    .map(raw => phones(raw).join("")));
  for (const form of vocabulary(index)) {
    if (own.has(form) || mannerOf(form) !== manner || lexical.explainsSeam(wi, form)) continue;
    const score = spanLogP(observed, [...form], channel);
    if (score > rivalLogP) { rivalLogP = score; rival = form; }
  }
  return { writtenLogP, rivalLogP, rival };
}
