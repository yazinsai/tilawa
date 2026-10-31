import { collapseHeld, deletionCost, insertionCost, subCost, UNIT } from "./fuzzy.js";
import type { ReferenceIndex } from "./reference.js";

const CONSONANTS = new Set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي");
const form = (s: string) => collapseHeld([...s]).join("");

/**
 * Align individual consonants independently of vowels, using the existing costs. Hard edits sharing an original
 * contiguous run on either side count as one independent change. This keeps a doubled identity change together
 * without inventing edits when vowels split a run, or erasing separate changes on both sides of a vowel.
 */
export function hardConsonantEdits(expected: string, observed: string): number {
  const a = form(expected), b = form(observed);
  const consonants = (s: string) => {
    const out: { at: number; run: number }[] = [];
    let run = -1;
    for (let i = 0; i < s.length; i++) if (CONSONANTS.has(s[i])) {
      if (!i || s[i - 1] !== s[i]) run++;
      out.push({ at: i, run });
    }
    return out;
  };
  const r = consonants(a), o = consonants(b), width = o.length + 1;
  // Only the current reference/observed runs can receive future edits. Preserve five frontier states so a later
  // substitution can join their components before they close: none, R, O, R/O separate, R/O joined.
  const cell = (i: number, j: number, state: number) => (i * width + j) * 5 + state;
  const cost = new Float64Array((r.length + 1) * width * 5).fill(Infinity);
  const closed = new Float64Array(cost.length).fill(Infinity);
  cost[0] = closed[0] = 0;
  for (let i = 0; i <= r.length; i++) for (let j = 0; j <= o.length; j++) for (let state = 0; state < 5; state++) {
    const from = cell(i, j, state);
    if (!Number.isFinite(cost[from])) continue;
    const step = (di: number, dj: number, charge: number) => {
      let activeR = state === 1 || state >= 3, activeO = state === 2 || state >= 3;
      let joined = state === 4;
      if (charge >= UNIT) {
        if (di) activeR = true;
        if (dj) activeO = true;
        if (di && dj) joined = true;
      }
      const ni = i + di, nj = j + dj;
      const keepR = activeR && ni < r.length && r[ni].run === r[i]?.run;
      const keepO = activeO && nj < o.length && o[nj].run === o[j]?.run;
      const finished = joined ? Number(!keepR && !keepO) : Number(activeR && !keepR) + Number(activeO && !keepO);
      const next = keepR && keepO ? (joined ? 4 : 3) : keepR ? 1 : keepO ? 2 : 0;
      const to = cell(ni, nj, next), c = cost[from] + charge, h = closed[from] + finished;
      if (c < cost[to] || (c === cost[to] && h < closed[to])) { cost[to] = c; closed[to] = h; }
    };
    if (i < r.length && j < o.length) step(1, 1, subCost(a[r[i].at], b[o[j].at]));
    if (i < r.length) step(1, 0, deletionCost(a, r[i].at));
    if (j < o.length) step(0, 1, insertionCost(a, r[i]?.at ?? a.length, b[o[j].at]));
  }
  return closed[cell(r.length, o.length, 0)];
}

/** Shared by checker instances; only supported reference pronunciations enter the whole-word vocabulary. */
const cache = new WeakMap<ReferenceIndex, ReturnType<typeof build>>();
function build(index: ReferenceIndex) {
  const forms = index.words.map((w, i) => [...new Set([
    index.phonemes.slice(w.start, w.end), ...(index.wordForms?.[i] ?? []),
  ].filter(Boolean).map(form))]);
  const isolated = new Set((index.isolatedWordForms ?? []).filter(Boolean).map(form));
  const connected = new Map<string, Set<string>>();
  const sameAyah = (a: number, b: number) => b >= 0 && b < index.words.length &&
    index.words[a].surah === index.words[b].surah && index.words[a].ayah === index.words[b].ayah;
  const neighbor = (wi: number, direction: number) => {
    let at = wi + direction;
    while (sameAyah(wi, at) && index.words[at].start === index.words[at].end) at += direction;
    return sameAyah(wi, at) ? at : -1;
  };
  const word = (wi: number) => index.phonemes.slice(index.words[wi].start, index.words[wi].end);
  const boundary = (text: string, before: boolean) => before ? form(text).at(-1) : form(text)[0];
  const contextKey = (before?: string, after?: string) => `${before ?? "^"}|${after ?? "$"}`;
  index.words.forEach((w, wi) => {
    if (w.start === w.end) return;
    const before = neighbor(wi, -1), after = neighbor(wi, 1);
    const key = contextKey(before < 0 ? undefined : boundary(word(before), true),
      after < 0 ? undefined : boundary(word(after), false));
    const text = form(word(wi));
    const contexts = connected.get(text) ?? new Set<string>();
    contexts.add(key);
    connected.set(text, contexts);
  });
  const matches = (wi: number, heard: string) => forms[wi].includes(form(heard));
  return {
    matches,
    alternative(wi: number, heard: string, context?: { before?: string; after?: string }): boolean {
      const normalized = form(heard);
      const key = context && contextKey(context.before === undefined ? undefined : boundary(context.before, true),
        context.after === undefined ? undefined : boundary(context.after, false));
      const known = isolated.has(normalized) || (key !== undefined && connected.get(normalized)?.has(key));
      return !!known && !matches(wi, heard) &&
        forms[wi].length > 0 && forms[wi].every(expected => hardConsonantEdits(expected, normalized) >= 2);
    },
  };
}

export function lexicalWords(index: ReferenceIndex) {
  let words = cache.get(index);
  if (!words) { words = build(index); cache.set(index, words); }
  return words;
}
