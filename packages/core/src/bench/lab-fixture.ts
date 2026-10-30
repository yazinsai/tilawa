/** Synthetic spliced-style sessions for the lab scorer's tests and its scorer fixture (`lab-fixture-cli.ts`). */
import type { WordRef } from "../events.js";
import type { ReferenceIndex } from "../reference.js";
import { editDistance, refIndex } from "./lab.js";
import { validateSession } from "./sessions.js";
import type { BenchSession } from "./sessions.js";

export interface FixturePart {
  kind?: "recitation" | "repeat" | "restart"; from: WordRef; to: WordRef;
  wrong?: { at: WordRef; said: WordRef; mutashabihat?: boolean }[]; omit?: WordRef[]; skip?: [WordRef, WordRef][];
  add?: { before: WordRef; said: WordRef }[]; pauseAfter?: number;
}
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const r3 = (x: number) => Math.round(x * 1000) / 1000;
const GAP_S = 0.05;
const refAt = (index: ReferenceIndex, i: number): WordRef => ({ surah: index.words[i].surah, ayah: index.words[i].ayah, word: index.words[i].word });
export const wordPhonemes = (index: ReferenceIndex, r: WordRef) => { const w = index.words[refIndex(index)(r)]; return index.phonemes.slice(w.start, w.end); };
export function lastWordOf(index: ReferenceIndex, surah: number, ayah: number): WordRef {
  let i = refIndex(index)({ surah, ayah, word: 1 });
  while (index.words[i + 1]?.surah === surah && index.words[i + 1].ayah === ayah) i++;
  return refAt(index, i);
}
/** A word from Al-Baqarah about as long as `at` and mostly different from it: a clear wrong word. */
export function donorWord(index: ReferenceIndex, at: WordRef): WordRef {
  const p = wordPhonemes(index, at);
  const i = index.words.findIndex(w => w.surah === 2 && Math.abs(w.end - w.start - p.length) <= 1 && editDistance(index.phonemes.slice(w.start, w.end), p) >= 0.7 * p.length);
  if (i < 0) throw new Error(`no donor for ${key(at)}`);
  return refAt(index, i);
}

/** Words timed by phoneme count (80 ms each, at least 250 ms, 50 ms apart), planted edits applied, repeats and restarts as non-errors. */
export function buildSession(index: ReferenceIndex, id: string, parts: FixturePart[], speaker = "r1"): BenchSession {
  const ix = refIndex(index), phon = (r: WordRef) => wordPhonemes(index, r);
  const s: BenchSession = { version: 1, id, track: "recitation", source: "recorded", speaker, audio: `${id}.wav`, durationS: 0,
    context: { kind: "none" }, spans: [], words: [], events: [], nonErrors: [] };
  const dur = (p: string) => r3(Math.max(0.25, p.length * 0.08));
  let t = 1;
  parts.forEach((part, n) => {
    const span = `p${n}`, kind = part.kind ?? "recitation", start = t;
    const skipped = new Map<number, number>();
    for (const [a, b] of part.skip ?? []) for (let i = ix(a); i <= ix(b); i++) skipped.set(i, ix(b));
    const omitted = new Set((part.omit ?? []).map(ix));
    for (let i = ix(part.from); i <= ix(part.to); i++) {
      const at = refAt(index, i);
      if (index.words[i].end === index.words[i].start) continue;
      const gap = { span, location: at, state: "confirmed", startS: r3(t), endS: r3(t + GAP_S) };
      if (omitted.has(i)) { s.events.push({ ...gap, errorClass: "omitted_word" }); continue; }
      if (skipped.has(i)) { if (!skipped.has(i - 1)) s.events.push({ ...gap, errorClass: "skipped_passage", through: refAt(index, skipped.get(i)!) }); continue; }
      const add = part.add?.find(x => key(x.before) === key(at));
      if (add) {
        const p = phon(add.said);
        s.events.push({ span, location: at, errorClass: "added_word", state: "confirmed", startS: r3(t), endS: r3(t + dur(p)), said: p });
        t = r3(t + dur(p) + GAP_S);
      }
      const wrong = part.wrong?.find(x => key(x.at) === key(at)), p = wrong ? phon(wrong.said) : phon(at);
      const iv = { startS: r3(t), endS: r3(t + dur(p)) };
      s.words.push({ span, at, ...iv });
      if (wrong) s.events.push({ span, location: at, errorClass: "wrong_word", state: "confirmed", ...iv, said: p, ...(wrong.mutashabihat ? { forms_other_ayah: true } : {}) });
      if (kind !== "recitation") s.nonErrors.push({ span, at, kind, ...iv });
      t = r3(t + dur(p) + GAP_S);
    }
    s.spans.push({ id: span, kind, from: part.from, to: part.to, startS: start, endS: r3(t - GAP_S) });
    t = r3(t + (part.pauseAfter ?? 0.6));
  });
  s.durationS = r3(t + 1);
  validateSession(s, index);
  return s;
}

/** The same session as heard by a listener that misses every planted wrong word (decodes the written word). */
export const deafToWrongWords = (s: BenchSession, index: ReferenceIndex): BenchSession =>
  ({ ...s, events: s.events.map(e => e.errorClass === "wrong_word" ? { ...e, said: wordPhonemes(index, e.location) } : e) });
