import { describe, expect, it } from "vitest";
import { CONSONANTS } from "../../src/engine/fuzzy.js";
import { DEFAULT, ayah, begin, drawn, index, key, passage, pause, recite, ref, say, speech, withdrawals } from "./takes.js";
import type { Fields, Piece } from "./takes.js";

/**
 * refinements.begunByConsonants judges the word a jump leaves by the tracker's one test of a word said (showSaid's): a
 * word said in full whose consonant the decoder heard as its engine-confusable partner, or did not hear, is said, so a
 * skip of the next word (the commonest memorisation slip) leaves it shown; a word left after its first syllable is
 * not. Each take runs at 1 and 3 phonemes a push, with the default policy and with the earlier defaults
 * (`declaredStart` and `withdrawBegun` off), beside the same take said cleanly. The takes are synthetic phoneme streams
 * built from the reference (`./takes.ts`).
 */

const STEPS = [1, 3];
const EARLIER_DEFAULTS: Fields = { ...DEFAULT, declaredStart: false, withdrawBegun: false };

describe("refinements.begunByConsonants: the word a jump leaves, judged by the tracker's test of a word said", () => {
  it("stays lit when said in full with one consonant misheard or not heard, then the next word left out (وَٱلْحَيَوٰةَ)", () => {
    // 67:1-3 declared; 67:2:4 said in full but heard otherwise; 67:2:5 left out; the take goes on at 67:2:6. Before
    // this rule the default policy withdrew 67:2:4 about 0.9 s after it lit and left it dark.
    const x = index.words[ayah(67, 2)[3]];
    const written = index.phonemes.slice(x.start, x.end);
    const heard: [string, Piece][] = [
      ["ح heard as ه", { said: written.replace("ح", "ه"), conf: 0.96 }],
      ["ت not heard", { said: written.replace("ت", ""), conf: 0.96 }],
      ["said cleanly", say(67, 2, 4, 4)],
    ];
    for (const [how, word] of heard) {
      const take = speech([say(67, 1), pause(15), say(67, 2, 1, 3), word, say(67, 2, 6), pause(15), say(67, 3)]);
      for (const step of STEPS) {
        for (const [mode, refinements] of [["the default policy", DEFAULT], ["the earlier defaults", EARLIER_DEFAULTS]] as const) {
          const t = recite(take, { passage: passage(67, 1, 3), step, refinements });
          const name = `${how}, step ${step}, ${mode}`;
          expect(withdrawals(t.shown).filter((w) => w.startsWith("67:2:4@")), `${name}: withdrawn`).toEqual([]);
          expect(drawn(t.shown).has("67:2:4"), `${name}: 67:2:4 lit at the end`).toBe(true);
          expect(t.mistakes, `${name}: marks`).toEqual(["omitted_word 67:2:5"]);
        }
      }
    }
  });

  it("leaves no more such words dark than the earlier defaults do, through the mushaf (every fourth place)", async () => {
    // In 8 ayahs at a fixed stride through the text, word 3 is said in full with its last consonant swapped for a usual
    // confusion (an engine-confusable partner or not), then word 4 is left out. The earlier defaults never light some
    // of these short misheard words at all; the default policy may darken no more than they leave dark. The same word
    // said cleanly is lit at the end.
    const swap: Record<string, string> = { "ح": "ه", "ه": "ح", "ق": "ك", "ك": "ق", "ت": "ط", "ط": "ت", "س": "ص", "ص": "س", "د": "ض", "ض": "د", "ذ": "ز", "ز": "ذ", "ع": "ء", "ء": "ع", "ث": "س", "ظ": "ذ", "ب": "م", "م": "ن", "ن": "م", "ر": "ل", "ل": "ر", "ف": "ث", "ج": "ش", "ش": "ج", "ي": "ء", "و": "ء", "خ": "غ", "غ": "خ" };
    const ayahs: [number, number][] = [];
    for (const w of index.words) if (w.word === 1 && w.end > w.start) ayahs.push([w.surah, w.ayah]);
    const places: { word: string; take: Piece[]; clean: Piece[]; p: ReturnType<typeof passage> }[] = [];
    let found = 0;
    for (let j = 7; j < ayahs.length && places.length < 8; j += 151) {
      const [s, a] = ayahs[j];
      const ws = ayah(s, a);
      if (ws.length < 6 || !ayah(s, a + 1).length || !ayah(s, a - 1).length) continue;
      const x = index.words[ws[2]];
      const ph = [...index.phonemes.slice(x.start, x.end)];
      let at = -1;
      for (let q = ph.length - 1; q >= 0; q--) if (CONSONANTS.has(ph[q]) && swap[ph[q]]) { at = q; break; }
      if (at < 1) continue;
      if (found++ % 4) continue;
      ph[at] = swap[ph[at]];
      const lead = [say(s, a - 1), pause(15), say(s, a, 1, 2)];
      const rest = [say(s, a, 5), pause(15), say(s, a + 1)];
      places.push({ word: key(ref(ws[2])), take: [...lead, { said: ph.join(""), conf: 0.96 }, ...rest], clean: [...lead, say(s, a, 3, 3), ...rest], p: passage(s, a - 1, a + 1) });
    }
    expect(places.length).toBe(8);
    for (const step of STEPS) {
      const dark = { defaults: [] as string[], earlier: [] as string[], clean: [] as string[] };
      for (const { word, take, clean, p } of places) {
        if (!drawn(recite(speech(take), { passage: p, step, refinements: DEFAULT, checker: false }).shown).has(word)) dark.defaults.push(word);
        if (!drawn(recite(speech(take), { passage: p, step, refinements: EARLIER_DEFAULTS, checker: false }).shown).has(word)) dark.earlier.push(word);
        if (!drawn(recite(speech(clean), { passage: p, step, refinements: DEFAULT, checker: false }).shown).has(word)) dark.clean.push(word);
        // A long scan: let the test runner's worker answer between takes.
        await new Promise((done) => setImmediate(done));
      }
      expect(dark.defaults.filter((w) => !dark.earlier.includes(w)), `step ${step}: dark with the defaults, lit with the earlier defaults`).toEqual([]);
      expect(dark.clean, `step ${step}: said cleanly, dark`).toEqual([]);
    }
  });

  it("still withdraws a word left after its first syllable (لِيَ of لِيَبْلُوَكُمْ), beside the take said in full", () => {
    // The case the switch was made for: لِيَبْلُوَكُمْ cut after لِيَ, then the take goes on at 67:2:7.
    const cut = speech([say(67, 1), pause(15), say(67, 2, 1, 4), begin(67, 2, 5, 3), pause(10), say(67, 2, 7), pause(15), say(67, 3)]);
    const full = speech([say(67, 1), pause(15), say(67, 2, 1, 5), say(67, 2, 7), pause(15), say(67, 3)]);
    for (const step of STEPS) {
      const t = recite(cut, { passage: passage(67, 1, 3), step, refinements: DEFAULT });
      expect(drawn(t.shown).has("67:2:5"), `step ${step}: لِيَ, lit at the end`).toBe(false);
      const c = recite(full, { passage: passage(67, 1, 3), step, refinements: DEFAULT });
      expect(drawn(c.shown).has("67:2:5"), `step ${step}: said in full, lit at the end`).toBe(true);
    }
  });
});
