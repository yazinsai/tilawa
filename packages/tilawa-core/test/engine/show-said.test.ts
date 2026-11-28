import { describe, expect, it } from "vitest";
import type { WordRef } from "../../src/engine/events.js";
import { CONSONANTS } from "../../src/engine/fuzzy.js";
import {
  ALL, DEFAULT, FRAME_MS, RC1, at, ayah, ayahs, begin, breakAt, drawn, index, key, noise, passage, pause, recite, say, shownAt, speech,
  spread, without,
} from "./takes.js";
import type { Piece, Speech, Take } from "./takes.js";

/**
 * refinements.showSaid: a word lights once it has been said (`Tracker.shown()`), not on its first sound: a word inside
 * an ayah once the next word is under way, an ayah's last word once heard as written through its last phoneme (a final
 * held vowel's first phoneme); at a pause or the take's end, the word stopped on once its consonants are heard. The
 * positions `push()` returns, which the checker reads, are unchanged.
 * Each take runs at 1 and 3 phonemes a push. The takes are synthetic phoneme streams built from the reference
 * (`./takes.ts`).
 */

const OFF = without("showSaid");
const STEPS = [1, 3];
const word = (k: string): WordRef => {
  const [surah, ayah_, w] = k.split(":").map(Number);
  return { surah, ayah: ayah_, word: w };
};
/**
 * When each word said was heard to its end: the start of its last phoneme, or of a final held vowel's first. Only the
 * words said as written and whole (each word's first saying).
 */
function finishedAt(s: Speech): Map<string, number> {
  const out = new Map<string, number>();
  for (const [k, ms] of s.saidAt) {
    const w = index.words[at(word(k))];
    const first = s.phonemes.findIndex((p) => p.start * FRAME_MS >= ms);
    const sounds = index.phonemes.slice(w.start, w.end);
    const heard = s.phonemes.slice(first, first + sounds.length).map((p) => p.p).join("");
    if (heard !== sounds) continue;
    const last = sounds.at(-1)!;
    let k0 = sounds.length - 1;
    while (k0 > 0 && sounds[k0 - 1] === last) k0--;
    const need = sounds.length - k0 >= 2 && !CONSONANTS.has(last) ? k0 + 1 : sounds.length;
    out.set(k, s.phonemes[first + need - 1].start * FRAME_MS);
  }
  return out;
}
/** The words lit before they were heard to their end, as "key(ms early)". */
const unfinished = (s: Speech, t: Take) => {
  const lit = shownAt(t.shown);
  return [...finishedAt(s)].flatMap(([k, ms]) => (lit.has(k) && lit.get(k)! < ms ? [`${k}(${ms - lit.get(k)!})`] : []));
};
/**
 * The words said that the page shows dark at the take's end, but those it shows dark with the field off too (a free
 * recitation's opening basmala, read as a phrase).
 */
const darkSaid = (s: Speech, t: Take, off?: Take) =>
  [...s.saidAt.keys()].filter((k) => !drawn(t.shown).has(k) && (!off || drawn(off.shown).has(k)));
/**
 * The words lit later than the word after them is begun and heard for `ms` more (or than the field off lights them,
 * if later: a free recitation's first lock comes once its place is found), as "key(ms late)"; those the field off
 * never lights are `darkSaid`'s.
 */
const late = (s: Speech, on: Take, off: Take, ms: number) => {
  const [lit, raw] = [shownAt(on.shown), shownAt(off.shown)];
  const said = [...s.saidAt].sort((a, b) => a[1] - b[1]);
  return said.slice(0, -1).flatMap(([k], i) => {
    if (!raw.has(k)) return [];
    const by = Math.max(said[i + 1][1] + ms, raw.get(k)!);
    return (lit.get(k) ?? Infinity) > by ? [`${k}(${(lit.get(k) ?? Infinity) - by})`] : [];
  });
};
/**
 * Each ayah's last word said (but the take's last) not lit before the next ayah's first word is said, nor when the field
 * off lights it (a lock that comes later), as keys; those the field off never lights are `darkSaid`'s.
 */
const darkThroughPause = (s: Speech, on: Take, off: Take) => {
  const [lit, raw] = [shownAt(on.shown), shownAt(off.shown)];
  const said = [...s.saidAt].sort((a, b) => a[1] - b[1]);
  return said.slice(0, -1).flatMap(([k], i) => {
    const [w, next] = [word(k), word(said[i + 1][0])];
    return raw.has(k) && w.ayah !== next.ayah && !((lit.get(k) ?? Infinity) <= Math.max(said[i + 1][1], raw.get(k)!)) ? [k] : [];
  });
};
/** The words said, in the order said. */
const order = (s: Speech) => [...s.saidAt].sort((a, b) => a[1] - b[1]).map(([k]) => k);
/**
 * Each ayah's first word said not shown before the next word is (alone), the take's first included, but those the
 * field off does not show alone either (`off`: a free recitation's first lock comes once its place is found, words into
 * the take, and lights them together).
 */
const notAlone = (s: Speech, t: Take, off?: Take) => {
  const said = order(s);
  const opens = (k: string) => key(index.words[ayah(word(k).surah, word(k).ayah)[0]]) === k;
  const alone = (lit: Map<string, number>, k: string, i: number) => (lit.get(k) ?? Infinity) < (lit.get(said[i + 1]) ?? Infinity);
  const [on, raw] = [shownAt(t.shown), off && shownAt(off.shown)];
  return said.slice(0, -1).filter((k, i) => (i === 0 || opens(k)) && !alone(on, k, i) && (!raw || alone(raw, k, i)));
};
/** The field changes what shows and nothing else: the positions the checker reads, the checker's events and marks. */
const sameUnderneath = (on: Take, off: Take, mode: string) => {
  expect(on.positions, `${mode}: positions`).toEqual(off.positions);
  expect(on.checker, `${mode}: checker events`).toEqual(off.checker);
  expect(on.mistakes, `${mode}: marks`).toEqual(off.mistakes);
};
/** Lit late by at most this: three phonemes of the next word (a phoneme is 120 ms) and a push of three more. */
const SLACK_MS = 6 * 3 * FRAME_MS;

/** Every flow property of a correct take, on and off, declared and free. */
function correct(take: Speech, p: [WordRef, WordRef], step: number, name: string) {
  for (const declared of [p, null]) {
    const mode = `${name}, ${declared ? "declared" : "free"}, step ${step}`;
    const on = recite(take, { passage: declared, step, refinements: ALL });
    const off = recite(take, { passage: declared, step, refinements: OFF });
    expect(unfinished(take, on), `${mode}: lit before finished`).toEqual([]);
    expect(darkSaid(take, on, off), `${mode}: said, dark`).toEqual([]);
    expect(notAlone(take, on, declared ? undefined : off), `${mode}: an ayah's first word not shown alone`).toEqual([]);
    expect(late(take, on, off, SLACK_MS), `${mode}: lit late`).toEqual([]);
    expect(darkThroughPause(take, on, off), `${mode}: an ayah's last word dark until the next ayah`).toEqual([]);
    sameUnderneath(on, off, mode);
  }
}

describe("refinements.showSaid: a correct recitation", () => {
  it("lights each word once said, never before, none left dark, each ayah's first word alone", () => {
    for (const [s, a, b] of [[67, 1, 4], [18, 9, 12], [2, 2, 5], [1, 1, 7], [112, 1, 4], [55, 10, 14]] as const) {
      for (const step of STEPS) correct(speech(ayahs(s, a, b)), passage(s, a, b), step, `${s}:${a}–${b}`);
    }
  }, 300_000);

  it("at a varied pace, each ayah's first sound heard softly", () => {
    // 60 to 150 ms a phoneme, 8 to 25 frames between ayahs, and every ayah's first phoneme heard at 0.4 to 0.7.
    for (const [s, a, b] of [[67, 1, 4], [36, 1, 6], [2, 1, 4], [26, 67, 70], [82, 1, 5], [3, 1, 4]] as const) {
      for (const step of STEPS) {
        const [ms, gap, soft] = [spread(s * 1000 + a + step, 60, 150), spread(s + b + step, 8, 25), spread(a + b + step, 0.4, 0.7)];
        const pieces: Piece[] = [];
        for (let x = a; x <= b; x++) pieces.push(say(s, x), pause(Math.round(gap())));
        const take = speech(pieces, { ms, opening: soft });
        for (const declared of [passage(s, a, b), null]) {
          const mode = `${s}:${a}–${b}, ${declared ? "declared" : "free"}, step ${step}`;
          const t = recite(take, { passage: declared, step, refinements: ALL });
          const off = recite(take, { passage: declared, step, refinements: OFF });
          expect(unfinished(take, t), `${mode}: lit before finished`).toEqual([]);
          expect(darkSaid(take, t, off), `${mode}: said, dark`).toEqual([]);
          expect(notAlone(take, t, declared ? undefined : off), `${mode}: not alone`).toEqual([]);
        }
      }
    }
  }, 300_000);

  it("the take's first word lights once said, before the second word", () => {
    for (const [s, a, b] of [[67, 1, 2], [18, 10, 11], [2, 255, 255], [36, 1, 3]] as const) {
      const take = speech(ayahs(s, a, b));
      const [first, second] = order(take);
      for (const step of STEPS) {
        const t = recite(take, { passage: passage(s, a, b), step, refinements: ALL });
        const lit = shownAt(t.shown);
        const mode = `${s}:${a}, step ${step}`;
        expect(lit.get(first), `${mode}: once said`).toBeGreaterThanOrEqual(finishedAt(take).get(first)!);
        expect(lit.get(first)!, `${mode}: before the second word`).toBeLessThan(lit.get(second)!);
      }
    }
  });

  it("short words, the muqatta'at and ayahs of one word", () => {
    // 112:1 قُل and 108:1's إِنَّا (two-sound and short words), 2:1 الٓمٓ, 36:1 يسٓ, 19:1 كٓهيعٓصٓ, 42:1–2, 50:1 قٓ,
    // 68:1 نٓ, 20:1 طه: an ayah of one word lights once said, before the next ayah's first word.
    for (const [s, a, b] of [[112, 1, 2], [108, 1, 2], [2, 1, 3], [36, 1, 3], [19, 1, 3], [42, 1, 3], [50, 1, 2], [68, 1, 2], [20, 1, 3]] as const) {
      for (const step of STEPS) correct(speech(ayahs(s, a, b)), passage(s, a, b), step, `${s}:${a}–${b}`);
    }
  }, 300_000);

  it("words whose last sounds join the next word's (idgham, iqlab, a joined hamzat al-wasl)", () => {
    // 2:5 هُدًى مِّن رَّبِّهِمْ (the nūn and tanwīn merged into the next word's doubled onset), 2:27 مِنۢ بَعْدِ (iqlab),
    // 1:2 and 67:1 (a word ending before a joined hamzat al-wasl), 2:2 هُدًى لِّلْمُتَّقِينَ.
    for (const [s, a, b] of [[2, 5, 6], [2, 27, 28], [1, 2, 4], [67, 1, 2], [2, 2, 3]] as const) {
      for (const step of STEPS) correct(speech(ayahs(s, a, b)), passage(s, a, b), step, `${s}:${a}–${b}`);
    }
  }, 300_000);

  it("ayahs that open with the same words light as without the field", () => {
    // The ayah next in order lights as the shared words are said, and the one said where they part.
    for (const [s, a, b] of [[67, 15, 18], [67, 22, 25], [55, 12, 16]] as const) {
      const take = speech(ayahs(s, a, b));
      for (const step of STEPS) {
        const on = recite(take, { passage: passage(s, a, b), step, refinements: ALL });
        const off = recite(take, { passage: passage(s, a, b), step, refinements: OFF });
        const mode = `${s}:${a}–${b}, step ${step}`;
        expect([...drawn(on.shown).keys()].sort(), `${mode}: lit at the end`).toEqual([...drawn(off.shown).keys()].sort());
        expect(darkSaid(take, on), `${mode}: said, dark`).toEqual([]);
        sameUnderneath(on, off, mode);
      }
    }
  });
});

describe("refinements.showSaid: a pause inside a word", () => {
  it("the word waits for the rest of it, and shows once said", () => {
    // A pause of 15 to 30 frames inside the third word of each ayah after its first phoneme, and inside the fifth
    // after its first two (a breath, a hesitation).
    for (const [s, a, b] of [[67, 1, 3], [18, 9, 11], [2, 2, 4], [36, 1, 5], [1, 2, 7]] as const) {
      for (const step of STEPS) {
        const gap = spread(s * 7 + a + step, 15, 30);
        let take = speech(ayahs(s, a, b));
        for (let x = a; x <= b; x++) {
          const words = ayah(s, x).length;
          if (words >= 3) take = breakAt(take, `${s}:${x}:3`, 1, Math.round(gap()));
          if (words >= 5) take = breakAt(take, `${s}:${x}:5`, 2, Math.round(gap()));
        }
        for (const declared of [passage(s, a, b), null]) {
          const mode = `${s}:${a}–${b}, ${declared ? "declared" : "free"}, step ${step}`;
          const t = recite(take, { passage: declared, step, refinements: ALL });
          expect(unfinished(take, t), `${mode}: lit before finished`).toEqual([]);
          expect(darkSaid(take, t), `${mode}: said, dark`).toEqual([]);
        }
      }
    }
  }, 300_000);

  it("a reciter who stops on a word after its last consonant is shown it at the take's end", () => {
    // 67:2's last word said without its final vowel hold, then the take ends.
    const w = index.words[ayah(67, 2).at(-1)!];
    const sounds = index.phonemes.slice(w.start, w.end);
    const cut = [...sounds].findLastIndex((p) => CONSONANTS.has(p)) + 1;
    const take = speech([say(67, 1), pause(15), say(67, 2, 1, ayah(67, 2).length - 1), { said: sounds.slice(0, cut) }]);
    for (const step of STEPS) {
      const t = recite(take, { passage: passage(67, 1, 2), step, refinements: ALL });
      expect(drawn(t.shown).has(key(w)), `step ${step}`).toBe(true);
    }
  });
});

describe("refinements.showSaid: a word begun, then skipped", () => {
  // The word's first one or two phonemes, then a skip forward: it was never said, so it never lights.
  const takes: [string, string, (n: number) => Piece[], [WordRef, WordRef]][] = [
    ["18:10:1 begun, then 18:11", "18:10:1",
      (n) => [say(18, 9), pause(15), begin(18, 10, 1, n), pause(15), say(18, 11), pause(15), say(18, 12)], passage(18, 9, 12)],
    ["67:2:5 begun, then 67:2:7", "67:2:5",
      (n) => [say(67, 1), pause(15), say(67, 2, 1, 4), begin(67, 2, 5, n), pause(10), say(67, 2, 7), pause(15), say(67, 3)], passage(67, 1, 3)],
    ["2:3:3 begun, then 2:4", "2:3:3",
      (n) => [say(2, 2), pause(15), say(2, 3, 1, 2), begin(2, 3, 3, n), pause(15), say(2, 4), pause(15), say(2, 5)], passage(2, 2, 5)],
  ];
  for (const [name, begun, pieces, p] of takes) {
    it(`${name}: never lit, with withdrawBegun off`, () => {
      for (const n of [1, 2]) {
        const take = speech(pieces(n));
        for (const step of STEPS) {
          const mode = `${n} phonemes, step ${step}`;
          const on = recite(take, { passage: p, step, refinements: { ...ALL, withdrawBegun: false } });
          const off = recite(take, { passage: p, step, refinements: { ...OFF, withdrawBegun: false } });
          expect(shownAt(off.shown).has(begun), `${mode}: lit with the field off`).toBe(true);
          expect(shownAt(on.shown).has(begun), `${mode}: lit`).toBe(false);
          expect(darkSaid(take, on), `${mode}: said, dark`).toEqual([]);
          sameUnderneath(on, off, mode);
        }
      }
    });
  }

  it("a word left out whole, then the next only begun or said, then a stop: the words said show, the one left out never", () => {
    const takes: [Piece[], [WordRef, WordRef], string[], string[]][] = [
      [[say(67, 17), pause(15), say(67, 18, 1, 5), begin(67, 18, 7, 2)], passage(67, 17, 19), ["67:18:5"], ["67:18:6", "67:18:7"]],
      [[say(67, 17), pause(15), say(67, 18, 1, 5), say(67, 18, 7, 7)], passage(67, 17, 19), ["67:18:5", "67:18:7"], ["67:18:6"]],
      [[say(67, 17), pause(15), say(67, 18, 1, 5), begin(67, 18, 7, 2), pause(15), say(67, 18, 8)], passage(67, 17, 19),
        ["67:18:5", "67:18:8"], ["67:18:6", "67:18:7"]],
      [[say(67, 21), pause(15), say(67, 22, 1, 1), begin(67, 22, 5, 2)], passage(67, 21, 23), ["67:22:1"],
        ["67:22:2", "67:22:3", "67:22:4", "67:22:5"]],
    ];
    for (const [pieces, p, lit, dark] of takes) {
      const take = speech(pieces);
      for (const step of STEPS) {
        const on = recite(take, { passage: p, step, refinements: ALL });
        const shown = shownAt(on.shown);
        const mode = `${dark[0]} left out, step ${step}`;
        for (const k of lit) expect(shown.has(k), `${mode}: ${k} lit`).toBe(true);
        for (const k of dark) expect(shown.has(k), `${mode}: ${k} dark`).toBe(false);
        sameUnderneath(on, recite(take, { passage: p, step, refinements: OFF }), mode);
      }
    }
  });
});

describe("refinements.showSaid: what shows before the place is found", () => {
  it("is what push() reports, with noLostBeforeLock's report of a place not found yet", () => {
    // Speech that is not the passage first: noLostBeforeLock reports "acquiring" where the search says "lost".
    const take = speech([noise("زظغذثخ"), pause(30), noise("طضصق"), pause(30), say(36, 6), pause(15), say(36, 7)]);
    for (const step of STEPS) {
      for (const refinements of [ALL, OFF]) {
        const t = recite(take, { passage: passage(36, 6, 7), step, refinements });
        const first = (ps: Take["shown"]) => ps.findIndex((p) => p.state === "locked");
        const mode = `step ${step}, ${refinements.showSaid ? "on" : "off"}`;
        expect(first(t.shown), mode).toBeGreaterThan(0);
        expect(t.shown.slice(0, first(t.shown)), mode).toEqual(t.positions.slice(0, first(t.positions)));
        expect(t.shown.slice(0, first(t.shown)).map((p) => p.state), mode).not.toContain("lost");
      }
    }
  });
});

describe("refinements.showSaid off", () => {
  it("shows the positions push() and end() return, as rc.1 does", () => {
    for (const [s, a, b] of [[67, 1, 3], [2, 1, 4]] as const) {
      const take = speech(ayahs(s, a, b));
      for (const step of STEPS) {
        for (const refinements of [OFF, RC1]) {
          const t = recite(take, { passage: passage(s, a, b), step, refinements });
          expect(t.shown, `${s}:${a}, step ${step}`).toEqual(t.positions);
        }
      }
    }
  });

  it("is on by default", () => {
    expect(DEFAULT.showSaid).toBe(true);
    const take = speech(ayahs(67, 1, 2));
    const t = recite(take, { passage: passage(67, 1, 2) });
    expect(unfinished(take, t)).toEqual([]);
    expect(t.shown).not.toEqual(t.positions);
  });
});
