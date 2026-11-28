import { describe, expect, it } from "vitest";
import type { PositionEvent } from "../../src/engine/events.js";
import {
  ALL, EARLIER, EARLIER_ON, FRAME_MS, RC1, at, ayah, ayahs, decided, earlierWithout, index, key, noise, only, passage, pause, phrases, reached, recite, ref,
  say, shownAt, speech, spread, without,
} from "./takes.js";
import type { Fields, Piece, Speech, Take } from "./takes.js";

/**
 * The refinements of rc.1's rules (`refinements` in the tracker's and checker's options, `policy.refinements` in a
 * session), each against the rc.1 rule it replaces. The takes are synthetic phoneme streams built from the reference
 * (`./takes.ts`). Each correction has a test of what it fixes, with that field `false` (rc.1's rule: the defect shows)
 * and with every correction on (`ALL`: fixed), and a test of the case its risk names.
 */

describe("refinements, control", () => {
  /** The take marks nothing, never loses its place, and shows every word of `words`, in order, none before it is said. */
  const silentInOrder = (take: Take, s: Speech, words: number[], label = "") => {
    expect(take.checker, `${label} silent`).toEqual([]);
    expect(take.positions.filter((p) => p.state === "lost"), `${label} never lost`).toEqual([]);
    const shown = shownAt(take.positions);
    let previous = 0;
    for (const wi of words) {
      const word = key(ref(wi));
      const time = shown.get(word);
      expect(time, `${label} ${word} shown`).toBeDefined();
      expect(time!, `${label} ${word} in order`).toBeGreaterThanOrEqual(previous);
      expect(time!, `${label} ${word} not before it is said`).toBeGreaterThanOrEqual(s.saidAt.get(word)!);
      previous = time!;
    }
  };

  it("a correct recitation stays silent and shows every word, in order, none before it is said", () => {
    const s = speech([say(67, 1), pause(15), say(67, 2), pause(15), say(67, 3)]);
    silentInOrder(recite(s, { passage: passage(67, 1, 3), refinements: ALL }), s, [...ayah(67, 1), ...ayah(67, 2), ...ayah(67, 3)]);
  });

  it("the same at a varied pace, with each ayah's first sound heard softly, three phonemes a push", () => {
    // 60 to 150 ms a phoneme, 8 to 25 frames between ayahs, and every ayah's first phoneme heard at 0.4 to 0.7: below
    // the tracker's breath cut, so each opening is a risky step there (refinements.confirmAdvance).
    for (const [su, a, b] of [[67, 1, 5], [55, 1, 13], [36, 1, 12], [2, 1, 5]] as const) {
      for (const declared of [true, false]) {
        const [ms, gap, soft] = [spread(su * 1000 + a, 60, 150), spread(su + b, 8, 25), spread(a + b, 0.4, 0.7)];
        const pieces: Piece[] = [];
        for (let x = a; x <= b; x++) pieces.push(say(su, x), pause(Math.round(gap())));
        const s = speech(pieces, { ms, opening: soft });
        const words = Array.from({ length: b - a + 1 }, (_, i) => ayah(su, a + i)).flat();
        silentInOrder(recite(s, { passage: declared ? passage(su, a, b) : null, refinements: ALL }), s, words, `${su}:${a}-${b} ${declared ? "declared" : "free"}`);
      }
    }
  });
});

describe("refinements.readNotSkipped", () => {
  // 59:23 said whole twice, then its first eight words and on to 59:24.
  const repeat = speech([say(59, 23), pause(20), say(59, 23), pause(20), say(59, 23, 1, 8), pause(10), say(59, 24)]);

  it("a repeat that rejoins where the reciter had been is not a skip", () => {
    expect(recite(repeat, { passage: passage(59, 23, 24), refinements: without("readNotSkipped") }).mistakes).toEqual(["skipped_passage 59:23:9"]);
    expect(recite(repeat, { passage: passage(59, 23, 24), refinements: ALL }).mistakes).toEqual([]);
  });

  it("a word heard only at the edge of a pass that breaks off is not read: leaving it out later is caught", () => {
    // Two passes break off right after 59:23:9 and go back; the third leaves 59:23:9 out.
    const s = speech([say(59, 22), pause(15), say(59, 23, 1, 9), pause(20), say(59, 23, 1, 9), pause(20), say(59, 23, 1, 8),
      say(59, 23, 10), pause(15), say(59, 24)]);
    expect(recite(s, { passage: passage(59, 22, 24), refinements: ALL }).mistakes).toEqual(["omitted_word 59:23:9"]);
  });
});

describe("refinements.reanchorBack", () => {
  it("follows a reciter who restarts the passage from behind the alignment window", () => {
    const s = speech([say(67, 1), pause(15), say(67, 2), pause(15), say(67, 3), pause(15), say(67, 4), pause(30),
      say(67, 1), pause(15), say(67, 2), pause(15), say(67, 3), pause(15), say(67, 4), pause(15), say(67, 5)]);
    // `commitAfterJump` also mends this restart: without both, it is marked.
    expect(recite(s, { passage: passage(67, 1, 5), refinements: earlierWithout("reanchorBack") }).mistakes.length).toBeGreaterThan(0);
    expect(recite(s, { passage: passage(67, 1, 5), refinements: EARLIER_ON }).mistakes).toEqual([]);
    expect(recite(s, { passage: passage(67, 1, 5), refinements: ALL }).mistakes).toEqual([]);
  });

  it("a lock shorter than JUMP_WORDS behind the window does not move it: a swap for earlier words stays caught", () => {
    // 67:5:2–3 (or 2–4) said as 67:1's first words, and the tracker locks there for that long.
    for (const n of [2, 3]) {
      const head = speech([say(67, 1), pause(15), say(67, 2), pause(15), say(67, 3), pause(15), say(67, 4), pause(15), say(67, 5, 1, 1)]);
      const s = speech([say(67, 1), pause(15), say(67, 2), pause(15), say(67, 3), pause(15), say(67, 4), pause(15), say(67, 5, 1, 1),
        say(67, 1, 1, n), say(67, 5, n + 2)]);
      const swapEnd = head.phonemes.length + speech([say(67, 1, 1, n)]).phonemes.length;
      const ws = ayah(67, 1);
      let injected = false;
      const take = recite(s, {
        passage: passage(67, 1, 5),
        refinements: ALL,
        inject: (pushed, events) => {
          if (injected || pushed < swapEnd) return events;
          injected = true;
          const end = ref(ws[n - 1]);
          const lock: PositionEvent = {
            type: "position", v: 1, timeMs: s.phonemes[pushed - 1].end * FRAME_MS, referenceVersion: index.version, expected: end,
            observed: [{ word: end, span: { start: ref(ws[0]), end }, confidence: 0.8, cost: 0.1, per: 0.1 }], state: "locked",
          };
          return [...events, lock];
        },
      });
      expect(injected).toBe(true);
      expect(take.mistakes).toEqual(["wrong_word 67:5:2"]);
    }
  });
});

describe("refinements.stutter", () => {
  const take = (word: string, refinements: Fields = ALL) =>
    recite(speech([say(18, 12), pause(15), say(18, 13, 1, 2), noise(word), say(18, 13, 4), pause(15), say(18, 14)]), { passage: passage(18, 12, 14), refinements });

  it("a word said with a block of sounds repeated is not a wrong word", () => {
    // عَلَيكَ said عَلَيكَيكَ.
    expect(take("عَلَيكَيكَ", without("stutter")).mistakes).toEqual(["wrong_word 18:13:3"]);
    expect(take("عَلَيكَيكَ").mistakes).toEqual([]);
  });

  it("another word with a repeated block is still a wrong word", () => {
    // عَلَيهِ (a different skeleton) and ءِلَيكَ (a different word), each with a block said twice.
    expect(take("عَلَيهِيهِ").mistakes).toEqual(["wrong_word 18:13:3"]);
    expect(take("ءِلَيكَيكَ").mistakes).toEqual(["wrong_word 18:13:3"]);
  });
});

describe("refinements.ledgerPartial", () => {
  it("an ayah whose ending is left out is not also a skipped ayah", () => {
    // With every other correction on, the ending's omission is decided before the ledger closes the ayah; rc.1's
    // timing shows the duplicate, and this correction alone removes it.
    const s = speech([say(2, 4), pause(15), say(2, 5, 1, 5), pause(15), say(2, 6), pause(15), say(2, 7)]);
    expect(recite(s, { passage: passage(2, 4, 7), refinements: RC1 }).mistakes.sort()).toEqual(["skipped_passage 2:5:1", "skipped_passage 2:5:6"]);
    expect(recite(s, { passage: passage(2, 4, 7), refinements: only("ledgerPartial") }).mistakes).toEqual(["skipped_passage 2:5:6"]);
    expect(recite(s, { passage: passage(2, 4, 7), refinements: ALL }).mistakes).toEqual(["skipped_passage 2:5:6"]);
  });

  it("an ayah left out whole, with a stray sound in its place, is still a skipped ayah", () => {
    const s = speech([say(67, 3), pause(15), noise("ع"), pause(15), say(67, 5), pause(15), say(67, 6)]);
    expect(recite(s, { passage: passage(67, 3, 6), refinements: ALL }).mistakes).toEqual(["skipped_passage 67:4:1"]);
  });
});

describe("refinements.omissionDue", () => {
  /** The omission's `suspected` event sent again with `due`, if any. */
  const dueOf = (t: Take, word: string) => t.checker.find((e) => e.due && key(e.location) === word);

  it("a sure omission is said due once the reciter has read on, and confirmed by rc.1's rule, two ayahs on", () => {
    // 36:6:4 left out.
    const s = speech([say(36, 6, 1, 3), say(36, 6, 5), pause(15), say(36, 7), pause(15), say(36, 8)]);
    const next = s.saidAt.get("36:7:1")!;
    const twoOn = s.saidAt.get("36:8:1")!;
    const before = recite(s, { passage: passage(36, 6, 8), refinements: without("omissionDue") });
    expect(before.mistakes).toEqual(["omitted_word 36:6:4"]);
    expect(dueOf(before, "36:6:4")).toBeUndefined();
    const after = recite(s, { passage: passage(36, 6, 8), refinements: ALL });
    expect(after.mistakes).toEqual(["omitted_word 36:6:4"]);
    const due = dueOf(after, "36:6:4")!;
    expect(due).toMatchObject({ state: "suspected", errorClass: "omitted_word" });
    expect(due.timeMs).toBeLessThan(next);
    const confirmed = decided(after, "omitted_word", "36:6:4")!;
    expect(confirmed).toMatchObject({ state: "confirmed", id: due.id, timeMs: decided(before, "omitted_word", "36:6:4")!.timeMs });
    expect(confirmed.timeMs).toBeGreaterThan(twoOn);
    // Every other event is rc.1's.
    expect(after.checker.filter((e) => !e.due)).toEqual(before.checker);
  });

  it("waits for two words begun after the suspicion: a re-read after a long pause is not said due as it starts", () => {
    // 59:22:6 left out, two more words, a 14 s pause, then the reciter goes back to 59:22:6 and reads on.
    const s = speech([say(59, 22, 1, 5), say(59, 22, 7, 8), pause(350), say(59, 22, 6), pause(15), say(59, 23), pause(15), say(59, 24)]);
    const take = recite(s, { passage: passage(59, 22, 24), refinements: ALL });
    expect(take.mistakes).not.toContain("omitted_word 59:22:6");
    expect(decided(take, "omitted_word", "59:22:6")?.state ?? "resolved").toBe("resolved");
    expect(dueOf(take, "59:22:6")).toBeUndefined();
  });

  it("an omission that is not sure is not said due, and keeps rc.1's rule", () => {
    // 36:6:4 said, unclearly, as 36:6:2.
    const ws = ayah(36, 6);
    const unclear = noise(index.phonemes.slice(index.words[ws[1]].start, index.words[ws[1]].end), 0.4);
    const s = speech([say(36, 6, 1, 3), unclear, say(36, 6, 5), pause(15), say(36, 7), pause(15), say(36, 8)]);
    const rc1 = decided(recite(s, { passage: passage(36, 6, 8), refinements: without("omissionDue") }), "omitted_word", "36:6:4");
    const take = recite(s, { passage: passage(36, 6, 8), refinements: ALL });
    const now = decided(take, "omitted_word", "36:6:4");
    expect(rc1?.state).toBe("uncertain");
    expect(now).toMatchObject({ state: "uncertain", timeMs: rc1!.timeMs });
    expect(now!.timeMs).toBeGreaterThan(s.saidAt.get("36:8:1")!);
    expect(dueOf(take, "36:6:4")).toBeUndefined();
  });
});

describe("refinements.confirmAdvance", () => {
  it("one misheard sound after a pause does not light the next ayah's first word", () => {
    const s = speech([say(36, 6), pause(12), noise("ع"), pause(30), say(36, 7, 1, 3)]);
    const early = reached(recite(s, { passage: passage(36, 6, 7), checker: false, step: 1, refinements: without("confirmAdvance") }), "36:7:1")!;
    expect(early.timeMs).toBeLessThan(s.saidAt.get("36:7:1")!);
    const now = reached(recite(s, { passage: passage(36, 6, 7), checker: false, step: 1, refinements: ALL }), "36:7:1")!;
    expect(now.timeMs).toBeGreaterThan(s.saidAt.get("36:7:1")!);
  });

  it("a faint sound after a pause (a breath) that matches the next ayah's first sound does not light its first word", () => {
    // 36:7 opens with ل; the reciter breathes a faint ل (below minConfidence), pauses, then says 36:7.
    const l = index.phonemes[index.words[ayah(36, 7)[0]].start];
    const s = speech([say(36, 6), pause(12), noise(l, 0.4), pause(30), say(36, 7, 1, 3)]);
    const early = reached(recite(s, { passage: passage(36, 6, 7), checker: false, step: 1, refinements: without("confirmAdvance") }), "36:7:1")!;
    expect(early.timeMs).toBeLessThan(s.saidAt.get("36:7:1")!);
    const now = reached(recite(s, { passage: passage(36, 6, 7), checker: false, step: 1, refinements: ALL }), "36:7:1")!;
    expect(now.timeMs).toBeGreaterThanOrEqual(s.saidAt.get("36:7:1")!);
    // The same sound said clearly after the pause is the ayah's opening, and shows as rc.1 shows it.
    const clear = speech([say(36, 6), pause(12), say(36, 7, 1, 3)]);
    const rc1 = reached(recite(clear, { passage: passage(36, 6, 7), checker: false, step: 1, refinements: without("confirmAdvance") }), "36:7:1")!;
    expect(reached(recite(clear, { passage: passage(36, 6, 7), checker: false, step: 1, refinements: ALL }), "36:7:1")!.timeMs).toBe(rc1.timeMs);
  });

  it("in continuous speech a word heard as written shows on its first sound, as in rc.1 and the kept version", () => {
    // 2:256:3 is فِ, whose opening other words nearby share: heard as written, it shows at once. 2:255 leads in, so the
    // word is in the run's middle in every version (the take's first lock comes at different times).
    const right = speech([say(2, 255), pause(15), say(2, 256, 1, 2), say(2, 256, 3, 3), say(2, 256, 4, 7)]);
    const show = (refinements: Fields) => shownAt(recite(right, { passage: passage(2, 255, 256), checker: false, step: 1, refinements }).positions).get("2:256:3");
    expect(show(ALL)).toBe(show(without("confirmAdvance")));
    expect(show(ALL)).toBe(show(EARLIER));
    expect(show(ALL)).toBeLessThan(right.saidAt.get("2:256:3")! + 3 * FRAME_MS);
  });

  it("a word entered on a sound not heard as written waits for two of its sounds, and still shows with its run", () => {
    // فِ heard as قِ: one of its two sounds as written.
    const misheard = speech([say(2, 256, 1, 2), noise("قِ"), say(2, 256, 4, 7)]);
    const rc1 = shownAt(recite(misheard, { passage: passage(2, 256, 256), checker: false, step: 1, refinements: without("confirmAdvance") }).positions);
    const shown = shownAt(recite(misheard, { passage: passage(2, 256, 256), checker: false, step: 1, refinements: ALL }).positions);
    for (let w = 1; w <= 7; w++) expect(shown.has(`2:256:${w}`), `2:256:${w}`).toBe(true);
    expect(shown.get("2:256:3")!).toBeGreaterThan(rc1.get("2:256:3")!);
    expect(shown.get("2:256:3")!).toBeLessThanOrEqual(shown.get("2:256:4")!);
  });
});

describe("refinements.onsetRun", () => {
  it("a doubled onset shows its word from the second half of the doubling", () => {
    // 59:22: هُوَ للَااهُ: the first ل closes هُوَ, the second opens ٱللَّهُ. A phoneme every 3 frames.
    const s = speech([say(59, 21), pause(15), say(59, 22, 1, 3)]);
    const onset = s.saidAt.get("59:22:2")!;
    const take = (refinements: Fields = ALL) => recite(s, { passage: passage(59, 21, 22), checker: false, step: 1, refinements });
    expect(reached(take(RC1), "59:22:2")!.timeMs).toBeLessThan(onset + 3 * FRAME_MS);
    const now = take();
    expect(reached(now, "59:22:2")!.timeMs).toBeGreaterThanOrEqual(onset + 3 * FRAME_MS);
    expect(reached(now, "59:22:2")!.timeMs).toBeLessThan(onset + 6 * FRAME_MS);
    // Meanwhile هُوَ stays shown: no position goes back to it after 59:22:2.
    const order = now.positions.filter((p) => p.state === "locked").map((p) => at(p.observed[0].word));
    for (let k = 1; k < order.length; k++) expect(order[k]).toBeGreaterThanOrEqual(order[k - 1]);
  });

  it("a four-phoneme ghunna, as written or held, shows its word from its second half", () => {
    // 36:15: بَشَرُ مِّثلُنَا: the tanween's idgham gives مِّثلُنَا a ghunna of four م (ممممِثلُنَاا).
    const word = index.words[at({ surah: 36, ayah: 15, word: 6 })];
    const text = index.phonemes.slice(word.start, word.end);
    expect(text.startsWith("مممم") && text[4] !== "م").toBe(true);
    for (const held of [0, 3]) {
      const said = "م".repeat(4 + held) + text.slice(4);
      const s = speech([say(36, 14), pause(15), say(36, 15, 1, 5), noise(said), say(36, 15, 7)]);
      const onset = speech([say(36, 14), pause(15), say(36, 15, 1, 5)]).phonemes.at(-1)!.start * FRAME_MS + 3 * FRAME_MS;
      const take = (refinements: Fields = ALL) => recite(s, { passage: passage(36, 14, 15), checker: false, step: 1, refinements });
      const rc1 = reached(take(without("onsetRun")), "36:15:6")!.timeMs;
      const now = take();
      const shown = reached(now, "36:15:6")!.timeMs;
      expect(rc1, `held ${held}`).toBeLessThan(onset + 3 * FRAME_MS);
      // The third م opens the run's second half: 6 frames after the first.
      expect(shown, `held ${held}`).toBeGreaterThanOrEqual(onset + 6 * FRAME_MS);
      expect(shown, `held ${held}`).toBeLessThan(onset + 9 * FRAME_MS);
      const order = now.positions.filter((p) => p.state === "locked").map((p) => at(p.observed[0].word));
      for (let k = 1; k < order.length; k++) expect(order[k], `held ${held}`).toBeGreaterThanOrEqual(order[k - 1]);
    }
  });

  it("a word that is only a doubled sound shows as rc.1 shows it, before the next word", () => {
    // The reference gives 67:16's فِى only the nasal of مَّن (ںںں) and reads its فِ into ٱلسَّمَآءِ: no phoneme of its own
    // follows the run, so waiting for one would show it with the next word. مَّن (مممَ) before it still waits for its فَ.
    const fi = index.words[at({ surah: 67, ayah: 16, word: 3 })];
    expect(index.phonemes.slice(fi.start, fi.end)).toBe("ںںں");
    const s = speech([say(67, 15), pause(15), say(67, 16)]);
    const take = (refinements: Fields = ALL) => shownAt(recite(s, { passage: passage(67, 15, 16), checker: false, step: 1, refinements }).positions);
    const rc1 = take(without("onsetRun"));
    const now = take();
    for (const n of ayah(67, 16)) expect(now.has(key(ref(n))), key(ref(n))).toBe(true);
    expect(now.get("67:16:3")).toBe(rc1.get("67:16:3"));
    expect(now.get("67:16:3")!).toBeLessThan(now.get("67:16:4")!);
    expect(now.get("67:16:2")!).toBeGreaterThan(rc1.get("67:16:2")!);
    expect(now.get("67:16:2")!).toBeLessThan(now.get("67:16:3")!);
  });
});

describe("refinements.declaredStart", () => {
  const opening = (pieces: Piece[], refinements: Fields = ALL) => recite(speech(pieces), { passage: passage(36, 6, 7), checker: false, refinements });

  it("the take's first word shows alone, on the passage's first sounds", () => {
    const pieces = [say(36, 6), pause(15), say(36, 7)];
    const said = speech(pieces).saidAt;
    const rc1 = opening(pieces, without("declaredStart"));
    expect(rc1.positions.find((p) => p.state === "locked")!.observed[0].word.word).toBeGreaterThan(2);
    const now = opening(pieces).positions.find((p) => p.state === "locked")!;
    expect(now.observed[0]).toMatchObject({ word: { surah: 36, ayah: 6, word: 1 }, span: { start: { surah: 36, ayah: 6, word: 1 } } });
    expect(now.timeMs).toBeLessThan(said.get("36:6:2")!);
  });

  it("after the isti'adha the passage still locks on its first word, once the phrase is over", () => {
    const istiadha = phrases.phrases.find((p) => p.name === "istiadha")!.p;
    const pieces = [noise(istiadha), pause(20), say(36, 6), pause(15), say(36, 7)];
    const said = speech(pieces).saidAt;
    const lock = opening(pieces).positions.find((p) => p.state === "locked")!;
    expect(key(lock.observed[0].word)).toBe("36:6:1");
    expect(lock.timeMs).toBeGreaterThanOrEqual(said.get("36:6:1")!);
    expect(lock.timeMs).toBeLessThan(said.get("36:6:2")!);
  });

  it("an opening heard with a sound missing still locks on the first word when it fits every rival worse by an edit", () => {
    // Each passage's first word with its second or third phoneme dropped, as a decoder drops one.
    for (const [s, a] of [[2, 96], [7, 22], [40, 44]] as const) {
      const first = index.words[ayah(s, a)[0]];
      const written = index.phonemes.slice(first.start, first.end);
      for (const k of [1, 2]) {
        const pieces = [noise(written.slice(0, k) + written.slice(k + 1)), say(s, a, 2), pause(15), say(s, a + 1)];
        const said = speech(pieces).saidAt;
        const take = (refinements: Fields = ALL) => recite(speech(pieces), { passage: passage(s, a, a + 1), checker: false, refinements });
        const rc1 = take(without("declaredStart")).positions.find((p) => p.state === "locked")!;
        expect(rc1.timeMs, `${s}:${a} without its phoneme ${k + 1}`).toBeGreaterThan(said.get(`${s}:${a}:2`)!);
        const now = take().positions.find((p) => p.state === "locked")!;
        expect(now.observed[0], `${s}:${a} without its phoneme ${k + 1}`).toMatchObject({ word: { surah: s, ayah: a, word: 1 }, span: { start: { surah: s, ayah: a, word: 1 } } });
        expect(now.timeMs, `${s}:${a} without its phoneme ${k + 1}`).toBeLessThan(said.get(`${s}:${a}:2`)!);
      }
    }
  });

  it("a reciter who does not start at the passage start is found as rc.1 found them", () => {
    const pieces = [say(36, 8), pause(15), say(36, 9)];
    expect(opening(pieces).positions).toEqual(opening(pieces, without("declaredStart")).positions);
  });
});

describe("refinements.noLostBeforeLock", () => {
  it("no lost place is reported before the place is first found", () => {
    const s = speech([noise("زظغذثخ"), pause(30), noise("طضصق"), pause(30), say(36, 6), pause(15), say(36, 7)]);
    const rc1 = recite(s, { passage: passage(36, 6, 7), checker: false, refinements: without("noLostBeforeLock") });
    expect(rc1.positions[0].state).toBe("lost");
    const now = recite(s, { passage: passage(36, 6, 7), checker: false, refinements: ALL });
    const first = now.positions.findIndex((p) => p.state === "locked");
    expect(first).toBeGreaterThan(0);
    expect(now.positions.slice(0, first).map((p) => p.state)).not.toContain("lost");
    expect(now.positions[0].state).toBe("acquiring");
  });

  it("once the place was found, losing it is still reported", () => {
    const junk = "زظغذثخطضصقزظغذثخطضصقزظغذثخطضصق";
    const now = recite(speech([say(36, 6), pause(30), noise(junk), noise(junk)]), { passage: passage(36, 6, 7), checker: false, refinements: ALL });
    expect(now.positions.at(-1)!.state).toBe("lost");
  });
});
