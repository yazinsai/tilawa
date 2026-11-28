import { describe, expect, it } from "vitest";
import { DEFAULTS } from "../../src/defaults.js";
import type { CheckerEvent, EngineEvent, PositionEvent, WordRef } from "../../src/engine/index.js";
import { createPacer } from "../../src/runtime/session.js";
import { resolvePolicy } from "../../src/session.js";
import { ANSWER_FRAMES, index, key, scriptedTake } from "./scripted.js";

const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const locked = (word: WordRef, t = 0): PositionEvent => ({
  type: "position", v: 1, timeMs: t, referenceVersion: index.version, expected: word, state: "locked",
  observed: [{ word, span: { start: w(112, 1, 1), end: word }, confidence: 0.9, cost: 0, per: 0 }],
});
const mistake = (word: WordRef): CheckerEvent => ({
  type: "checker", v: 1, id: `${key(word)}#1`, state: "suspected", errorClass: "omitted_word", location: word, expected: "", confidence: 1, timeMs: 0,
});

const opensAyah = (r: WordRef) => r.word === 1;
const take = (pushMs: number, pace: boolean, surah: number, ayahs: number[], skip: string[] = []) =>
  scriptedTake({ pushMs, policy: { refinements: { pace } }, surah, ayahs, skip });

describe("refinements.pace: the pacer", () => {
  it("shows a word that came with the one before it at the last push before it is due, on the audio clock", () => {
    // Two words completed by one answer, said 240 ms apart.
    const [a, b] = [locked(w(112, 1, 2), 1000), locked(w(112, 1, 3), 1240)];
    // Pushes of 160 ms: b is due at 1600 + 240; the push ending at 1760 is the last before it.
    let pacer = createPacer(opensAyah);
    expect(pacer.tick(1600, 160)).toEqual([]);
    expect(pacer.admit([a, b])).toEqual([a]);
    expect(pacer.tick(1760, 160)).toEqual([b]);
    // Pushes of 16 ms: the push that ends at 1840, its due time.
    pacer = createPacer(opensAyah);
    pacer.tick(1600, 16);
    expect(pacer.admit([a, b])).toEqual([a]);
    const shown: number[] = [];
    for (let clock = 1616; clock <= 1900; clock += 16) if (pacer.tick(clock, 16).length) shown.push(clock);
    expect(shown).toEqual([1840]);
    // Pushes of 240 ms: the next push ends when it is due.
    pacer = createPacer(opensAyah);
    pacer.tick(1600, 240);
    expect(pacer.admit([a, b])).toEqual([a]);
    expect(pacer.tick(1840, 240)).toEqual([b]);
    // Pushes longer than the gap hold nothing.
    for (const ms of [256, 640]) {
      pacer = createPacer(opensAyah);
      pacer.tick(1600, ms);
      expect(pacer.admit([a, b]), `${ms} ms`).toEqual([a, b]);
    }
  });

  it("holds a word only when it comes with the one before it, and never an ayah's first word", () => {
    const pacer = createPacer(opensAyah);
    pacer.tick(1600, 160);
    const [a, b] = [locked(w(112, 1, 2), 1000), locked(w(112, 1, 3), 1300)];
    expect(pacer.admit([a])).toEqual([a]);
    // The next push: b came alone, however far from a it was said.
    pacer.tick(1760, 160);
    expect(pacer.admit([b])).toEqual([b]);
    // An ayah's first word is shown at once, with the word held before it.
    const [c, d, e] = [locked(w(112, 1, 4), 1500), locked(w(112, 2, 1), 1800), locked(w(112, 2, 2), 2100)];
    pacer.tick(2400, 160);
    expect(pacer.admit([c])).toEqual([c]);
    pacer.tick(2560, 160);
    const f = locked(w(112, 2, 3), 2400);
    expect(pacer.admit([d, e, f])).toEqual([d]);
    // e is due 300 ms after d (2860), f 300 ms after e (3160).
    expect(pacer.tick(2720, 160)).toEqual([e]);
    expect(pacer.tick(2880, 160)).toEqual([]);
    expect(pacer.tick(3040, 160)).toEqual([f]);
  });

  it("takes the next push to be as long as the longest recent push, so uneven pushes do not show a word late", () => {
    const [a, b] = [locked(w(112, 1, 2), 1000), locked(w(112, 1, 3), 1240)];
    // A 640 ms push, then a 16 ms one: b, due 240 ms on, would wait for the next 640 ms push and show 400 ms late.
    let pacer = createPacer(opensAyah);
    pacer.tick(1600, 640);
    pacer.tick(1616, 16);
    expect(pacer.admit([a, b])).toEqual([a, b]);
    // Once the latest eight pushes are all short, b is held again, and shown by the push that ends when it is due.
    pacer = createPacer(opensAyah);
    pacer.tick(1600, 640);
    let clock = 1600;
    for (let k = 0; k < 8; k++) pacer.tick((clock += 16), 16);
    expect(pacer.admit([a, b])).toEqual([a]);
    const shown: number[] = [];
    while (!shown.length) if (pacer.tick((clock += 16), 16).length) shown.push(clock);
    expect(shown).toEqual([1728 + 240]);
  });

  it("tells a held event from a shown one, so a session's mistakes() lists only what was shown", () => {
    const pacer = createPacer(opensAyah);
    pacer.tick(1600, 160);
    const [a, b] = [locked(w(112, 1, 2), 1000), locked(w(112, 1, 3), 1240)];
    const m = mistake(w(112, 1, 3));
    expect(pacer.admit([a, b, m])).toEqual([a]);
    expect([pacer.holds(a), pacer.holds(b), pacer.holds(m)]).toEqual([false, true, true]);
    expect(pacer.tick(1760, 160)).toEqual([b, m]);
    expect(pacer.holds(m)).toBe(false);
  });

  it("never reorders: anything behind a held position waits with it, and the take's end shows everything", () => {
    const pacer = createPacer(opensAyah);
    pacer.tick(1600, 160);
    const [a, b] = [locked(w(112, 1, 2), 1000), locked(w(112, 1, 3), 1240)];
    const back = locked(w(112, 1, 2), 1300);
    const lost: PositionEvent = { ...a, state: "lost", observed: [] };
    const m = mistake(w(112, 1, 3));
    expect(pacer.admit([a, m, b, back, lost])).toEqual([a, m]);
    expect(pacer.flush()).toEqual([b, back, lost]);
    expect(pacer.flush()).toEqual([]);
    // Repeats and steps back are never held themselves.
    pacer.tick(1760, 160);
    const again = [locked(w(112, 1, 2), 1400), locked(w(112, 1, 1), 1440)];
    expect(pacer.admit(again)).toEqual(again);
    // A word held when an ayah's first word comes is shown with it.
    pacer.tick(1920, 160);
    const [x, y, z] = [locked(w(112, 1, 3), 1500), locked(w(112, 1, 4), 1700), locked(w(112, 2, 1), 1750)];
    expect(pacer.admit([x, y, z])).toEqual([x, y, z]);
  });
});

describe("refinements.pace: said-to-shown in a session, by push size", () => {
  for (const pushMs of [16, 100, 160, 256, 640]) {
    it(`pushes of ${pushMs} ms: holds a word no longer than it was said after the word it came with, never an ayah's first`, async () => {
      for (const [surah, ayahs] of [[67, [1, 2, 3]], [112, [1, 2, 3, 4]]] as [number, number[]][]) {
        const off = await take(pushMs, false, surah, ayahs);
        const on = await take(pushMs, true, surah, ayahs);
        // The same positions in the same order, and the same checker events.
        const positions = (es: EngineEvent[]) => es.filter((e) => e.type === "position");
        expect(positions(on.events)).toEqual(positions(off.events));
        expect(on.events.filter((e) => e.type === "checker")).toEqual(off.events.filter((e) => e.type === "checker"));
        expect([...on.words.keys()]).toEqual([...off.words.keys()]);
        expect(on.words.size).toBe(index.words.filter((x) => x.surah === surah && ayahs.includes(x.ayah) && x.end > x.start).length);
        // The word each word came with: the first word new in the same push without pacing.
        const head = new Map(off.pushes.flatMap((ks) => ks.map((k) => [k, ks[0]] as const)));
        for (const [k, o] of off.words) {
          const p = on.words.get(k)!;
          expect(p.shown, `${k} not before it arrives`).toBeGreaterThanOrEqual(o.shown);
          expect(p.shown - o.shown, `${k} held no longer than it was said after ${head.get(k)}`)
            .toBeLessThanOrEqual(o.evidence - off.words.get(head.get(k)!)!.evidence);
          if (k.endsWith(":1")) expect(p.shown, `${k}, an ayah's first word, is not held`).toBe(o.shown);
        }
        // Pushes shorter than the gap between two words show them apart; pushes of a model answer or more change nothing.
        const together = (r: typeof on) => r.pushes.filter((ks) => ks.length > 1).length;
        expect(together(off), `${surah}: an answer completes two words`).toBeGreaterThan(0);
        if (pushMs <= 160) expect(together(on), `${surah}`).toBeLessThan(together(off));
        if (pushMs >= 320) expect(on.pushes, `${surah}`).toEqual(off.pushes);
      }
    }, 120_000);
  }
});

describe("refinements.pace: an ayah's last word before a long pause", () => {
  it("shows during the pause, within one model answer of when it showed unpaced, at pushes of 16 to 320 ms", async () => {
    // 2.4 s of silence after each ayah: the pause brings no new word, and the pushes of silence show what is held.
    // declaredStart on (the default), so the passage's first ayah shows as it is said: off, the first lock comes from
    // global search, after 112:2 has begun, and 112:1's last word shows with it, paced or not.
    for (const pushMs of [16, 160, 320]) {
      const take = (pace: boolean) => scriptedTake({ pushMs, policy: { refinements: { pace } }, surah: 112, ayahs: [1, 2, 3, 4], gap: 60 });
      const [off, on] = [await take(false), await take(true)];
      for (const a of [1, 2, 3]) {
        const last = index.words.filter((x) => x.surah === 112 && x.ayah === a && x.end > x.start).at(-1)!;
        const k = key(last);
        const [o, p] = [off.words.get(k)!, on.words.get(k)!];
        expect(p.shown - o.shown, `${pushMs} ms: ${k}`).toBeLessThanOrEqual(ANSWER_FRAMES * 40);
        expect(p.shown, `${pushMs} ms: ${k} shown before the next ayah is said`).toBeLessThan(on.words.get(`112:${a + 1}:1`)!.said);
      }
    }
  }, 120_000);
});

describe("refinements.pace: mistakes() in a session", () => {
  it("after every push lists exactly the confirmations shown so far", async () => {
    // A word said as other sounds (67:1:5), confirmed after the delay; pushes of 16 and 160 ms.
    for (const pushMs of [16, 160]) {
      const t = await scriptedTake({ pushMs, surah: 67, ayahs: [1, 2, 3], change: { "67:1:5": "زظغذث" } });
      expect(t.confirmed.at(-1)!.length, `${pushMs} ms: the word is confirmed`).toBeGreaterThan(0);
      t.listed.forEach((ids, k) => expect(ids, `${pushMs} ms, push ${k}`).toEqual(t.confirmed[k]));
    }
  }, 120_000);
});

describe("refinements policy", () => {
  it("resolves every refinement on by default but markWhenSure, and each can be switched alone", () => {
    const names = Object.keys(DEFAULTS.policy.refinements);
    expect(names.sort()).toEqual([
      "begunByConsonants", "commitAfterJump", "confirmAdvance", "consonantConfirm", "declaredLeadIn", "declaredMove", "declaredStart",
      "firstAyahOut", "ledgerPartial", "leftOnNewAudio", "markWhenSure", "marksInPassage", "noLostBeforeLock", "omissionDue", "onsetRun",
      "pace", "passageWindow", "readNotSkipped", "reanchorBack", "repeatNotWrong", "restartNoEntry", "rivalHold", "showSaid", "stutter",
      "withdrawBegun",
    ]);
    const off = (r: Record<string, boolean>) => Object.entries(r).filter(([, on]) => !on).map(([k]) => k).sort();
    expect(off(resolvePolicy({}, 40).refinements)).toEqual(["markWhenSure"]);
    expect(off(resolvePolicy({ refinements: { pace: false } }, 40).refinements)).toEqual(["markWhenSure", "pace"]);
    expect(off(resolvePolicy({ refinements: { declaredStart: false } }, 40).refinements)).toEqual(["declaredStart", "markWhenSure"]);
    expect(off(resolvePolicy({ refinements: { markWhenSure: true } }, 40).refinements)).toEqual([]);
    expect(off(resolvePolicy({ refinements: { readNotSkipped: false, rivalHold: false } }, 40).refinements)).toEqual(["markWhenSure", "readNotSkipped", "rivalHold"]);
    expect(off(resolvePolicy({ refinements: { declaredMove: false, begunByConsonants: false } }, 40).refinements)).toEqual(["begunByConsonants", "declaredMove", "markWhenSure"]);
  });
});
