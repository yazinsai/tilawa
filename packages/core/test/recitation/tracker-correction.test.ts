/**
 * Correction-mode tracker pieces on hand-built tracker state (no audio): the
 * expected-passage jump window, the skipped-ayah check and end-of-ayah
 * anchoring of settled verdicts.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { requireCorpus } from "./paths";
import { QuranCorpus } from "../../src/recitation/corpus";
import { QuranIndex, preamblePending, ISTIADHA, BASMALA } from "../../src/recitation/search";
import { RecitationEngine } from "../../src/recitation/engine";
import { Tracker } from "../../src/recitation/tracker";
import { VerdictTracer } from "../../src/recitation/verdicts";
import { DEFAULT_CONFIG } from "../../src/recitation/config";
import { costTable } from "../../src/recitation/phonemeCost";

const corpus = new QuranCorpus(JSON.parse(readFileSync(requireCorpus(), "utf8")));
const index = new QuranIndex(corpus, DEFAULT_CONFIG);
const table = costTable();

/** Tracker whose heard chars are `words` (global word indices), each char
 * placed by hand on the ref cells of `onto` (same length list, `null` = the
 * word itself): what the aligner did, not what the reciter said. */
function handTracker(surah: number, start: number, words: number[], onto: Array<number | null>): Tracker {
  const t = new Tracker(corpus, table, surah, start, DEFAULT_CONFIG);
  let frame = 0;
  words.forEach((w, k) => {
    const chars = [...corpus.wordPhonemes(w)];
    const target = onto[k] ?? w;
    const local = target - t.firstWord;
    const from = t.wordStarts[local]!;
    const to = local + 1 < t.wordStarts.length ? t.wordStarts[local + 1]! : t.len;
    chars.forEach((ch, i) => {
      t.heard.push({ ch, frame: frame += 2, margin: 0.99 });
      t.trail.push(Math.min(to, from + 1 + Math.floor((i * (to - from)) / chars.length)));
    });
  });
  return t;
}

const ayahWords = (s: number, a: number) =>
  Array.from({ length: corpus.ayahWordCount(s, a) }, (_, i) => corpus.ayahFirstWord(s, a) + i);

describe("expected-passage jump window", () => {
  it("adds outsideJumpCost only to words outside the window", () => {
    const cfg = { ...DEFAULT_CONFIG, outsideJumpCost: 9 };
    const win = { firstWord: corpus.ayahFirstWord(55, 16), endWord: corpus.ayahFirstWord(55, 17) };
    const t = new Tracker(corpus, table, 55, win.firstWord, cfg, win);
    const local = (w: number) => w - t.firstWord;
    expect(t.jumpExtra[local(win.firstWord)]).toBe(0);
    expect(t.jumpExtra[local(win.endWord - 1)]).toBe(0);
    expect(t.jumpExtra[local(corpus.ayahFirstWord(55, 13))]).toBe(9);
    expect(t.column[t.wordStarts[local(corpus.ayahFirstWord(55, 13))]!]).toBe(cfg.jumpCost + 9);
  });

  it("is all zeros without a window, so the DP is unchanged", () => {
    const start = corpus.ayahFirstWord(55, 16);
    const a = new Tracker(corpus, table, 55, start, DEFAULT_CONFIG);
    const b = new Tracker(corpus, table, 55, start, { ...DEFAULT_CONFIG, outsideJumpCost: 9 });
    expect(a.jumpExtra.every((x) => x === 0)).toBe(true);
    const chars = [...corpus.ayahPhonemes(55, 16)].map((ch, i) => ({ ch, frame: i * 2, margin: 0.9 }));
    a.feed(chars);
    b.feed(chars);
    expect(Array.from(b.column)).toEqual(Array.from(a.column));
    expect(b.trail).toEqual(a.trail);
  });
});

describe("preamblePending", () => {
  it("holds while an isti'adha or basmala may still be growing", () => {
    expect(preamblePending(ISTIADHA.slice(0, 14), table)).toBe(true);
    expect(preamblePending(BASMALA.slice(0, 12), table)).toBe(true);
    expect(preamblePending(corpus.ayahPhonemes(112, 1).slice(0, 14), table)).toBe(false);
  });
});

describe("skippedAyah", () => {
  const engine = new RecitationEngine(corpus, index, DEFAULT_CONFIG);
  engine.setCorrection(true);
  engine.setExpected({ surah: 87, ayah: 13, ayahEnd: 15 });
  const a13 = ayahWords(87, 13);
  const a14 = ayahWords(87, 14);
  const a15 = ayahWords(87, 15);

  it("ayah 15's audio bent onto ayah 14 reads as a skip of 14", () => {
    const onto = [...a13.map(() => null), ...a15.map((_, i) => a14[Math.min(i, a14.length - 1)]!)];
    const t = handTracker(87, a13[0]!, [...a13, ...a15], onto);
    expect(engine.skippedAyah(t)).toEqual({ surah: 87, ayah: 14 });
  });

  it("ayah 14 itself on ayah 14 is not a skip", () => {
    const t = handTracker(87, a13[0]!, [...a13, ...a14], [...a13, ...a14].map(() => null));
    expect(engine.skippedAyah(t)).toBeNull();
  });

  it("needs the expected passage and correction mode", () => {
    const onto = [...a13.map(() => null), ...a15.map((_, i) => a14[Math.min(i, a14.length - 1)]!)];
    const t = handTracker(87, a13[0]!, [...a13, ...a15], onto);
    const plain = new RecitationEngine(corpus, index, DEFAULT_CONFIG);
    plain.setExpected({ surah: 87, ayah: 13, ayahEnd: 15 });
    expect(plain.skippedAyah(t)).toBeNull();
    plain.setCorrection(true);
    plain.setExpected(null);
    expect(plain.skippedAyah(t)).toBeNull();
  });
});

describe("anchorAyahEnd", () => {
  // 112:1 qul huwa llahu ahad, reciter drops "llahu" (word 2): the aligner put
  // "ahad" on word 2 and the take ended there.
  const w = ayahWords(112, 1);
  const said = [w[0]!, w[1]!, w[3]!];
  const onto = [null, null, w[2]!];

  it("off: the last word has no verdict and the dropped word reads as heard", () => {
    const tracer = new VerdictTracer(handTracker(112, w[0]!, said, onto), table, DEFAULT_CONFIG);
    const vs = tracer.verdicts(true);
    expect(vs.find((v) => v.wordIndex === w[3])).toBeUndefined();
    expect(vs.find((v) => v.wordIndex === w[2])?.state).not.toBe("skipped");
  });

  it("on: settled verdicts realign through the ayah end", () => {
    const tracer = new VerdictTracer(handTracker(112, w[0]!, said, onto), table, DEFAULT_CONFIG);
    tracer.anchorAyahEnd = 2;
    const vs = tracer.verdicts(true);
    expect(vs.find((v) => v.wordIndex === w[2])?.state).toBe("skipped");
    expect(vs.find((v) => v.wordIndex === w[3])?.state).toBe("ok");
    // Live (unsettled) verdicts are not anchored.
    expect(tracer.verdicts(false).find((v) => v.wordIndex === w[3])).toBeUndefined();
  });
});
