import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { anchorSkipRuns, duplicateSpans } from "../src/checker-anchors.js";
import type { AnchorVisit } from "../src/checker-anchors.js";
import { createChecker } from "../src/checker.js";
import type { CheckerEvent, DecodedPhoneme, WordRef } from "../src/events.js";
import type { PhonemeEvidence } from "../src/evidence.js";
import { loadReferenceIndex } from "../src/reference.js";
import type { PhraseAsset, ReferenceAsset } from "../src/reference.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8"));
const phrases: PhraseAsset = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const at = (surah: number, ayah: number, word: number) =>
  index.words.findIndex((w) => w.surah === surah && w.ayah === ayah && w.word === word);

describe("duplicate spans", () => {
  const spans = duplicateSpans(index.words, index.phonemes);

  it("treats a surah refrain as duplicate text and the ayah between two copies as unique", () => {
    expect(spans.unique[at(55, 28, 1)]).toBe(false);
    expect(spans.unique[at(55, 30, 1)]).toBe(false);
    expect(spans.unique[at(55, 29, 1)]).toBe(true);
    expect(spans.unique[at(26, 108, 1)]).toBe(false);
    expect(spans.unique[at(26, 110, 1)]).toBe(false);
    expect(spans.unique[at(26, 130, 2)]).toBe(true);
    expect(spans.unique[at(26, 103, 1)]).toBe(false);
    expect(spans.unique[at(77, 34, 1)]).toBe(false);
    expect(spans.unique[at(77, 33, 1)]).toBe(true);
  });
});

describe("anchor skip runs", () => {
  const visit = (word: number, at: number): AnchorVisit => ({ word, obsFrom: at, obsTo: at + 1, counts: true, wrong: false });

  it("puts the skip on the unique ayah between two heard copies of a refrain", () => {
    // L, refrain, unique, refrain, R — both refrains heard.
    const form = ["L", "a", "b", "u", "a", "b", "R"];
    const unique = [true, false, false, true, false, false, true];
    const ayahStart = [true, true, false, true, true, false, true];
    const visits = [0, 1, 2, 1, 2, 6].map((word, i) => visit(word, i));
    expect(anchorSkipRuns({ form, unique, ayahStart, visits })).toEqual([{ from: 0, to: 6, runs: [{ word: 3, last: 3 }] }]);
  });

  it("does not warn when the refrain is repeated and the unique anchors stay contiguous", () => {
    const form = ["L", "a", "b", "R"];
    const unique = [true, false, false, true];
    const ayahStart = [true, true, false, true];
    const visits = [0, 1, 2, 1, 2, 3].map((word, i) => visit(word, i));
    expect(anchorSkipRuns({ form, unique, ayahStart, visits })).toEqual([{ from: 0, to: 3, runs: [] }]);
  });

  it("keeps a repeated boundary word on the ayah after the skip", () => {
    // ولا [أنتم عابدون] ولا أنا: the heard ولا is the one after the skipped ayah.
    const form = ["L", "w", "x", "y", "w", "R"];
    const unique = [true, false, true, true, false, true];
    const ayahStart = [true, true, false, false, true, true];
    const visits = [visit(0, 0), visit(4, 1), visit(5, 2)];
    expect(anchorSkipRuns({ form, unique, ayahStart, visits })[0].runs).toEqual([{ word: 1, last: 3 }]);
  });

  it("keeps the first الناس when the skipped ayah starts on a different word", () => {
    const form = ["L", "n", "m", "n", "R"];
    const unique = [true, false, true, false, true];
    const ayahStart = [true, false, true, false, true];
    const visits = [visit(0, 0), visit(1, 1), visit(4, 2)];
    expect(anchorSkipRuns({ form, unique, ayahStart, visits })[0].runs).toEqual([{ word: 2, last: 3 }]);
  });
});

describe("refrain skips", () => {
  const find = (r: WordRef) => index.words.findIndex((w) => w.surah === r.surah && w.ayah === r.ayah && w.word === r.word);
  const ref = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });

  /** Oracle phonemes and a lock on each recited word, with an optional inclusive skip and one repeated stretch. */
  function recite(o: { from: WordRef; to: WordRef; skip?: [WordRef, WordRef]; repeat?: [WordRef, WordRef] }): CheckerEvent[] {
    const first = find(o.from);
    const last = find(o.to);
    const skipLo = o.skip ? find(o.skip[0]) : -1;
    const skipHi = o.skip ? find(o.skip[1]) : -1;
    const repLo = o.repeat ? find(o.repeat[0]) : -1;
    const repHi = o.repeat ? find(o.repeat[1]) : -1;
    const phonemes: DecodedPhoneme[] = [];
    const marks: { frame: number; word: WordRef }[] = [];
    let frame = 0;
    const emitWord = (i: number) => {
      const w = index.words[i];
      if (w.end === w.start) return;
      marks.push({ frame, word: { surah: w.surah, ayah: w.ayah, word: w.word } });
      for (const p of index.phonemes.slice(w.start, w.end)) {
        phonemes.push({ p, start: frame, end: frame + 1, conf: 0.99 });
        frame += 3;
      }
    };
    for (let i = first; i <= last; i++) {
      if (i >= skipLo && i <= skipHi) continue;
      emitWord(i);
      if (i === repHi) for (let k = repLo; k <= repHi; k++) emitWord(k);
    }
    const checker = createChecker(index, { phrases, policy: { localWindow: false, minConfidence: 0 } });
    const out: CheckerEvent[] = [];
    let mark = 0;
    for (let i = 0; i < phonemes.length; i += 8) {
      const chunk = phonemes.slice(i, i + 8);
      while (mark + 1 < marks.length && marks[mark + 1].frame <= chunk[0].start) mark++;
      const word = marks[mark]?.word ?? o.from;
      const evidence: PhonemeEvidence = { inventory: index.inventoryId, frameMs: 20, phonemes: chunk };
      out.push(...checker.push(evidence, [{ type: "position", v: 1, timeMs: chunk[0].start * 20, referenceVersion: index.version,
        expected: word, state: "locked", observed: [{ word, span: { start: o.from, end: word }, confidence: 1, cost: 0, per: 0 }] }]));
    }
    out.push(...checker.end());
    return out;
  }

  const confirmedAt = (events: CheckerEvent[]) =>
    events.filter((e) => e.state === "confirmed").map((e) => `${e.errorClass} ${key(e.location)}`);

  it("catches the unique ayah between two refrain copies", () => {
    const events = recite({ from: ref(55, 26, 1), to: ref(55, 32, 1), skip: [ref(55, 29, 1), ref(55, 29, 10)] });
    expect(confirmedAt(events)).toContain("skipped_passage 55:29:1");
    expect(confirmedAt(events).some((e) => e.startsWith("skipped_passage 55:28") || e.startsWith("skipped_passage 55:30"))).toBe(false);
  });

  it("catches the formula ayah between the two فاتقوا refrains", () => {
    const events = recite({ from: ref(26, 106, 1), to: ref(26, 112, 1), skip: [ref(26, 109, 1), ref(26, 109, 11)] });
    expect(confirmedAt(events)).toContain("skipped_passage 26:109:1");
  });

  it("catches a skipped ayah in front of a refrain", () => {
    const events = recite({ from: ref(26, 127, 1), to: ref(26, 133, 1), skip: [ref(26, 130, 1), ref(26, 130, 4)] });
    expect(confirmedAt(events)).toContain("skipped_passage 26:130:1");
    expect(confirmedAt(events).some((e) => e.includes("26:131"))).toBe(false);
  });

  it("places the skip on the refrain ayah, not the look-alike word before it", () => {
    const events = recite({ from: ref(26, 100, 1), to: ref(26, 106, 1), skip: [ref(26, 103, 1), ref(26, 103, 8)] });
    expect(confirmedAt(events)).toContain("skipped_passage 26:103:1");
    expect(confirmedAt(events).some((e) => e.includes("26:102"))).toBe(false);
  });

  it("catches the ayah before ويل يومئذ للمكذبين", () => {
    const events = recite({ from: ref(77, 31, 1), to: ref(77, 36, 1), skip: [ref(77, 33, 1), ref(77, 33, 3)] });
    expect(confirmedAt(events)).toContain("skipped_passage 77:33:1");
  });

  it("stays quiet when a refrain is repeated and nothing unique was skipped", () => {
    const events = recite({ from: ref(55, 26, 1), to: ref(55, 31, 1), repeat: [ref(55, 28, 1), ref(55, 28, 4)] });
    expect(confirmedAt(events).filter((e) => e.includes("skipped") || e.includes("omitted"))).toEqual([]);
  });
});
