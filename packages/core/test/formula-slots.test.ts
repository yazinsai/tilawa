import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { formulaSlots } from "../src/formula-slots.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import { channelFromCounts, emptyCounts, spanLogP } from "../src/noisy-channel.js";
import { loadReferenceIndex } from "../src/reference.js";
import type { CheckerPolicy } from "../src/checker.js";
import type { PositionEvent, WordRef } from "../src/events.js";
import type { ReferenceAsset } from "../src/reference.js";

const asset: ReferenceAsset = {
  version: "formula-test", text: "synthetic", profile: {},
  inventory: { id: "formula-test", tokens: "زمَنكلصسحهبغضت" },
  ayahs: [
    { s: 1, a: 1, w: ["زم", "نك", "صحب"], p: "زَمَنَكَصَحَب", o: [0, 4, 8], iso: ["زَمَ", "نَكَ", "صَحَب"] },
    { s: 1, a: 2, w: ["غض", "تب", "كل"], p: "غَضَتَبَكَلَ", o: [0, 4, 8], iso: ["غَضَ", "تَبَ", "كَلَ"] },
    { s: 1, a: 3, w: ["زم", "نك", "سهب"], p: "زَمَنَكَسَهَب", o: [0, 4, 8], iso: ["زَمَ", "نَكَ", "سَهَب"] },
    // A matching lead in a different ayah must not turn a mid-ayah word into a final slot.
    { s: 1, a: 4, w: ["زم", "نك", "صحب", "تب"], p: "زَمَنَكَصَحَبتَبَ", o: [0, 4, 8, 13], iso: ["زَمَ", "نَكَ", "صَحَب", "تَبَ"] },
    { s: 1, a: 5, w: ["نك", "غض"], p: "نَكَغَضَ", o: [0, 4], iso: ["نَكَ", "غَضَ"] },
  ],
};
for (const ayah of asset.ayahs) ayah.a++;
asset.ayahs.unshift({ s: 1, a: 1, w: ["بسم"], p: "بَسَم", o: [0], iso: ["بَسَم"] });
const index = loadReferenceIndex(asset);
const from: WordRef = { surah: 1, ayah: 2, word: 1 };
const target: WordRef = { surah: 1, ayah: 2, word: 3 };
const to: WordRef = { surah: 1, ayah: 3, word: 3 };
const counts = emptyCounts([...index.tokens]);
for (let i = 0; i < counts.phones.length; i++) counts.sub[i][i] = 100;
counts.nWritten = 100 * counts.phones.length;

describe("ayah-final formula index", () => {
  it("links distinct final fillers after the same two-word lead", () => {
    expect(formulaSlots(index).get(3)).toEqual(["سَهَب"]);
    expect(formulaSlots(index).get(9)).toEqual(["صَحَب"]);
  });

  it("excludes mid-ayah words, short ayahs and absent text", () => {
    expect(formulaSlots(index).has(12)).toBe(false);
    expect(formulaSlots(index).has(index.words.length - 1)).toBe(false);
    expect(formulaSlots({ ...index, wordText: undefined }).size).toBe(0);
  });
});

function check({ swap = true, pause = true, beforeSlotPause = false, locked = true, channel = counts, acoustics = "decoded", confidence = 0.99, policy = {} }: {
  swap?: boolean; pause?: boolean; beforeSlotPause?: boolean; locked?: boolean; channel?: typeof counts;
  acoustics?: "decoded" | "mixed" | "flat" | "off"; confidence?: number; policy?: CheckerPolicy;
} = {}) {
  const fixture = syntheticFixture(index, { id: "formula-slot", from, to, edits: [
    ...(swap ? [{ kind: "sub" as const, at: 8, to: "س" }, { kind: "sub" as const, at: 10, to: "ه" }] : []),
    ...(pause ? [{ kind: "gap" as const, at: 13, frames: 20 }] : []),
    ...(beforeSlotPause ? [{ kind: "gap" as const, at: 8, frames: 20 }] : []),
  ] });
  const raw = fixture.evidence[0]!;
  raw.phonemes.forEach((p) => { p.conf = confidence; });
  const flat: [string, number][] = ["", ...index.tokens].map((p) => [p, 1 / (index.tokens.size + 1)]);
  const evidence = acoustics === "off" ? raw : acoustics === "flat"
    ? withPosteriors(index, raw, () => flat, flat.length) : acoustics === "mixed"
      ? withPosteriors(index, raw, (j) => j === 8 ? [["س", 0.75], ["ص", 0.2], ["", 0.05]]
        : j === 10 ? [["ه", 0.75], ["ح", 0.2], ["", 0.05]] : undefined)
      : withPosteriors(index, raw);
  const lock: PositionEvent = {
    v: 1, type: "position", state: "locked", timeMs: 0, referenceVersion: index.version, expected: target,
    observed: [{ word: target, span: { start: from, end: target }, confidence: 1, cost: 0, per: 0 }],
  };
  const checker = createChecker(index, { passage: { from, to }, channel, policy: { confirm: { after: "next-word" }, ...policy } });
  checker.push(evidence, locked ? [lock] : []);
  checker.end();
  return checker.mistakes().map((e) => `${e.errorClass} ${e.location.ayah}:${e.location.word}`);
}

describe("closed formula choice under the listener channel", () => {
  it("confirms a soft consonant swap that wins the closed choice", () => {
    expect(check()).toEqual(["wrong_word 2:3"]);
    expect(check({ channel: emptyCounts([]) })).toEqual([]);
  });

  it("keeps the written word and abstains on a tie", () => {
    expect(check({ swap: false })).toEqual([]);
    const tied = emptyCounts([...index.tokens]);
    const ch = channelFromCounts(tied);
    expect(spanLogP([..."سَهَب"], [..."صَحَب"], ch)).toBe(spanLogP([..."سَهَب"], [..."سَهَب"], ch));
    expect(check({ channel: tied })).toEqual([]);
  });

  it("requires the lock and a labelled pause", () => {
    expect(check({ pause: false })).toEqual([]);
    expect(check({ locked: false })).toEqual([]);
  });

  it("scores the slot's own visit when the reciter paused before it", () => {
    expect(check({ beforeSlotPause: true })).toEqual(["wrong_word 2:3"]);
  });

  it("requires twice the support margin when the changed phones are weak", () => {
    const policy: CheckerPolicy = { localWindow: { nats: 4 } };
    expect(check({ acoustics: "mixed", policy })).toEqual(["wrong_word 2:3"]);
    expect(check({ acoustics: "mixed", confidence: 0.7, policy })).toEqual([]);
  });

  it("retains affirmative support and confidence gates", () => {
    expect(check({ acoustics: "flat" })).toEqual([]);
    expect(check({ acoustics: "off" })).toEqual([]);
    expect(check({ confidence: 0.5 })).toEqual([]);
    expect(check({ policy: { localWindow: { nats: 1e6 } } })).toEqual([]);
  });
});
