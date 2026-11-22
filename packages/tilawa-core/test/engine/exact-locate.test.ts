import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { EvidenceError, createTracker, loadReferenceIndex } from "../../src/engine/index.js";
import type { PhonemeEvidence, PositionEvent, ReferenceAsset, WordRef } from "../../src/engine/index.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);

const wordIndex = (r: WordRef) => index.words.findIndex((w) => w.surah === r.surah && w.ayah === r.ayah && w.word === r.word);

function evidenceFor(from: WordRef, to: WordRef, inventory = index.inventoryId): PhonemeEvidence {
  const text = index.phonemes.slice(index.words[wordIndex(from)].start, index.words[wordIndex(to)].end);
  return {
    inventory,
    frameMs: 20,
    phonemes: [...text].map((p, i) => ({ p, start: i * 3, end: i * 3 + 1, conf: 0.99 })),
  };
}

function locate(evidence: PhonemeEvidence): PositionEvent {
  const tracker = createTracker(index);
  tracker.push(evidence);
  const events = tracker.end();
  expect(events).toHaveLength(1);
  return events[0] as PositionEvent;
}

const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const lastWord = (surah: number, ayah: number) => asset.ayahs.find((a) => a.s === surah && a.a === ayah)!.w.length;

describe("reference index", () => {
  it("records the versions it was built from", () => {
    expect(index.version).toBe("ref-v2");
    expect(asset.text).toMatch(/^quran-transcript \d/);
    expect(asset.profile).toMatchObject({ rewaya: "hafs" });
    expect(index.tokens.size).toBe(42);
    expect(asset.ayahs).toHaveLength(6236);
  });

  // In "إِنَّ ٱلْإِنسَـٰنَ" the doubled noon and its vowel are إِنَّ's; hamzat al-wasl gives the next word nothing to start with.
  it("keeps a doubled consonant before hamzat al-wasl in the earlier word", () => {
    const e = evidenceFor(w(100, 6, 1), w(100, 6, 4));
    // The decoder missed the onset: only the tail of إِنَّ survives.
    const clipped = { ...e, phonemes: e.phonemes.slice(2) };
    expect(locate(clipped).observed[0].span.start).toEqual(w(100, 6, 1));
    expect(index.phonemes[index.words[wordIndex(w(100, 6, 2))].start]).toBe("ل");
    expect(index.phonemes[index.words[wordIndex(w(2, 29, 10))].start]).toBe("س");
  });
});

describe("exact locate", () => {
  it("locks a single ayah to its full word span", () => {
    const e = locate(evidenceFor(w(1, 2, 1), w(1, 2, 4)));
    expect(e.state).toBe("locked");
    expect(e.observed).toHaveLength(1);
    expect(e.observed[0].span).toEqual({ start: w(1, 2, 1), end: w(1, 2, 4) });
    expect(e.observed[0].word).toEqual(w(1, 2, 4));
    expect(e.referenceVersion).toBe("ref-v2");
  });

  it("finds the starting word of a mid-ayah start", () => {
    const e = locate(evidenceFor(w(2, 255, 5), w(2, 255, 14)));
    expect(e.state).toBe("locked");
    expect(e.observed[0].span).toEqual({ start: w(2, 255, 5), end: w(2, 255, 14) });
  });

  it("spans consecutive ayahs", () => {
    const e = locate(evidenceFor(w(2, 1, 1), w(2, 2, lastWord(2, 2))));
    expect(e.state).toBe("locked");
    expect(e.observed[0].span).toEqual({ start: w(2, 1, 1), end: w(2, 2, lastWord(2, 2)) });
  });

  it("keeps every occurrence of a repeated verse and never locks", () => {
    const target = asset.ayahs.find((a) => a.s === 55 && a.a === 13)!;
    const occurrences = asset.ayahs.filter((a) => a.p === target.p).map((a) => a.a);
    expect(occurrences.length).toBeGreaterThan(1);

    const tracker = createTracker(index);
    const events = [...tracker.push(evidenceFor(w(55, 13, 1), w(55, 13, target.w.length))), ...tracker.end()];
    for (const e of events as PositionEvent[]) expect(e.state).not.toBe("locked");

    const final = events.at(-1) as PositionEvent;
    expect(final.state).toBe("ambiguous");
    expect(final.observed.map((c) => c.span.start.ayah).sort((a, b) => a - b)).toEqual(occurrences);
  });

  it("reports lost when nothing in the Quran matches", () => {
    const e = locate({ ...evidenceFor(w(1, 2, 1), w(1, 2, 4)), phonemes: [..."ظظظظظظظظ"].map((p, i) => ({ p, start: i, end: i, conf: 0.9 })) });
    expect(e.state).toBe("lost");
    expect(e.observed).toEqual([]);
  });
});

describe("evidence validation", () => {
  it("rejects a mismatched inventory", () => {
    expect(() => createTracker(index).push(evidenceFor(w(1, 2, 1), w(1, 2, 4), "other-inventory"))).toThrow(EvidenceError);
  });

  it("rejects phonemes outside the inventory", () => {
    const e = evidenceFor(w(1, 2, 1), w(1, 2, 4));
    e.phonemes[0] = { ...e.phonemes[0], p: "x" };
    expect(() => createTracker(index).push(e)).toThrow(EvidenceError);
  });

  it("rejects a frame stride change within a session", () => {
    const tracker = createTracker(index);
    tracker.push(evidenceFor(w(1, 2, 1), w(1, 2, 2)));
    expect(() => tracker.push({ ...evidenceFor(w(1, 2, 3), w(1, 2, 4)), frameMs: 40 })).toThrow(EvidenceError);
  });
});
