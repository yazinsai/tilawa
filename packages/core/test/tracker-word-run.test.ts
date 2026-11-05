import { describe, expect, it } from "vitest";
import type { PositionEvent } from "../src/events.js";
import type { LocalPosition } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";
import { committedWordSpan, createWordRun } from "../src/tracker-word-run.js";
import { wordFit } from "../src/word-fit.js";

const index = loadReferenceIndex({ version: "test", text: "test", profile: {}, inventory: { id: "test", tokens: "بتجحخ" },
  ayahs: [{ s: 1, a: 1, w: ["a", "b", "c", "d", "e"], p: "بتجحخ", o: [0, 1, 2, 3, 4], iso: ["ب", "ت", "ج", "ح", "خ"] }] });
const position = (offset: number, origin = 0): LocalPosition => ({ offset, origin, edits: 0, recentCost: 0, stepCost: 0.5 });
const event = (word: number): PositionEvent => ({ type: "position", v: 1, timeMs: 0, referenceVersion: "test", state: "locked", expected: null,
  observed: [{ word: { surah: 1, ayah: 1, word: word + 1 }, span: { start: { surah: 1, ayah: 1, word: 1 }, end: { surah: 1, ayah: 1, word: word + 1 } }, confidence: 1, cost: 0, per: 0 }] });

describe("private committed word spans", () => {
  it("closes the middle span on the next committed step, allowing a soft entry", () => {
    const run = createWordRun(index), a = event(1), b = event(2), c = event(3);
    run.commit(a, position(1), position(2), { p: "ت", start: 10, end: 12, conf: 1 });
    run.commit(b, position(2), position(3), { p: "ج", start: 20, end: 22, conf: 1 });
    expect(committedWordSpan(b)).toBeUndefined();
    const serialized = JSON.stringify(c);
    run.commit(c, position(3), position(4), { p: "ح", start: 30, end: 32, conf: 1 });
    expect(committedWordSpan(c)).toEqual({ word: 2, frameFrom: 20, frameTo: 30 });
    expect(JSON.stringify(c)).toBe(serialized);
    expect(committedWordSpan(JSON.parse(serialized))).toBeUndefined();
  });

  it.each(["hold", "restart", "backward", "skip"])("rejects neighboring commits across a %s", kind => {
    const run = createWordRun(index);
    run.commit(event(1), position(1), position(2), { p: "ت", start: 10, end: 10, conf: 1 });
    run.commit(event(2), position(2), position(3), { p: "ج", start: 20, end: 20, conf: 1 });
    if (kind === "hold") run.reset();
    if (kind === "restart") run.observe(position(3), position(3, 2));
    if (kind === "backward") run.observe(position(3), position(2));
    if (kind === "skip") run.observe(position(2), position(4));
    const next = event(3);
    run.commit(next, position(3), position(4), { p: "ح", start: 30, end: 30, conf: 1 });
    expect(committedWordSpan(next)).toBeUndefined();
  });

  it("rejects a jump into the middle word and a noncommitted neighbor", () => {
    const run = createWordRun(index);
    run.commit(event(1), position(1), position(2), { p: "ت", start: 10, end: 10, conf: 1 });
    run.commit(event(2), position(2), position(3, 2), { p: "ج", start: 20, end: 20, conf: 1 });
    const next = event(3);
    run.commit(next, position(3, 2), position(4, 2), { p: "ح", start: 30, end: 30, conf: 1 });
    expect(committedWordSpan(next)).toBeUndefined();
  });
});

describe("written and rival fits on the same full span", () => {
  it("allows a soft held-sound difference even when the rival has lower weighted cost", () => {
    expect(wordFit("ااام", "ام")).toEqual({ cost: 4, hard: 0 });
    expect(wordFit("ام", "ام")).toEqual({ cost: 0, hard: 0 });
  });

  it("keeps genuine contradictory consonants in favor of the rival", () => {
    const heard = "قمر";
    expect(wordFit("شمس", heard).hard).toBeGreaterThan(wordFit("قمر", heard).hard);
    expect(wordFit("قمر", heard)).toEqual({ cost: 0, hard: 0 });
  });

  it("charges unmatched consonant tails and uses existing soft confusions", () => {
    expect(wordFit("بت", "بتق")).toEqual({ cost: 10, hard: 1 });
    expect(wordFit("بتق", "بت")).toEqual({ cost: 10, hard: 1 });
    expect(wordFit("ق", "ك")).toEqual({ cost: 5, hard: 0 });
  });
});
