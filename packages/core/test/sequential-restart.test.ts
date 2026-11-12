import { describe, expect, it } from "vitest";
import { createRestartEvidence, fitRestartChannel, restartKernel, restartRecordingLogP } from "../src/sequential-restart.js";
import type { RestartChannel, RestartGeometry } from "../src/sequential-restart.js";
import { UNIT } from "../src/fuzzy.js";
const phones = ["ب", "ت", "ث", "د"];
const channel: RestartChannel = { phones, rows: phones.map((_, i) => [...phones.map((_, j) => i === j ? 0.98 : 0.005), 0.005]),
  insertion: 0.01, inserted: phones.map(() => 0.25) };
const geometry = (reference = "بببببتتتتتثثثثثددددد"): RestartGeometry => ({ reference, lo: 0, hi: reference.length,
  starts: [0, 5, 10, 15].filter(at => at < reference.length), movement: () => 2 * UNIT, entry: () => 0 });
const mass = (xs: Float64Array) => xs.reduce((total, p) => total + Math.exp(p), 0);

describe("normalized sequential restart evidence", () => {
  it("assigns total probability one to the next token, including deletion paths and window overflow", () => {
    const g = geometry(), step = restartKernel(channel, g);
    for (const at of [0, 4, g.hi]) {
      const seed = new Float64Array(g.hi + 1).fill(-Infinity); seed[at] = 0;
      expect(phones.reduce((total, p) => total + mass(step(seed, p)), 0)).toBeCloseTo(1, 12);
    }
  });
  it("marginalizes restart times under one proper prior, rather than adding another full trial at each look", () => {
    const c: RestartChannel = { phones: ["ب"], rows: [[0.8, 0.2]], insertion: 0.1, inserted: [1] };
    const g: RestartGeometry = { reference: "بببب", lo: 0, hi: 4, starts: [0, 1, 2, 3], movement: () => 2 * UNIT, entry: () => 0 };
    const e = createRestartEvidence({ channel: c, threshold: 0 }, g, 0, 2 * UNIT);
    for (let t = 1; t <= 12; t++) {
      e.step("ب", 0);
      const restartMass = e.comparisons().reduce((s, c) => s + Math.exp(c.logOdds + 6), 0);
      expect(restartMass).toBeCloseTo(t / (t + 1), 12);
      expect(e.comparisons().every(c => !c.admitted)).toBe(true);
    }
  });
  it("counts each destination once even if the structural list contains duplicates", () => {
    const g = geometry();
    const unique = createRestartEvidence({ channel, threshold: 0 }, g, 0, 2 * UNIT);
    const duplicate = createRestartEvidence({ channel, threshold: 0 }, { ...g, starts: [...g.starts, ...g.starts] }, 0, 2 * UNIT);
    for (const p of "ببتد") { unique.step(p, 0); duplicate.step(p, 0); }
    expect(duplicate.comparisons()).toEqual(unique.comparisons());
    expect(duplicate.comparison(g.hi + 1).supported).toBe(false);
  });
  it("does not consume evidence when a comparison is read again", () => {
    const e = createRestartEvidence({ channel, threshold: 0 }, geometry(), 0, 2 * UNIT);
    e.step("ب", 0);
    const first = e.comparison(15);
    for (let i = 0; i < 10; i++) expect(e.comparison(15)).toEqual(first);
    expect(first.tokens).toBe(1);
  });
  it("accumulates destination-specific support for a genuine jump", () => {
    const e = createRestartEvidence({ channel, threshold: 0 }, geometry(), 0, 2 * UNIT);
    for (let i = 0; i < 3; i++) e.step("ب", i);
    const before = e.comparison(15);
    for (let i = 0; i < 5; i++) e.step("د", 3);
    expect(e.comparison(15).logOdds).toBeGreaterThan(before.logOdds);
    expect(e.comparison(15).admitted).toBe(true);
    expect(e.comparison(10).logOdds).toBeLessThan(e.comparison(15).logOdds);
  });
  it("does not commit a transient one-token destination advantage contradicted by continuation", () => {
    const e = createRestartEvidence({ channel, threshold: 0 }, geometry(), 0, 2 * UNIT);
    for (let i = 0; i < 5; i++) e.step("ب", i);
    e.step("د", 5);
    const transient = e.comparison(15);
    expect(transient.increment).toBeGreaterThan(0);
    expect(transient.admitted).toBe(false);
    for (let i = 0; i < 4; i++) e.step("ت", 5 + i);
    expect(e.comparison(15).logOdds).toBeLessThan(transient.logOdds);
    expect(e.comparison(15).admitted).toBe(false);
  });
  it("rewinds the evidence process exactly without mutating a retained snapshot", () => {
    const e = createRestartEvidence({ channel, threshold: 0 }, geometry(), 0, 2 * UNIT);
    e.step("ب", 0); const saved = e.save();
    const tail = () => { for (const p of "بتد") e.step(p, 1); return e.comparisons(); };
    const first = tail(); e.restore(saved); expect(tail()).toEqual(first);
  });
  it("freezes clean-training probabilities in normalized owned rows", () => {
    const records = [{ written: "بتثد", heard: "بتثد" }, { written: "بتثد", heard: "بتد" }];
    const fitted = fitRestartChannel(records, phones, 4);
    for (const row of [...fitted.rows, fitted.inserted]) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(restartRecordingLogP(fitted, records[0])).toBeGreaterThan(restartRecordingLogP(fitted, { written: "دددد", heard: "بتثد" }));
    expect(() => restartKernel({ ...fitted, inserted: [1, 1, 1, 1] }, geometry())).toThrow("Unnormalized");
  });
});
