import { describe, expect, it } from "vitest";
import { transcriptionLattice, warningEventMass } from "../src/event-lattice.js";

const rows = (ps: number[][]) => ps.map(p => Float64Array.from(p, Math.log));
const limits = { prefixes: 1000, labels: 3, frames: 20 };

describe("reference-free event posterior", () => {
  it("matches exhaustive raw CTC paths, including blank-separated repeated labels", () => {
    const ps = [[.2,.5,.3],[.4,.1,.5],[.3,.4,.3]];
    const expected = new Map<string, number>();
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) for (let c = 0; c < 3; c++) {
      const path = [a,b,c], text = path.filter((id, j) => id && (!j || id !== path[j-1])).map(id => "بت"[id-1]).join("");
      expected.set(text, (expected.get(text) ?? 0) + ps[0][a]*ps[1][b]*ps[2][c]);
    }
    const actual = transcriptionLattice(rows(ps), ["ب","ت"], 7, limits);
    expect(actual.length).toBe(expected.size);
    for (const t of actual) {
      expect(Math.exp(t.logMass)).toBeCloseTo(expected.get(t.phonemes.map(p => p.p).join(""))!, 12);
      expect(t.phonemes.every(p => p.start >= 7 && p.end < 10)).toBe(true);
    }
    expect(actual.reduce((sum, t) => sum + Math.exp(t.logMass), 0)).toBeCloseTo(1, 12);
  });

  it("pools individually weak quiet strings instead of choosing a transcription", () => {
    const lattice = transcriptionLattice(rows([[.1,.45,.25,.2]]), ["ب","ت","ن"], 0, { ...limits, labels: 4 });
    const mass = warningEventMass(lattice, phones => phones[0]?.p === "ب" ? ["wrong_word"] : []);
    expect(lattice[0].phonemes[0].p).toBe("ب");
    expect(mass.quiet).toBeCloseTo(.55);
    expect(mass.warning).toBeCloseTo(.45);
  });

  it("different warning classes and rivals contribute to any-warning mass", () => {
    const lattice = transcriptionLattice(rows([[.2,.4,.4]]), ["ب","ت"], 0, limits);
    const mass = warningEventMass(lattice, phones => phones.length ? [phones[0].p === "ب" ? "wrong_word" : "added_word"] : []);
    expect(mass.warning).toBeCloseTo(.8);
    expect(mass.quiet).toBeCloseTo(.2);
  });

  it("reports discarded probability separately and abstains beyond the frozen frame bound", () => {
    const lattice = transcriptionLattice(rows([[.2,.4,.4]]), ["ب","ت"], 0, { ...limits, prefixes: 1 });
    expect(warningEventMass(lattice, () => []).retained).toBeCloseTo(.4);
    expect(transcriptionLattice(rows([[1,0,0],[1,0,0]]), ["ب","ت"], 0, { ...limits, frames: 1 })).toEqual([]);
  });
});
