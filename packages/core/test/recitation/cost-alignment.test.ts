import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { VECTORS } from "./paths";
import { DEFAULT_CONFIG } from "../../src/recitation/config";
import { ALPHABET, charCost, costTable, UNKNOWN_ID } from "../../src/recitation/phonemeCost";
import {
  alignSemiGlobal,
  normalizedDistance,
} from "../../src/recitation/alignment";


function load<T>(name: string): T {
  return JSON.parse(readFileSync(resolve(VECTORS, name), "utf8")) as T;
}

describe("config", () => {
  // Correction-mode tracker knobs are SDK-only; the shared engine keys must
  // still match the reference vector exactly.
  const CORRECTION_ONLY = ["anchorAyahEnd", "backfillRatio", "outsideJumpCost", "skipMargin",
    "skipMaxDistance", "skipMaxHead", "skipMinChars", "stopAlignDistance"];
  it("matches default_config.json", () => {
    const vec = load<Record<string, number>>("default_config.json");
    const shared = Object.fromEntries(Object.entries(DEFAULT_CONFIG).filter(([k]) => !CORRECTION_ONLY.includes(k)));
    expect(shared).toEqual(vec);
    expect(Object.keys(DEFAULT_CONFIG).filter((k) => !(k in vec)).sort()).toEqual(CORRECTION_ONLY);
  });
});

describe("phoneme cost", () => {
  const table = costTable();
  const vec = load<{
    alphabet: string;
    size: number;
    unknownId: number;
    matrix: number[][];
    worked: { heard: string; expected: string; cost: number }[];
  }>("cost_table.json");

  it("alphabet and size", () => {
    expect(ALPHABET).toBe(vec.alphabet);
    expect(table.size).toBe(vec.size);
    expect(table.unknownId).toBe(vec.unknownId);
  });

  it("dense matrix is exact float32", () => {
    for (let i = 0; i < vec.size; i++) {
      for (let j = 0; j < vec.size; j++) {
        expect(table.cost(i, j)).toBe(vec.matrix[i]![j]);
      }
    }
  });

  it("worked char costs", () => {
    for (const row of vec.worked) {
      if (row.heard === "ة") {
        expect(charCost(row.heard, row.expected)).toBe(row.cost);
        expect(table.cost(UNKNOWN_ID, table.id(row.expected))).toBe(1);
      } else {
        expect(charCost(row.heard, row.expected)).toBe(row.cost);
      }
    }
  });
});

describe("alignment", () => {
  const table = costTable();
  const vec = load<{
    pairs: { name: string; a: string; b: string; distance: number }[];
    semiGlobalExample: {
      query: string;
      ref: string;
      headSkipCost: number;
      cost: number;
      distance: number;
      refStart: number;
      refEnd: number;
      queryStart: number;
    };
  }>("alignment.json");

  it("normalizedDistance matches 20 pairs", () => {
    for (const p of vec.pairs) {
      const d = normalizedDistance(table.encode(p.a), table.encode(p.b), table);
      expect(d, p.name).toBe(p.distance);
    }
  });

  it("semi-global BASMALA example", () => {
    const ex = vec.semiGlobalExample;
    const r = alignSemiGlobal(
      table.encode(ex.query),
      table.encode(ex.ref),
      0,
      ex.ref.length,
      table,
      ex.headSkipCost,
    );
    expect(r.cost).toBe(ex.cost);
    expect(r.distance).toBe(ex.distance);
    expect(r.refStart).toBe(ex.refStart);
    expect(r.refEnd).toBe(ex.refEnd);
    expect(r.queryStart).toBe(ex.queryStart);
  });
});
