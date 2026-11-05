import { describe, expect, it } from "vitest";
import type { DecodedPhoneme } from "@tilawa/core";
import { preserveSeamGrids } from "../src/gap-decode.js";

describe("full-stream grids survive cold-gap token replacement", () => {
  it("moves removed carriers onto surviving tokens without changing phones", () => {
    const p = (start: number): DecodedPhoneme => ({ p: "ن", start, end: start, conf: 0.9 });
    const before = [
      { ...p(1), seamGrid: { startFrame: 0, ids: [0, 1, 0, 0] } },
      { ...p(5), seamGrid: { startFrame: 4, ids: [0, 1, 0, 0] } },
      { ...p(9), seamGrid: { startFrame: 8, ids: [0, 1] } },
    ];
    const after = [p(5), before[2]];
    const kept = preserveSeamGrids(before, after);
    expect(kept.map(({ seamGrid: _, ...p }) => p)).toEqual(after.map(({ seamGrid: _, ...p }) => p));
    expect(kept.map(p => p.seamGrid)).toEqual([
      { startFrame: 0, ids: [0, 1, 0, 0, 0, 1, 0, 0] },
      { startFrame: 8, ids: [0, 1] },
    ]);
    expect(before[0].seamGrid.ids).toEqual([0, 1, 0, 0]);
  });
});
