import { describe, expect, it } from "vitest";
import type { ReferenceIndex, WrongWordRecheck } from "@tilawa/core";
import { eventPosteriorRecheck } from "../src/event-posterior.js";

const index = { tokens: new Set(["ب", "ت"]) } as ReferenceIndex;
const candidate: WrongWordRecheck = { word: { surah: 1, ayah: 1, word: 1 }, observed: "ب",
  fromFrame: 4, toFrame: 5, phonemes: [{ p: "ب", start: 4, end: 4, conf: .4 }],
  rows: [Float64Array.from([.25, .4, .35], Math.log)] };

describe("enlarged event lattice veto", () => {
  it("requires a quiet majority to survive enlargement", () => {
    const limits = { prefixes: 1, labels: 3, frames: 5 };
    const enlarged = { ...limits, prefixes: 10 };
    // The smallest beam is quiet, but the union of two warning-producing explanations wins.
    const recheck = eventPosteriorRecheck(index, undefined, limits, enlarged);
    expect(recheck(candidate, phones => phones[0]?.p === "ب" ? [] : ["wrong_word"])).toBe(false);
  });

  it("counts alternative warning classes against a veto", () => {
    const recheck = eventPosteriorRecheck(index);
    expect(recheck(candidate, phones => !phones.length ? [] :
      [phones[0].p === "ب" ? "wrong_word" : "added_word"])).toBe(false);
  });

  it("vetoes when multiple quiet strings form a stable majority", () => {
    const recheck = eventPosteriorRecheck(index);
    expect(recheck(candidate, phones => phones[0]?.p === "ب" ? ["wrong_word"] : [])).toBe(true);
  });
});
