import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createLocalSearch } from "../../src/engine/local.js";
import { loadReferenceIndex } from "../../src/engine/index.js";
import type { ReferenceAsset } from "../../src/engine/index.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);

describe("createLocalSearch: wordMatches", () => {
  it("is counted only when asked for (refinements.confirmAdvance), and nothing else changes", () => {
    // 36:6 said with one sound changed, from its first word.
    const first = index.words.find((w) => w.surah === 36 && w.ayah === 6)!;
    const last = index.words.filter((w) => w.surah === 36 && w.ayah === 6).at(-1)!;
    const said = [...index.phonemes.slice(first.start, last.end)];
    said[5] = said[5] === "ز" ? "ظ" : "ز";
    const start = { offset: first.start, origin: first.start, edits: 0 };
    const counting = createLocalSearch(index, start);
    const not = createLocalSearch(index, start, { countMatches: false });
    let counted = 0;
    for (const p of said) {
      counting.step(p);
      not.step(p);
      const [a, b] = [counting.position(), not.position()];
      expect(b.wordMatches).toBe(0);
      expect({ ...b, wordMatches: a.wordMatches }).toEqual(a);
      counted += a.wordMatches;
    }
    expect(counted).toBeGreaterThan(0);
  });
});
