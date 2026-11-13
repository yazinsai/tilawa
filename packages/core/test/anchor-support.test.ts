import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createLocalSearch } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";
import { independentAnchorSupport } from "../src/tracker.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));

describe("independent destination support", () => {
  it("requires fresh observations to re-establish the destination", () => {
    expect(independentAnchorSupport("كتبل", "كتبل")).toBe(true);
    expect(independentAnchorSupport("كتبل", "بل")).toBe(false);
    expect(independentAnchorSupport("كتبل", "سضظ")).toBe(false);
  });

  it("charges the destination's onset when only its rhyme remains", () => {
    expect(independentAnchorSupport("كتبنذر", "نذر")).toBe(false);
    expect(independentAnchorSupport("كتبنذر", "كتبنذر")).toBe(true);
  });
});

describe("local run observation provenance", () => {
  const first = index.words.find((w) => w.surah === 2 && w.ayah === 255 && w.word === 1)!;
  const make = () => createLocalSearch(index, { offset: first.start, origin: first.start, edits: 0 });

  it("keeps token identities separate from reference offsets and restores them with the path", () => {
    const search = make();
    const text = index.phonemes.slice(first.start, first.end);
    for (let i = 0; i < text.length; i++) search.step(text[i], 100 + i);
    expect(search.position()).toMatchObject({ supportFrom: 100, supportTo: 100 + text.length });
    const saved = search.save();
    const position = search.position();
    search.step("س", 1000);
    search.restore(saved);
    expect(search.position()).toEqual(position);
    search.step(index.phonemes[first.end], 100 + text.length);
    expect(search.position().supportTo).toBe(101 + text.length);
  });

  it("resets run support on a jump without changing scores or selected positions", () => {
    const a = make(), b = make();
    const prefix = index.phonemes.slice(first.start, first.start + 80);
    const restart = index.phonemes.slice(first.start + 20, first.start + 55);
    let token = 0;
    let changedRun = false;
    let previous = first.start;
    for (const p of prefix + restart) {
      a.step(p, token + 500);
      b.step(p, token);
      const { supportFrom: af, supportTo: at, ...ap } = a.position();
      const { supportFrom: bf, supportTo: bt, ...bp } = b.position();
      expect(ap).toEqual(bp);
      expect(af).toBe(bf + 500);
      expect(at).toBe(bt + 500);
      if (ap.origin !== previous) {
        changedRun = true;
        expect(af).toBeGreaterThan(500);
      }
      previous = ap.origin;
      token++;
    }
    expect(changedRun).toBe(true);
  });
});
