import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createSteppedAyahs } from "../src/stepped-ayahs.js";
import type { LocalPosition } from "../src/local.js";
import { createTracker } from "../src/tracker.js";
import { syntheticFixture } from "../src/fixtures/index.js";
import { loadReferenceIndex } from "../src/reference.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const words = index.words.filter(w => w.surah === 55 && w.ayah === 29);
const position = (offset: number, origin = words[0].start, stepCost = 0, edits = 0): LocalPosition =>
  ({ offset, origin, stepCost, edits, recentCost: stepCost });

const complete = (coverage: ReturnType<typeof createSteppedAyahs>, word: typeof words[number], costAt = -1) => {
  for (let at = word.start; at < word.end; at++) {
    coverage.step(position(at), position(at + 1, words[0].start, at === costAt ? 0.5 : 0));
  }
};

describe("committed zero-edit ayah coverage", () => {
  it("exports only after two consecutive next-word entries of the same ayah", () => {
    const c = createSteppedAyahs(index);
    c.step(position(words[0].start), position(words[0].start + 1));
    expect(c.take()).toEqual([]);
    c.step(position(words[1].start), position(words[1].start + 1));
    expect(c.take()).toEqual([{ surah: 55, ayah: 29 }]);
    complete(c, words[2]);
    expect(c.take()).toEqual([]);
  });

  it("does not combine entries across a cost-bearing word entry", () => {
    const c = createSteppedAyahs(index);
    complete(c, words[0], words[0].start);
    complete(c, words[1]);
    expect(c.take()).toEqual([]);
    complete(c, words[2]);
    expect(c.take()).toEqual([{ surah: 55, ayah: 29 }]);
  });

  it("does not count a landing word as an in-order step", () => {
    const c = createSteppedAyahs(index);
    complete(c, words[0]);
    c.step(position(words[0].end), position(words[1].start + 1, words[1].start));
    for (let at = words[1].start + 1; at < words[1].end; at++) {
      c.step(position(at, words[1].start), position(at + 1, words[1].start));
    }
    complete(c, words[2]);
    expect(c.take()).toEqual([]);
  });

  it("does not combine nonconsecutive words or words from separate runs", () => {
    const c = createSteppedAyahs(index);
    complete(c, words[0]);
    complete(c, words[2]);
    expect(c.take()).toEqual([]);
    c.reset();
    complete(c, words[1]);
    expect(c.take()).toEqual([]);
  });

  it("does not join a pair across an ayah boundary", () => {
    const c = createSteppedAyahs(index);
    const previous = index.words.filter(w => w.surah === 55 && w.ayah === 28).at(-1)!;
    complete(c, previous);
    complete(c, words[0]);
    expect(c.take()).toEqual([]);
  });

  it("does not witness an insertion, deletion, or a path edit", () => {
    for (const after of [position(words[1].start), position(words[1].start + 2), position(words[1].start + 1, words[0].start, 0, 1)]) {
      const c = createSteppedAyahs(index);
      complete(c, words[0]);
      c.step(position(words[1].start), after);
      complete(c, words[2]);
      expect(c.take()).toEqual([]);
    }
  });

  it("carries coverage on existing tracker events and leaves a genuinely skipped ayah unnoted", () => {
    const from = { surah: 55, ayah: 28, word: 1 };
    const to = { surah: 55, ayah: 31, word: 3 };
    const run = (skip: boolean) => {
      const fixture = syntheticFixture(index, { id: "coverage", from, to, chunkSize: 1,
        edits: skip ? [{ kind: "del", at: words[0].start - index.words.find(w => w.surah === 55 && w.ayah === 28)!.start,
          count: words.at(-1)!.end - words[0].start }] : [] });
      const t = createTracker(index);
      return [...fixture.evidence.flatMap(e => t.push(e)), ...t.end()].flatMap(e => e.type === "position" ? e.steppedAyahs ?? [] : []);
    };
    expect(run(false)).toContainEqual({ surah: 55, ayah: 29 });
    expect(run(true)).not.toContainEqual({ surah: 55, ayah: 29 });
  });
});
