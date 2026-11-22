import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createTracker } from "../src/tracker.js";
import { createChecker } from "../src/checker.js";
import { loadReferenceIndex } from "../src/reference.js";
import { chunk } from "../src/bench/index.js";
import type { QudRecording } from "../src/bench/index.js";
import { finalPosition, replay, syntheticFixture } from "../src/fixtures/index.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8"));
// Student phone evidence from the first 22 seconds of two clean practice openings. No posteriors are needed for
// the false skip warning: the fault is the tracker committing to a briefly separated, incorrect global candidate.
const prefixes = JSON.parse(readFileSync(new URL("fixtures/startup-prefixes.json", import.meta.url), "utf8")) as QudRecording[];

describe("canonical repeats during initial acquisition", () => {
  it.each(prefixes)("keeps $id out of a briefly preferred wrong passage", { timeout: 15000 }, rec => {
    const tracker = createTracker(index, { mode: "free", phrases });
    const checker = createChecker(index, { phrases });
    const positions = [];
    for (const e of chunk(rec, 0.04, false)) {
      const p = tracker.push(e);
      positions.push(...p);
      checker.push(e, p);
    }
    const end = tracker.end();
    positions.push(...end);
    checker.end(end);
    const locks = positions.filter(e => e.type === "position" && e.state === "locked");
    expect(locks.length).toBeGreaterThan(0);
    expect(locks.every(e => e.type === "position" && e.observed[0].word.surah === 26)).toBe(true);
    expect(checker.mistakes()).toEqual([]);
  });

  it("still locates a unique single-push clip", () => {
    const f = syntheticFixture(index, { id: "unique", from: { surah: 108, ayah: 1, word: 1 }, to: { surah: 108, ayah: 3, word: 3 }, chunkSize: 10000 });
    expect(finalPosition(replay(index, f))?.state).toBe("locked");
  });
});
