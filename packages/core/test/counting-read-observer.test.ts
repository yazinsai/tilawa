import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import type { CountingReadVisit } from "../src/checker.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture } from "../src/fixtures/index.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const from = { surah: 1, ayah: 2, word: 1 }, to = { surah: 1, ayah: 4, word: 3 };
describe("counting visit observation", () => {
  it("exposes complete matching visits without changing checker decisions", () => {
    const fixture = syntheticFixture(index, { id: "read-observer", from, to, chunkSize: 5 });
    const seen: CountingReadVisit[] = [];
    const observed = createChecker(index, { passage: { from, to }, onCountingVisit(v) { seen.push(v); } });
    const original = createChecker(index, { passage: { from, to } });
    const a = [], b = [];
    for (const e of fixture.evidence) { a.push(...observed.push(e, [])); b.push(...original.push(e, [])); }
    a.push(...observed.end()); b.push(...original.end());
    expect(a).toEqual(b);
    expect(observed.mistakes()).toEqual(original.mistakes());
    expect(seen.some(v => v.hard === 0 && v.toFrame > v.fromFrame)).toBe(true);
  });
});
