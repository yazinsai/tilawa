import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "../src/index.js";
import type { ReferenceAsset, WordRef } from "../src/index.js";
import { finalPosition, replay, syntheticFixture } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../assets/ref-v1.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const ayah = (s: number, a: number) => asset.ayahs.find((x) => x.s === s && x.a === a)!;

function locate(from: WordRef, to: WordRef, edits: Edit[]) {
  const final = finalPosition(replay(index, syntheticFixture(index, { id: "t", from, to, edits }), { trace: true }))!;
  return { final, top: final.observed[0], path: final.trace!.path! };
}

describe("fuzzy locate", () => {
  // 100:6 ends "كَنُۥۥدڇ"; dropping one ۥ shortens the madd.
  const madd = ayah(100, 6).p.lastIndexOf("ۥ");
  const consonant = ayah(100, 6).p.lastIndexOf("د");
  const whole = [w(100, 6, 1), w(100, 6, 4)] as const;

  it("charges a madd-length change less than a consonant deletion but counts both in the PER", () => {
    const shortened = locate(...whole, [{ kind: "del", at: madd, count: 1 }]);
    const dropped = locate(...whole, [{ kind: "del", at: consonant, count: 1 }]);
    for (const r of [shortened, dropped]) {
      expect(r.final.state).toBe("locked");
      expect(r.top.span).toEqual({ start: whole[0], end: whole[1] });
    }
    expect(shortened.top.cost).toBeGreaterThan(0);
    expect(shortened.top.cost).toBeLessThan(dropped.top.cost);
    expect(shortened.top.per).toBeGreaterThan(0);
    for (const r of [shortened, dropped]) expect(r.path.filter((o) => o.op !== "match")).toHaveLength(1);
  });

  it("charges a madd extension cheaply but still counts it in the PER", () => {
    const extended = locate(...whole, [{ kind: "ins", at: madd, text: "ۥ" }]);
    const inserted = locate(...whole, [{ kind: "ins", at: madd, text: "ك" }]);
    expect(extended.top.cost).toBeLessThan(inserted.top.cost);
    expect(extended.top.per).toBe(inserted.top.per);
    expect(extended.path.filter((o) => o.op === "ins")).toEqual([expect.objectContaining({ obs: "ۥ" })]);
  });

  it("locates 100:1 with ض→د, reporting a nonzero PER and the substitution in the edit path", () => {
    const at = ayah(100, 1).p.indexOf("ض");
    const r = locate(w(100, 1, 1), w(100, 1, 2), [{ kind: "sub", at, to: "د" }]);
    expect(r.final.state).toBe("locked");
    expect(r.top.span).toEqual({ start: w(100, 1, 1), end: w(100, 1, 2) });
    expect(r.top.per).toBeGreaterThan(0);
    expect(r.path.filter((o) => o.op !== "match")).toEqual([{ op: "sub", ref: "ض", obs: "د", refAt: at, obsAt: at }]);
  });

  it("exposes alignment cost and PER as separate fields", () => {
    const { top } = locate(w(100, 1, 1), w(100, 1, 2), [{ kind: "sub", at: ayah(100, 1).p.indexOf("ض"), to: "د" }]);
    expect(top).toHaveProperty("cost");
    expect(top).toHaveProperty("per");
    expect(top.cost).not.toBe(top.per);
  });

  it("matches only from a word start", () => {
    const start = index.words.find((x) => x.surah === 2 && x.ayah === 255 && x.word === 3)!.start;
    const { top } = locate(w(2, 255, 3), w(2, 255, 12), [{ kind: "del", at: 0, count: 2 }]);
    expect(top.span.start).toEqual(w(2, 255, 3));
    expect(index.words.some((x) => x.start === start)).toBe(true);
  });
});
