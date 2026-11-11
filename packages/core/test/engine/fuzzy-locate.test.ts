import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createTracker, loadReferenceIndex } from "../../src/engine/index.js";
import type { PhonemeEvidence, PositionEvent, ReferenceAsset, WordRef } from "../../src/engine/index.js";
import { finalPosition, replay, syntheticFixture } from "../fixtures/synthetic.js";
import type { Edit } from "../fixtures/synthetic.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
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

  it("charges a shortened ghunna like a madd change but counts it in the PER", () => {
    const ghunna = ayah(100, 6).p.indexOf("ںںں");
    const shortened = locate(...whole, [{ kind: "del", at: ghunna, count: 1 }]);
    const dropped = locate(...whole, [{ kind: "del", at: consonant, count: 1 }]);
    expect(shortened.top.cost).toBeLessThan(dropped.top.cost);
    expect(shortened.path.filter((o) => o.op !== "match")).toEqual([expect.objectContaining({ op: "del", ref: "ں" })]);
  });

  it("charges a dropped plain noon in full but a shortened ghunna run cheaply", () => {
    // Noon and meem runs are 1 (plain) or 4 (mushaddad with ghunna) in this inventory.
    const text = ayah(100, 6).p;
    const plain = text.search(/(?<!ن)ن(?!ن)/);
    // `cost` is per observed phoneme, so this is the cost in plain edits.
    const edits = (r: ReturnType<typeof locate>) => r.top.cost * r.path.filter((o) => o.op !== "del").length;
    expect(edits(locate(...whole, [{ kind: "del", at: plain, count: 1 }]))).toBeCloseTo(1);
    expect(edits(locate(...whole, [{ kind: "del", at: text.indexOf("نننن"), count: 1 }]))).toBeCloseTo(0.2);
    expect(edits(locate(...whole, [{ kind: "del", at: ayah(100, 6).p.indexOf("ںںں"), count: 1 }]))).toBeCloseTo(0.2);
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

  it("locates a clip cut mid-word to the word it starts inside", () => {
    const { final, top } = locate(w(2, 255, 3), w(2, 255, 12), [{ kind: "del", at: 0, count: 2 }]);
    expect(final.state).toBe("locked");
    expect(top.span).toEqual({ start: w(2, 255, 3), end: w(2, 255, 12) });
    expect(top.per).toBe(0);
  });
});

const evidence = (text: string, conf: (i: number) => number = () => 0.99): PhonemeEvidence => ({
  inventory: index.inventoryId,
  frameMs: 20,
  phonemes: [...text].map((p, i) => ({ p, start: i * 3, end: i * 3 + 1, conf: conf(i) })),
});
const textOf = (from: WordRef, to: WordRef) => {
  const find = (r: WordRef) => index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
  return index.phonemes.slice(find(from).start, find(to).end);
};
const run = (e: PhonemeEvidence) => {
  const t = createTracker(index, { trace: true });
  t.push(e);
  return t.end().at(-1) as PositionEvent;
};

describe("clip edges", () => {
  it("drops low-confidence leading phonemes instead of stretching the span backwards", () => {
    const junk = "يرَب";
    const e = run(evidence(junk + textOf(w(100, 1, 1), w(100, 1, 2)), (i) => (i < junk.length ? 0.2 : 0.99)));
    expect(e.state).toBe("locked");
    expect(e.observed[0].span).toEqual({ start: w(100, 1, 1), end: w(100, 1, 2) });
    expect(e.observed[0].per).toBeGreaterThan(0);
    expect(e.trace!.path!.slice(0, junk.length).every((o) => o.op === "ins")).toBe(true);
  });

  it("keeps confident leading speech as part of the span", () => {
    const e = run(evidence(textOf(w(99, 8, 6), w(100, 1, 2))));
    expect(e.observed[0].span).toEqual({ start: w(99, 8, 6), end: w(100, 1, 2) });
  });

  it("locates the passage after a leading basmala and flags the prefix", () => {
    const e = run(evidence(textOf(w(1, 1, 1), w(1, 1, 4)) + textOf(w(100, 1, 1), w(100, 2, 2))));
    expect(e.state).toBe("locked");
    expect(e.observed[0]).toMatchObject({ span: { start: w(100, 1, 1), end: w(100, 2, 2) }, per: 0, prefix: "basmala" });
  });

  it("still reads the basmala of al-Fatiha as 1:1", () => {
    const e = run(evidence(textOf(w(1, 1, 1), w(1, 2, 4))));
    expect(e.state).toBe("locked");
    expect(e.observed[0].span).toEqual({ start: w(1, 1, 1), end: w(1, 2, 4) });
    expect(e.observed[0].prefix).toBeUndefined();
  });
});
