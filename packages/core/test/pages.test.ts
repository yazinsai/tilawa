import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex, pageRange } from "../src/reference.js";
import type { PageAsset } from "../src/reference.js";

const raw = readFileSync(new URL("../assets/madani-pages-v1.json", import.meta.url), "utf8");
const pages: PageAsset = JSON.parse(raw);
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));

describe("Madani page table", () => {
  it("is the recorded asset", () => {
    expect(createHash("sha256").update(raw).digest("hex")).toBe("abdb0075dedab5f1ced1c3429e78b66c6101b0caed114aa417fd7cc7b999df6f");
    expect(pages.source).toMatch(/tanzil/);
  });

  it("has 604 contiguous pages covering 1:1..114:6", () => {
    const ayahs = [...new Set(index.words.map((w) => `${w.surah}:${w.ayah}`))];
    const covered = pages.pages.flatMap((_, i) => {
      const { from, to } = pageRange(pages, i + 1);
      return ayahs.slice(ayahs.indexOf(`${from.surah}:${from.ayah}`), ayahs.indexOf(`${to.surah}:${to.ayah}`) + 1);
    });
    expect(pages.pages).toHaveLength(604);
    expect(covered).toEqual(ayahs);
  });

  it("knows well-known page starts", () => {
    expect(pageRange(pages, 2)).toEqual({ from: { surah: 2, ayah: 1 }, to: { surah: 2, ayah: 5 } });
    expect(pageRange(pages, 50).from).toEqual({ surah: 3, ayah: 1 });
    expect(pageRange(pages, 604)).toEqual({ from: { surah: 112, ayah: 1 }, to: { surah: 114, ayah: 6 } });
    expect(() => pageRange(pages, 605)).toThrow();
  });
});
