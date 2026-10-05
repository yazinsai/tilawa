import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "../src/index.js";
import type { ReferenceAsset, WordRef } from "../src/index.js";
import { syntheticFixture } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";
import { relabelClip, summarize } from "../src/relabel/index.js";
import type { Pass1Clip } from "../src/relabel/index.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../assets/ref-v1.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const lastWord = (s: number, a: number) => asset.ayahs.find((x) => x.s === s && x.a === a)!.w.length;

let row = 0;
const clip = (label: { surah: number; ayah: number } | null, from: WordRef, to: WordRef, edits: Edit[] = []): Pass1Clip => ({
  shard: "test",
  row: row++,
  path: "x.wav",
  status: "ok",
  filenameLabel: label,
  evidence: syntheticFixture(index, { id: "t", from, to, edits }).evidence[0],
});
const failed = (status: "decode_failed" | "too_long"): Pass1Clip => ({
  shard: "test",
  row: row++,
  path: "x.wav",
  status,
  filenameLabel: { surah: 100, ayah: 1 },
  evidence: null,
});

describe("relabelClip", () => {
  it("proposes the matched span for a clean clip, with no exclusion", () => {
    const l = relabelClip(index, clip({ surah: 100, ayah: 6 }, w(100, 6, 1), w(100, 6, lastWord(100, 6))));
    expect(l.labelKind).toBe("proposed");
    expect(l.state).toBe("locked");
    expect(l.exclusion).toBeNull();
    expect(l.proposed).toEqual({ start: w(100, 6, 1), end: w(100, 6, lastWord(100, 6)) });
    expect(l.filenameLabel).toEqual({ surah: 100, ayah: 6 });
    expect(l.pattern).toBe("filename ayah");
    expect(l.agrees).toBe(true);
    expect(l.per).toBe(0);
    expect(l.fit).toBeGreaterThan(0);
    expect(l.separation).toBeGreaterThan(0.5);
  });

  it("reports an offset filename label", () => {
    const l = relabelClip(index, clip({ surah: 100, ayah: 5 }, w(100, 6, 1), w(100, 6, lastWord(100, 6))));
    expect(l.pattern).toBe("off by +1 ayah");
    expect(l.agrees).toBe(false);
  });

  it("reports a clip containing two ayahs", () => {
    const l = relabelClip(index, clip({ surah: 100, ayah: 6 }, w(100, 6, 1), w(100, 7, lastWord(100, 7))));
    expect(l.state).toBe("locked");
    expect(l.pattern).toBe("contains 2 ayahs (+0..+1)");
  });

  it("excludes an ambiguous clip and lists its competing candidates", () => {
    const l = relabelClip(index, clip({ surah: 2, ayah: 1 }, w(2, 1, 1), w(2, 1, lastWord(2, 1))));
    expect(l.exclusion).toBe("ambiguous");
    expect(l.proposed).toBeNull();
    expect(l.candidates.length).toBeGreaterThan(1);
  });

  it("excludes noise as lost", () => {
    const l = relabelClip(index, clip({ surah: 100, ayah: 1 }, w(100, 1, 1), w(100, 1, 1), [
      { kind: "del", at: 0, count: 1000 },
      { kind: "noise", at: 0, length: 60, seed: 7 },
    ]));
    expect(l.exclusion).toBe("lost");
  });

  it("excludes too-short evidence as insufficient", () => {
    const l = relabelClip(index, clip({ surah: 100, ayah: 1 }, w(100, 1, 1), w(100, 1, 1)));
    expect(l.exclusion).toBe("insufficient_evidence");
  });

  it("excludes an empty decode as insufficient", () => {
    const c = clip({ surah: 100, ayah: 1 }, w(100, 1, 1), w(100, 1, 1));
    const l = relabelClip(index, { ...c, evidence: { ...c.evidence!, phonemes: [] } });
    expect(l.exclusion).toBe("insufficient_evidence");
    expect(l.state).toBeNull();
  });

  it("passes failed pass1 rows through as exclusions", () => {
    expect(relabelClip(index, failed("decode_failed")).exclusion).toBe("decode_failed");
    expect(relabelClip(index, failed("too_long")).exclusion).toBe("too_long");
  });

  it("is deterministic", () => {
    const c = clip({ surah: 100, ayah: 6 }, w(100, 6, 1), w(100, 7, 3), [{ kind: "sub", at: 4, to: "ق" }]);
    expect(JSON.stringify(relabelClip(index, c))).toBe(JSON.stringify(relabelClip(index, c)));
  });
});

describe("summarize", () => {
  it("counts outcomes, agreement and offset patterns, and gives every excluded clip one reason", () => {
    const labels = [
      relabelClip(index, clip({ surah: 100, ayah: 6 }, w(100, 6, 1), w(100, 6, lastWord(100, 6)))),
      relabelClip(index, clip({ surah: 100, ayah: 5 }, w(100, 6, 1), w(100, 6, lastWord(100, 6)))),
      relabelClip(index, clip({ surah: 2, ayah: 1 }, w(2, 1, 1), w(2, 1, lastWord(2, 1)))),
      relabelClip(index, failed("too_long")),
    ];
    const s = summarize(labels);
    expect(s.labelKind).toBe("proposed");
    expect(s.clips).toBe(4);
    expect(s.outcomes).toEqual({ locked: 2, ambiguous: 1, too_long: 1 });
    expect(s.agreement).toEqual({ locked: 2, agree: 1, rate: 0.5 });
    expect(s.patterns).toEqual({ "filename ayah": 1, "off by +1 ayah": 1 });
    for (const l of labels) expect(l.state === "locked" ? l.exclusion === null : l.exclusion !== null).toBe(true);
  });
});
