import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "../src/index.js";
import type { ReferenceAsset, WordRef, WordSpan } from "../src/index.js";
import { syntheticFixture } from "../src/fixtures/index.js";
import { relabelClip } from "../src/relabel/index.js";
import type { Pass1Clip, ProposedLabel } from "../src/relabel/index.js";
import { buildManifests, bucketClip, textOnlyLabel } from "../src/buckets/index.js";
import type { BucketCutoffs, ManifestClip } from "../src/buckets/index.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const lastWord = (s: number, a: number) => asset.ayahs.find((x) => x.s === s && x.a === a)!.w.length;
const ayah = (s: number, a: number): WordSpan => ({ start: w(s, a, 1), end: w(s, a, lastWord(s, a)) });

let row = 0;
const relabel = (label: { surah: number; ayah: number }, span: WordSpan): ProposedLabel => {
  const clip: Pass1Clip = {
    shard: "test",
    row: row++,
    path: "x.wav",
    status: "ok",
    filenameLabel: label,
    evidence: syntheticFixture(index, { id: "t", from: span.start, to: span.end, edits: [] }).evidence[0],
  };
  return relabelClip(index, clip);
};

describe("textOnlyLabel", () => {
  it("labels the ar-Rahman refrain by its text, listing every occurrence it was confused with", () => {
    const l = relabel({ surah: 55, ayah: 13 }, ayah(55, 13));
    expect(l.exclusion).toBe("ambiguous");
    const t = textOnlyLabel(index, l)!;
    expect(t.text).toBe(asset.ayahs.find((x) => x.s === 55 && x.a === 13)!.p);
    expect(t.occurrences.length).toBe(l.candidates.length);
    expect(t.occurrences).toContainEqual(ayah(55, 13));
  });

  it("gives no text label when the candidates' phonemes differ", () => {
    const l = relabel({ surah: 55, ayah: 13 }, ayah(55, 13));
    const mixed = { ...l, candidates: [...l.candidates.slice(0, 2), { span: ayah(55, 14), cost: 0.1, per: 0.1 }] };
    expect(textOnlyLabel(index, mixed)).toBeNull();
  });

  it("only applies to ambiguous clips", () => {
    const l = relabel({ surah: 100, ayah: 6 }, ayah(100, 6));
    expect(l.state).toBe("locked");
    expect(textOnlyLabel(index, l)).toBeNull();
  });
});

describe("bucketClip", () => {
  const cutoffs: BucketCutoffs = { minConfidence: 0.2, cleanMaxPer: 0.03, cleanMinTeacherConf: 0.9, deviantMaxPer: 0.3, rejectBelowTeacherConf: 0.5 };
  const confident = { meanConf: 0.95, durationS: 4 };
  const locked = relabel({ surah: 100, ayah: 6 }, ayah(100, 6));
  const withPer = (per: number, confidence = locked.confidence) => ({ ...locked, per, confidence });

  it("puts a confident, near-exact lock in clean-match with its span as the label", () => {
    const b = bucketClip(index, locked, confident, cutoffs);
    expect(b.bucket).toBe("clean-match");
    expect(b.label).toEqual({ kind: "span", span: ayah(100, 6) });
  });

  it("puts a located clip with local mismatches in deviant", () => {
    expect(bucketClip(index, withPer(0.1), confident, cutoffs).bucket).toBe("deviant");
  });

  it("keeps an exact match out of clean-match when the teacher was unsure of it", () => {
    expect(bucketClip(index, locked, { ...confident, meanConf: 0.7 }, cutoffs).bucket).toBe("deviant");
  });

  it("rejects a lock too far from its passage to be a recitation of it", () => {
    expect(bucketClip(index, withPer(0.5), confident, cutoffs)).toMatchObject({ bucket: "reject", reason: "high_per" });
  });

  it("calls a lock under the adopted confidence ambiguous, without a label", () => {
    const b = bucketClip(index, withPer(0, 0.1), confident, cutoffs);
    expect(b).toMatchObject({ bucket: "ambiguous", reason: "low_confidence", label: null });
  });

  it("recovers a refrain clip into clean-match by its text", () => {
    const l = relabel({ surah: 55, ayah: 13 }, ayah(55, 13));
    const b = bucketClip(index, l, confident, cutoffs);
    expect(b.bucket).toBe("clean-match");
    expect(b.label).toMatchObject({ kind: "text", text: asset.ayahs.find((x) => x.s === 55 && x.a === 13)!.p });
  });

  it("leaves an ambiguous clip with differing candidates in ambiguous", () => {
    const l = relabel({ surah: 55, ayah: 13 }, ayah(55, 13));
    const mixed = { ...l, candidates: [...l.candidates.slice(0, 2), { span: ayah(55, 14), cost: 0.1, per: 0.1 }] };
    expect(bucketClip(index, mixed, confident, cutoffs)).toMatchObject({ bucket: "ambiguous", reason: "ambiguous" });
  });

  it("rejects failed decodes, lost clips and low teacher confidence, with the reason", () => {
    const failed = { ...locked, ...{ proposed: null, state: null, exclusion: "decode_failed" as const, candidates: [] } };
    expect(bucketClip(index, failed, null, cutoffs)).toMatchObject({ bucket: "reject", reason: "decode_failed" });
    expect(bucketClip(index, { ...locked, state: "lost", exclusion: "lost" }, confident, cutoffs)).toMatchObject({ bucket: "reject", reason: "lost" });
    expect(bucketClip(index, locked, { ...confident, meanConf: 0.3 }, cutoffs)).toMatchObject({ bucket: "reject", reason: "low_teacher_confidence" });
  });
});

describe("buildManifests", () => {
  const bucketed = { bucket: "clean-match" as const, reason: "match" as const, label: { kind: "span" as const, span: ayah(100, 6) } };
  const clips = (n: number, clusterOf: (i: number) => string | null): ManifestClip[] =>
    Array.from({ length: n }, (_, i) => ({ shard: "s", row: i, path: `100_6_${1000000000 + i}.flac`, durationS: 3, cluster: clusterOf(i), ...bucketed }));
  const opts = { version: "tlog-v1", excludeStems: new Set<string>(), fractions: { val: 0.05, test: 0.05 } };

  it("never splits a speaker cluster across train, val and test", () => {
    const m = buildManifests(clips(4000, (i) => `c${i % 400}`), opts);
    const splitOf = new Map<string, Set<string>>();
    for (const [name, rows] of Object.entries(m.files)) for (const r of rows) (splitOf.get(r.cluster!) ?? splitOf.set(r.cluster!, new Set()).get(r.cluster!)!).add(name.split("/")[1]);
    expect([...splitOf.values()].every((s) => s.size === 1)).toBe(true);
    expect(m.manifest.counts["clean-match"].test.clips).toBeGreaterThan(0);
    expect(m.manifest.counts["clean-match"].val.clips).toBeGreaterThan(0);
    expect(m.manifest.counts["clean-match"].train.clips).toBeGreaterThan(3000);
  });

  it("fills test and val to their fractions with whole clusters, keeping oversized clusters in train", () => {
    // One cluster holds 30% of the clips; the rest are 700 clusters of 2.
    const m = buildManifests(clips(2000, (i) => (i < 600 ? "giant" : `c${i % 700}`)), { ...opts, maxClusterShare: 0.01 });
    const c = m.manifest.counts["clean-match"];
    expect(c.test.clips).toBeGreaterThanOrEqual(100);
    expect(c.test.clips).toBeLessThanOrEqual(102);
    expect(c.val.clips).toBeGreaterThanOrEqual(100);
    expect(c.val.clips).toBeLessThanOrEqual(102);
    expect(m.files["clean-match/train"].filter((r) => r.cluster === "giant")).toHaveLength(600);
  });

  it("removes benchmark IDs, by filename stem, and counts them", () => {
    const m = buildManifests(clips(10, () => "c"), { ...opts, excludeStems: new Set(["100_6_1000000003", "100_6_1000000007"]) });
    const paths = Object.values(m.files).flat().map((r) => r.path);
    expect(paths).toHaveLength(8);
    expect(paths).not.toContain("100_6_1000000003.flac");
    expect(m.manifest.excluded).toEqual({ benchmark: 2 });
  });

  it("files every clip under its bucket and split, and is deterministic", () => {
    const input = clips(500, (i) => (i % 3 ? `c${i % 50}` : null));
    const a = buildManifests(input, opts);
    expect(Object.values(a.files).flat()).toHaveLength(500);
    expect(Object.keys(a.files).every((k) => /^(clean-match|deviant|ambiguous|reject)\/(train|val|test)$/.test(k))).toBe(true);
    expect(JSON.stringify(buildManifests(input, opts))).toBe(JSON.stringify(a));
    expect(a.manifest.version).toBe("tlog-v1");
  });
});
