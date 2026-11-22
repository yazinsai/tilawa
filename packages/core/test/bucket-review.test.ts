import { describe, expect, it } from "vitest";
import type { BucketedClip, BucketCutoffs } from "../src/buckets/index.js";
import { CUTOFF_STRATA, drawCutoffSample, scoreCutoffs } from "../src/buckets/review.js";
import type { CutoffAnswer, CutoffClip } from "../src/buckets/review.js";

const cutoffs: BucketCutoffs = { minConfidence: 0.2, cleanMaxPer: 0.03, cleanMinTeacherConf: 0.9, deviantMaxPer: 0.3, rejectBelowTeacherConf: 0.5 };
const span = { kind: "span" as const, span: { start: { surah: 1, ayah: 1, word: 1 }, end: { surah: 1, ayah: 1, word: 4 } } };

let row = 0;
const clip = (b: Pick<BucketedClip, "bucket" | "reason">, per: number | null, meanConf: number | null, label: BucketedClip["label"] = b.bucket === "clean-match" || b.bucket === "deviant" ? span : null): CutoffClip => ({
  shard: "s",
  row: row++,
  path: `1_1_${row}.flac`,
  ...b,
  label,
  per,
  meanConf,
  confidence: 0.9,
});
const clean = (per = 0, meanConf = 0.97) => clip({ bucket: "clean-match", reason: "match" }, per, meanConf);
const deviant = (per = 0.12, meanConf = 0.95) => clip({ bucket: "deviant", reason: "mismatch" }, per, meanConf);

describe("drawCutoffSample", () => {
  const population = [
    ...Array.from({ length: 200 }, () => clean()),
    ...Array.from({ length: 50 }, () => clean(0.025)),
    ...Array.from({ length: 100 }, () => deviant()),
    ...Array.from({ length: 30 }, () => clip({ bucket: "reject", reason: "decode_failed" }, null, null)),
  ];

  it("fills each stratum's quota at most, records populations, and never samples undecoded clips", () => {
    const s = drawCutoffSample(population, cutoffs, { seed: 1, quota: { clean_core: 10, clean_edge: 5, deviant_core: 7 } });
    expect(s.strata.clean_core).toEqual({ population: 200, sampled: 10 });
    expect(s.strata.clean_edge).toEqual({ population: 50, sampled: 5 });
    expect(s.strata.deviant_core).toEqual({ population: 100, sampled: 7 });
    expect(s.unsampled).toEqual({ decode_failed: 30 });
    expect(s.clips).toHaveLength(22);
    expect(s.clips.every((c) => c.reason !== "decode_failed")).toBe(true);
    expect(Object.keys(s.strata).every((k) => (CUTOFF_STRATA as readonly string[]).includes(k))).toBe(true);
  });

  it("is deterministic for a seed", () => {
    const a = drawCutoffSample(population, cutoffs, { seed: 3, quota: { clean_core: 10 } });
    expect(drawCutoffSample(population, cutoffs, { seed: 3, quota: { clean_core: 10 } })).toEqual(a);
  });
});

describe("scoreCutoffs", () => {
  it("weights answers back to stratum populations and re-buckets clips under each candidate cutoff", () => {
    // 100 exact clips at PER 0 (2 sampled, both judged exact); 100 at PER 0.025 (2 sampled, one has a mistake).
    const population = [...Array.from({ length: 100 }, () => clean(0)), ...Array.from({ length: 100 }, () => clean(0.025))];
    const sample = drawCutoffSample(population, cutoffs, { seed: 1, quota: { clean_core: 2, clean_edge: 2 } });
    const answers: CutoffAnswer[] = sample.clips.map((c, i) => ({ shard: c.shard, row: c.row, verdict: c.per === 0 || i % 2 ? "exact" : "mistakes" }));
    const edgeWrong = sample.clips.filter((c) => c.per !== 0 && answers.find((a) => a.row === c.row)!.verdict === "mistakes").length;
    expect(edgeWrong).toBe(1);

    const s = scoreCutoffs(sample, answers, { cleanMaxPer: [0.01, 0.03], cleanMinTeacherConf: [0.9] });
    const strict = s.grid.find((g) => g.cleanMaxPer === 0.01)!;
    const loose = s.grid.find((g) => g.cleanMaxPer === 0.03)!;
    expect(strict.clean.precision.rate).toBe(1);
    expect(strict.clean.share).toBeCloseTo(0.5);
    expect(loose.clean.precision.rate).toBeCloseTo(0.75);
    expect(loose.clean.share).toBeCloseTo(1);
  });

  it("leaves unsure answers out", () => {
    const population = Array.from({ length: 10 }, () => clean());
    const sample = drawCutoffSample(population, cutoffs, { seed: 1, quota: { clean_core: 2 } });
    const answers: CutoffAnswer[] = [
      { shard: "s", row: sample.clips[0].row, verdict: "exact" },
      { shard: "s", row: sample.clips[1].row, verdict: "unsure" },
    ];
    const g = scoreCutoffs(sample, answers, { cleanMaxPer: [0.03], cleanMinTeacherConf: [0.9] }).grid[0];
    expect(g.clean.precision).toMatchObject({ k: 1, n: 1, rate: 1 });
  });
});
