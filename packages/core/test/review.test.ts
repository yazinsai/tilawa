import { describe, expect, it } from "vitest";
import { drawSample, score, stratumOf, wilson } from "../src/review/index.js";
import type { ReviewAnswer } from "../src/review/index.js";
import type { ProposedLabel } from "../src/relabel/index.js";
import type { WordSpan } from "../src/events.js";

const span = (a: number, b = a, sw = 1, ew = 4): WordSpan => ({ start: { surah: 100, ayah: a, word: sw }, end: { surah: 100, ayah: b, word: ew } });
const lastWord = () => 4;

let row = 0;
const label = (o: Partial<ProposedLabel>): ProposedLabel => ({
  labelKind: "proposed",
  shard: "s",
  row: row++,
  path: `${row}.flac`,
  filenameLabel: { surah: 100, ayah: 1 },
  proposed: null,
  state: "locked",
  confidence: 0.9,
  fit: null,
  separation: null,
  leadEdits: null,
  cost: null,
  per: null,
  candidates: [],
  exclusion: null,
  pattern: null,
  agrees: null,
  ...o,
});
const agree = () => label({ proposed: span(1), pattern: "filename ayah", agrees: true });
const answer = (l: ProposedLabel, s: WordSpan | null, verdict: ReviewAnswer["verdict"] = s ? "span" : "not_recitation"): ReviewAnswer => ({
  shard: l.shard,
  row: l.row,
  verdict,
  span: s,
  ms: 1000,
});

describe("stratumOf", () => {
  it("puts boundary confidence first, then posterior disagreement, then the outcome", () => {
    expect(stratumOf(label({ confidence: 0.1, pattern: "filename ayah" }))).toBe("near_threshold");
    const l = agree();
    expect(stratumOf(l, { ...l, proposed: span(2) })).toBe("posteriors_disagree");
    expect(stratumOf(l, l)).toBe("agrees");
    expect(stratumOf(label({ pattern: "off by -1 ayah" }))).toBe("offset");
    expect(stratumOf(label({ pattern: "contains 2 ayahs (+0..+1)" }))).toBe("multi_ayah");
    expect(stratumOf(label({ state: "lost", confidence: null }))).toBe("lost");
    expect(stratumOf(label({ state: "acquiring", confidence: 0.5 }))).toBe("insufficient");
    expect(stratumOf(label({ state: null }))).toBeNull();
  });
});

describe("drawSample", () => {
  const labels = [...Array.from({ length: 100 }, agree), ...Array.from({ length: 5 }, () => label({ state: "lost", confidence: null }))];

  it("takes at most the quota per stratum, a small stratum whole, and records populations", () => {
    const s = drawSample(labels, { quota: { agrees: 10 } });
    expect(s.strata.agrees).toEqual({ population: 100, sampled: 10 });
    expect(s.strata.lost).toEqual({ population: 5, sampled: 5 });
    expect(s.clips).toHaveLength(15);
  });

  it("is the same for the same seed and differs for another", () => {
    const rows = (seed: number) => drawSample(labels, { seed, quota: { agrees: 10 } }).clips.map((c) => c.row);
    expect(rows(1)).toEqual(rows(1));
    expect(rows(2)).not.toEqual(rows(1));
  });
});

describe("wilson", () => {
  it("brackets the rate and widens for small samples", () => {
    const [lo, hi] = wilson(9, 10);
    expect(lo).toBeLessThan(0.9);
    expect(hi).toBeGreaterThan(0.9);
    const [lo2, hi2] = wilson(90, 100);
    expect(hi2 - lo2).toBeLessThan(hi - lo);
  });
});

describe("score", () => {
  it("counts exact-span precision, false locks, exclusions, misses and wrong filenames", () => {
    const right = agree();
    const wrongWords = agree();
    const offset = label({ proposed: span(2), pattern: "off by +1 ayah" });
    const noise = agree();
    const lost = label({ state: "lost", confidence: null });
    const unsure = agree();
    const labels = [right, wrongWords, offset, noise, lost, unsure];
    const sample = drawSample(labels);
    const answers = [
      answer(right, span(1)),
      answer(wrongWords, span(1, 1, 2, 4)),
      answer(offset, span(2)),
      answer(noise, null),
      answer(lost, span(3)),
      answer(unsure, null, "unsure"),
    ];
    const s = score("t", sample, labels, answers, lastWord);
    const agrees = s.strata.find((x) => x.stratum === "agrees")!;
    expect(agrees.reviewed).toBe(3);
    expect(agrees.precision).toMatchObject({ k: 1, n: 3 });
    expect(agrees.ayahPrecision).toMatchObject({ k: 2, n: 3 });
    expect(agrees.falseLock).toMatchObject({ k: 2, n: 3 });
    expect(agrees.filenameWrong).toMatchObject({ k: 2, n: 3 });
    const off = s.strata.find((x) => x.stratum === "offset")!;
    expect(off.precision).toMatchObject({ k: 1, n: 1 });
    expect(off.filenameWrong).toMatchObject({ k: 1, n: 1 });
    const lostScore = s.strata.find((x) => x.stratum === "lost")!;
    expect(lostScore.exclusion).toMatchObject({ k: 1, n: 1 });
    expect(lostScore.missed).toMatchObject({ k: 1, n: 1 });
  });

  it("weights strata by their share of the run, not of the sample", () => {
    // 90 agreeing clips, all right; 10 offset clips, all wrong. Review 5 of each.
    const good = Array.from({ length: 90 }, agree);
    const bad = Array.from({ length: 10 }, () => label({ proposed: span(2), pattern: "off by +1 ayah" }));
    const labels = [...good, ...bad];
    const sample = drawSample(labels, { quota: { agrees: 5, offset: 5 } });
    const byRow = new Map(labels.map((l) => [l.row, l]));
    const answers = sample.clips.map((c) => answer(byRow.get(c.row)!, c.stratum === "agrees" ? span(1) : span(3)));
    const s = score("t", sample, labels, answers, lastWord);
    expect(s.weighted.precision.rate).toBeCloseTo(0.9);
    expect(s.sweep[0].coverage).toBeCloseTo(1);
  });

  it("keeps fewer, surer locks as the minimum confidence rises", () => {
    const sure = Array.from({ length: 5 }, () => label({ proposed: span(1), pattern: "filename ayah", confidence: 0.9 }));
    const shaky = Array.from({ length: 5 }, () => label({ proposed: span(2), pattern: "filename ayah", confidence: 0.25 }));
    const labels = [...sure, ...shaky];
    const sample = drawSample(labels);
    const answers = labels.map((l) => answer(l, span(1)));
    const s = score("t", sample, labels, answers, lastWord);
    const at = (t: number) => s.sweep.find((x) => x.minConfidence === t)!;
    expect(at(0.2).precision.rate).toBeCloseTo(0.5);
    expect(at(0.3).precision.rate).toBe(1);
    expect(at(0.3).coverage).toBeCloseTo(0.5);
  });
});
