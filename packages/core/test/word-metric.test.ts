import { expect, it } from "vitest";
import { createWordMetric, phoneEmbedding, wordMetricEdit, wordMetricPathScore } from "../src/word-metric.js";
import type { WordMetricPath, WordMetricStep } from "../src/word-metric.js";
import { metricFixture } from "./word-metric-fixture.js";

function path(steps: WordMetricStep[]): WordMetricPath | null {
  let p: WordMetricPath | null = null;
  for (const step of steps) p = { previous: p, step };
  return p;
}

it("composes unlisted phone strings, keeps order evidence and normalizes embeddings", () => {
  const { model } = metricFixture();
  const metric = createWordMetric(model, "unit");
  const v = phoneEmbedding(model.written, model.tokens, "فنبتد")!;
  expect(v.reduce((n, x) => n + x * x, 0)).toBeCloseTo(1, 12);
  expect(metric.cost("فنبتد", "فنبتد")).toBe(0);
  expect(metric.cost("فنبتد", "دتبنف")!).toBeGreaterThan(0);
  expect(metric.cost("فنبتد", "x")).toBeNull();
  expect(metric.cost("", "فنبتد")).toBeNull();
});

it("owns the model and rejects incompatible, malformed or nonfinite encoders", () => {
  const { model } = metricFixture();
  const metric = createWordMetric(model, "unit");
  const expected = metric.cost("بتد", "فنق");
  model.scale = 100;
  model.written.convolution[0][0].fill(500);
  expect(metric.cost("بتد", "فنق")).toBe(expected);
  expect(() => createWordMetric(model, "other")).toThrow();
  model.scale = NaN;
  expect(() => createWordMetric(model, "unit")).toThrow();
  const broken = metricFixture().model;
  broken.observed.projection[0].pop();
  expect(() => createWordMetric(broken, "unit")).toThrow();
});

it("retains duration, movement and whole-word deletion charges without forgiving unmatched suffixes", () => {
  const steps: WordMetricStep[] = [
    { cost: 10, word: 0, reference: "ن", observed: "ب" },
    { cost: 2, word: 0, reference: "ن", duration: true },
    { cost: 2, word: 0, reference: "ن", duration: true },
    { cost: 20.015, move: true },
    { cost: 0, word: 2, reference: "ف", observed: "ف" },
    { cost: 10, word: 3, reference: "ق" },
  ];
  const result = wordMetricPathScore(path(steps), { cost: (r, o) => r && o ? 1 : null });
  expect(result.raw).toBeCloseTo(44.015, 12);
  expect(result.fixed).toBeCloseTo(24.015, 12);
  expect(result.identity).toBe(20);
  expect(result.learned).toBeCloseTo(36.015, 12);
  expect(result.observations).toBe(2);
  expect(result.words[0].written).toBe("ن");
  expect(result.words[1].written).toBe("ف");
  expect(result.words[2].learned).toBe(10);
});

it("can reverse word-identity ordering while keeping the same observation count and movement price", () => {
  const score = (r: string, raw: number) => wordMetricPathScore(path([
    { cost: 20.01, move: true }, { cost: raw, word: 1, reference: r, observed: "بتد" },
  ]), { cost: r => r === "بتد" ? 1 : 5 });
  const near = score("بتد", 4), rhyme = score("فنق", 3);
  expect(near.raw).toBeGreaterThan(rhyme.raw);
  expect(rhyme.learned).toBeGreaterThan(near.learned);
  expect(near.observations).toBe(rhyme.observations);
  expect(near.fixed).toBe(rhyme.fixed);
});

it("uses directed existing edit prices and splits repeated occurrences at a movement edge", () => {
  expect(wordMetricEdit("نننن", 1, null, "del", 0).duration).toBe(true);
  expect(wordMetricEdit("ن", 0, "ب", "sub", 0).duration).toBe(false);
  expect(wordMetricEdit("َ", 0, null, "del", 0).cost).toBe(5);
  const result = wordMetricPathScore(path([
    { cost: 0, word: 0, reference: "ب", observed: "ب" },
    { cost: 20, move: true },
    { cost: 0, word: 0, reference: "ب", observed: "ب" },
  ]), { cost: () => 0 });
  expect(result.words).toHaveLength(2);
  expect(result.fixed).toBe(20);
  expect(result.observations).toBe(2);
});

it("extends an audited open word without charging its earlier partial fit twice", () => {
  const metric = { cost: (r: string) => r.length };
  const first: WordMetricPath = { previous: null, step: { cost: 0, word: 0, reference: "ب", observed: "ب" } };
  const prefix = wordMetricPathScore(first, metric);
  prefix.words[0].written = "caller edit";
  const second: WordMetricPath = { previous: first, step: { cost: 0, word: 0, reference: "ت", observed: "ت" } };
  const full = wordMetricPathScore(second, metric);
  expect(full.words).toHaveLength(1);
  expect(full.words[0].written).toBe("بت");
  expect(full.learned).toBe(2);
  expect(full.observations).toBe(2);
});
