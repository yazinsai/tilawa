import { expect, it } from "vitest";
import { createLocalSearch } from "../src/local.js";
import type { LocalWordMetricOptions } from "../src/local.js";
import { metricFixture } from "./word-metric-fixture.js";

type Comparison = Parameters<NonNullable<LocalWordMetricOptions["audit"]>>[0];
const start = { offset: 0, origin: 0, edits: 0 };

it("audits the complete common interval without changing original positions or edit costs", () => {
  const { index, model } = metricFixture();
  let n = 0;
  const comparisons: Comparison[] = [];
  const original = createLocalSearch(index, start);
  const audit = createLocalSearch(index, start, { model, mode: "audit", audit(c) {
    expect(c.candidates.every(p => p.score.observations === n)).toBe(true);
    comparisons.push(c);
  } });
  for (const p of "بتدبتدفنق") {
    n++;
    original.step(p);
    audit.step(p);
    expect(audit.position()).toEqual(original.position());
  }
  expect(comparisons).toHaveLength(1);
  const restart = comparisons[0].candidates.find(p => p.offset === comparisons[0].original)!;
  expect(restart.origin).toBe(9);
  expect(restart.score.observations).toBe(7);
  expect(restart.score.fixed).toBeCloseTo(20.006, 12);
  expect(restart.score.raw).toBeCloseTo(20.006, 12);
});

it("rewinds trajectory evidence and exports independent score snapshots", () => {
  const { index, model } = metricFixture();
  const comparisons: Comparison[] = [];
  const local = createLocalSearch(index, start, { model, mode: "audit", audit: c => { comparisons.push(c); } });
  for (const p of "بتدبتد") local.step(p);
  const saved = local.save();
  local.step("ف");
  const position = local.position(), comparison = structuredClone(comparisons[0]);
  comparisons[0].candidates[0].score.words[0].written = "changed";
  local.step("ن");
  local.restore(saved);
  local.step("ف");
  expect(local.position()).toEqual(position);
  expect(comparisons[1]).toEqual(comparison);
});

it("applies a frozen metric by choosing an existing complete trajectory", () => {
  const { index, model } = metricFixture();
  let last: Comparison | undefined;
  const local = createLocalSearch(index, start, { model, audit: c => { last = c; } });
  model.scale = 1e6;
  for (const p of "بتدبتدف") local.step(p);
  expect(last).toBeDefined();
  const selected = last!.candidates.find(p => p.offset === last!.selected)!;
  expect(local.position().offset).toBe(selected.offset);
  expect(local.position().origin).toBe(selected.origin);
  expect(selected.score.observations).toBe(7);
});
