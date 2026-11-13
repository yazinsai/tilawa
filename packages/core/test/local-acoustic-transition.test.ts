import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { createLocalSearch } from "../src/local.js";
import type { AcousticCandidate } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const start = index.words[100].start;
const text = index.phonemes.slice(start, start + 48);

it("leaves unambiguous progress unchanged and never calls the acoustic scorer", () => {
  let calls = 0;
  const initial = { offset: start, origin: start, edits: 0 };
  const baseline = createLocalSearch(index, initial);
  const acoustic = createLocalSearch(index, initial, { score: () => { calls++; return 0; } });
  for (const p of text) {
    baseline.step(p);
    acoustic.step(p);
    expect(acoustic.position()).toEqual(baseline.position());
  }
  expect(calls).toBe(0);
});

it("exports unchanged search candidates covering exactly the same observation suffix", () => {
  let inspected: AcousticCandidate[] = [];
  const local = createLocalSearch(index, { offset: start, origin: start, edits: 0 }, {
    training: true, inspect: (c, original, selected) => { inspected = c; expect(selected).toBe(original); },
  });
  for (const p of text.slice(0, 24)) local.step(p);
  expect(inspected.length).toBeGreaterThan(1);
  expect(new Set(inspected.map(c => c.observations))).toEqual(new Set([16]));
  expect(new Set(inspected.map(c => c.origin)).size).toBeGreaterThan(1);
});

it("rewinds candidate provenance with the DP when a heard phrase is cancelled", () => {
  const local = createLocalSearch(index, { offset: start, origin: start, edits: 0 }, { training: true });
  for (const p of text.slice(0, 10)) local.step(p);
  const snapshot = local.save();
  for (const p of "قزز") local.step(p);
  local.restore(snapshot);
  for (const p of text.slice(10, 20)) local.step(p);
  const fresh = createLocalSearch(index, { offset: start, origin: start, edits: 0 });
  for (const p of text.slice(0, 20)) fresh.step(p);
  expect(local.position()).toEqual(fresh.position());
});

it("keeps the original decision when any competing acoustic interval is unavailable", () => {
  const initial = { offset: start, origin: start, edits: 0 };
  const baseline = createLocalSearch(index, initial);
  let calls = 0;
  const compared = createLocalSearch(index, initial, { training: true, score: () => {
    calls++;
    return calls % 2 ? -1000 : null;
  } });
  for (const p of text.slice(0, 24)) {
    baseline.step(p); compared.step(p);
    expect(compared.position()).toEqual(baseline.position());
  }
  expect(calls).toBeGreaterThan(0);
});
