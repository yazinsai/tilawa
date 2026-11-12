import { describe, expect, it } from "vitest";
import { emphasisLogLikelihood, emphasisObservation, emphasisTransition, emphasisUpdate, fitEmphasisModel, isEmphatic, posteriorObservation } from "../src/emphasis-channel.js";
import type { EmphasisModel } from "../src/emphasis-channel.js";
import { createLocalSearch } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";
import { trackerInput } from "../src/tracker.js";

const phones = [..."طبتنزر"];
const model = (kind: EmphasisModel["kind"] = "persistent"): EmphasisModel => ({
  inventory: "test", phones, kind, persistence: 0.75, mass: 8,
  background: phones.map(() => 1 / phones.length),
  emission: phones.map(() => [phones.map(() => 1 / phones.length), phones.map(() => 1 / phones.length)]),
});
const indexFor = (words: string[]) => loadReferenceIndex({ version: "test", text: "test", profile: {}, inventory: { id: "test", tokens: phones.join("") },
  ayahs: [{ s: 1, a: 1, w: words, p: words.join(""), o: words.map((_, i) => words.slice(0, i).join("").length), iso: words }] });
const state = (local: ReturnType<typeof createLocalSearch>) => {
  const saved = local.save() as { coloration: Float64Array; best: number; lo: number };
  return saved.coloration[saved.best - saved.lo];
};

describe("emphasis domain", () => {
  it("activates only on an emphatic articulation and carries its belief across later phones", () => {
    expect([..."صضطظ"].every(isEmphatic)).toBe(true);
    expect(isEmphatic("ت")).toBe(false);
    const m = model(), q = m.background;
    const first = emphasisUpdate(m, 0, "ط", q);
    expect(first.colored).toBe(1);
    const second = emphasisUpdate(m, first.colored, "ب", q);
    expect(second.colored).toBe(0.75);
    expect(emphasisUpdate(m, second.colored, "ت", q).colored).toBe(0.5625);
    expect(emphasisUpdate(m, 0, "ب", q).colored).toBe(0);
  });

  it("removes history dependence in the identical-capacity independent control", () => {
    const m = model("independent");
    expect(m.emission).toEqual(model().emission);
    expect(emphasisUpdate(m, 0, "ب", m.background)).toEqual(emphasisUpdate(m, 1, "ب", m.background));
    for (const p of phones) for (const row of emphasisTransition(m, p)) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
  });

  it("conditions the same observation on both reference identity and the shared state", () => {
    const m = model();
    const emission = m.emission.map(states => states.map(row => [...row]));
    emission[1][0] = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2];
    emission[1][1] = [0.1, 0.6, 0.1, 0.1, 0.05, 0.05];
    const colored = { ...m, emission }, q = [0, 1, 0, 0, 0, 0];
    expect(emphasisUpdate(colored, 1, "ب", q).cost).toBeLessThan(emphasisUpdate(colored, 0, "ب", q).cost);
    expect(emphasisUpdate(colored, 1, "ب", q).cost).toBeLessThan(emphasisUpdate(colored, 1, "ت", q).cost);
    expect(emphasisObservation(colored)(1, "ب").cost).toBe(0);
    expect(emphasisObservation(colored, [NaN])(1, "ب").cost).toBe(0);
  });

  it("sums every placement and coloration path once", () => {
    const m = model(), record = { written: "طب", observed: [m.background, m.background] };
    const visit = (i: number, j: number, s: number): number => {
      if (i === record.written.length) return j === record.observed.length ? 0.9 : 0.1 / phones.length * visit(i, j + 1, s);
      let sum = 0.1 * visit(i + 1, j, s);
      if (j < record.observed.length) {
        sum += 0.1 / phones.length * visit(i, j + 1, s);
        const t = emphasisTransition(m, record.written[i]);
        for (let next = 0; next < 2; next++) sum += 0.8 * t[s][next] / phones.length * visit(i + 1, j + 1, next);
      }
      return sum;
    };
    expect(Math.exp(emphasisLogLikelihood(m, record))).toBeCloseTo(visit(0, 0, 0), 12);
  });

  it("fits positive normalized emissions while marginalizing ambiguous placement", () => {
    const records = [{ written: "طب", observed: [[0.5, 0.5, 0, 0, 0, 0]] }, { written: "بت", observed: [[0, 0.8, 0.2, 0, 0, 0], [0, 0.2, 0.8, 0, 0, 0]] }];
    for (const kind of ["persistent", "independent"] as const) {
      const fitted = fitEmphasisModel(records, phones, kind, 0.75, 8);
      for (const states of fitted.emission) for (const row of states) {
        expect(row.every(x => x > 0)).toBe(true);
        expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
      }
      expect(Number.isFinite(emphasisLogLikelihood(fitted, records[0]))).toBe(true);
      expect(fitted).toEqual(fitEmphasisModel(records, phones, kind, 0.75, 8));
    }
  });

  it("preserves posterior uncertainty and original absolute frame coordinates", () => {
    const rows = { k: 2, startFrame: 8, nFrames: 2, ids: [0, 1, 0, 2], probs: [0.4, 0.4, 0.2, 0.6] };
    const q = posteriorObservation(rows, ["ط", "ب"], { start: 8, end: 9 })!;
    expect(q[0]).toBeCloseTo(0.6 / 1.4);
    expect(q[1]).toBeCloseTo(0.8 / 1.4);
    expect(posteriorObservation(rows, ["ط", "ب"], { start: 7, end: 8 })).toBeUndefined();
  });

  it("retains observations through trackerInput without changing listener tokens or acquisition posteriors", () => {
    const input = { inventory: "test", frameMs: 40, phonemes: [{ p: "ب", start: 4, end: 4, conf: 0.6 }],
      posteriors: { startFrame: 4, nFrames: 1, k: 3, ids: [0, 1, 2], probs: [0.1, 0.6, 0.3] } };
    const before = JSON.stringify(input), transformed = trackerInput(input, model());
    expect(transformed.posteriors).toBeUndefined();
    expect(transformed.phonemes[0].phonePosterior?.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(transformed.phonemes.map(({ phonePosterior: _q, ...ph }) => ph)).toEqual(input.phonemes);
    expect(JSON.stringify(input)).toBe(before);
    expect(trackerInput({ ...input, inventory: "other" }, model()).phonemes).toBe(input.phonemes);
  });
});

describe("local emphasis state", () => {
  it("survives a word boundary and restores the path-specific belief on rewind", () => {
    const local = createLocalSearch(indexFor(["ط", "بت"]), { offset: 0, origin: 0, edits: 0 }, model());
    local.step("ط");
    expect(state(local)).toBe(1);
    local.step("ب");
    expect(state(local)).toBe(0.75);
    const saved = local.save(), position = local.position();
    local.step("ت");
    expect(state(local)).toBe(0.5625);
    local.restore(saved);
    expect(local.position()).toEqual(position);
    expect(state(local)).toBe(0.75);
  });

  it("does not activate coloration on a deleted reference emphatic", () => {
    const local = createLocalSearch(indexFor(["بطزر"]), { offset: 0, origin: 0, edits: 0 }, model());
    for (const p of "بزر") local.step(p);
    expect(local.position().offset).toBe(4);
    expect(state(local)).toBe(0);
  });

  it("keeps existing identity, duration and movement charges with a neutral acoustic model", () => {
    const index = indexFor(["بننن", "تز"]);
    const plain = createLocalSearch(index, { offset: 1, origin: 0, edits: 0 });
    const acoustic = createLocalSearch(index, { offset: 1, origin: 0, edits: 0 }, model());
    for (const p of "نتز") {
      plain.step(p); acoustic.step(p, model().background);
      expect(acoustic.position()).toEqual(plain.position());
    }
    expect(plain.position().recentCost * 3).toBeCloseTo(0.4);
  });
});
