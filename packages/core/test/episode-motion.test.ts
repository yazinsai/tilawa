import { describe, expect, it } from "vitest";
import { episodeTrainingGraph, episodeTrainingLogP, episodeTransitionCost, fitEpisodeMotion, validateEpisodeModel } from "../src/episode-motion.js";
import type { EpisodeMotionModel } from "../src/episode-motion.js";
import { createLocalSearch } from "../src/local.js";
import type { ReferenceIndex } from "../src/reference.js";

const model: EpisodeMotionModel = {
  initial: [.9, .1], background: [.7, .2, .05, .05],
  joint: [[ [.63, .07], [.04, .16], [.01, .04], [.01, .04] ], [ [.14, .56], [.02, .18], [.005, .045], [.005, .045] ]],
};

describe("normalized episode model", () => {
  it("sums over every event boundary and recovery state, agreeing with exhaustive enumeration", () => {
    const graph = episodeTrainingGraph({ words: [{ phones: "a" }, { phones: "b", omit: true }], heard: "a" }, ["a", "b"]);
    const walk = (node: number, state: number): number => {
      if (node === graph.nodes - 1) return .9;
      return graph.arcs.filter(a => a.from === node).reduce((sum, arc) => sum + Math.exp(arc.logP)
        * [0, 1].reduce((v, next) => v + model.joint[state][arc.event][next] * walk(arc.to, next), 0), 0);
    };
    const p = model.initial.reduce((sum, prior, s) => sum + prior * walk(0, s), 0);
    expect(episodeTrainingLogP(model, graph)).toBeCloseTo(Math.log(p), 10);
  });

  it("retains exactly one scripted omission and repeat in every complete alignment", () => {
    const graph = episodeTrainingGraph({ words: [{ phones: "a", omit: true }, { phones: "b" }, { phones: "b", repeat: true }], heard: "bb" }, ["a", "b"]);
    const paths = (node: number, omitted: number, repeated: number): void => {
      if (node === graph.nodes - 1) {
        expect(omitted).toBe(1); expect(repeated).toBe(1); return;
      }
      for (const arc of graph.arcs.filter(a => a.from === node)) paths(arc.to, omitted + Number(arc.event === 2), repeated + Number(arc.event === 3));
    };
    paths(0, 0, 0);
  });

  it("preserves joint normalization in the likelihood-ratio contribution", () => {
    validateEpisodeModel(model);
    for (let s = 0; s < 2; s++) {
      const events = ["read", "edit", "omit", "repeat"] as const;
      const sum = events.reduce((a, event, e) => a + [0, 1].reduce((b, t) => b + model.background[e]
        * Math.exp(-3 * episodeTransitionCost(model, s, event, t) / 10), 0), 0);
      expect(sum).toBeCloseTo(1, 12);
    }
    expect(() => validateEpisodeModel({ ...model, background: [1, 1, 1, 1] })).toThrow(/normalized/);
  });

  it("fits positive distributions with latent boundaries and includes an independent control", () => {
    const records = [
      { words: [{ phones: "ab" }, { phones: "cd", omit: true }, { phones: "ef" }], heard: "abef" },
      { words: [{ phones: "ab" }, { phones: "cd" }, { phones: "ef" }], heard: "abcdef" },
      { words: [{ phones: "ab" }, { phones: "ab", repeat: true }, { phones: "ef" }], heard: "ababef" },
    ];
    const graphs = records.map(r => episodeTrainingGraph(r, [..."abcdef"]));
    for (const dependent of [true, false]) {
      const fitted = fitEpisodeMotion(graphs, 1, dependent);
      expect(() => validateEpisodeModel(fitted)).not.toThrow();
      expect(graphs.every(g => Number.isFinite(episodeTrainingLogP(fitted, g)))).toBe(true);
      expect(fitted.dependent).toBe(dependent);
    }
  });
});

const simpleIndex = (): ReferenceIndex => {
  const phonemes = "abcdefghijklmnopqr";
  const words = Array.from({ length: 6 }, (_, i) => ({ surah: 1, ayah: 1, word: i + 1, start: i * 3, end: i * 3 + 3 }));
  return { version: "synthetic", inventoryId: "synthetic", tokens: new Set(phonemes), phonemes, words,
    grams: new Map(), gramSize: 4, wordAt: offset => Math.min(words.length - 1, Math.max(0, Math.floor(offset / 3))) };
};

describe("local episode candidates", () => {
  const start = { offset: 3, origin: 0, edits: 0 };
  it("keeps the independent control and absent model on the existing path", () => {
    const index = simpleIndex();
    const baseline = createLocalSearch(index, start);
    const control = createLocalSearch(index, start, { ...model, dependent: false });
    for (const p of "deghijkl") {
      baseline.step(p); control.step(p);
      expect(control.position()).toEqual(baseline.position());
    }
  });

  it("preserves unambiguous uninterrupted progress and reports only the acoustic costs", () => {
    const search = createLocalSearch(simpleIndex(), start, model);
    for (const [i, p] of [..."defghijklmno"].entries()) {
      search.step(p);
      expect(search.position().offset).toBe(4 + i);
      expect(search.position().recentCost).toBe(0);
      expect(search.position().edits).toBe(0);
    }
  });

  it("rewinds both latent states without modifying a saved snapshot", () => {
    const search = createLocalSearch(simpleIndex(), start, model);
    for (const p of "deg") search.step(p);
    const saved = search.save();
    const first = [..."hijk"].map(p => { search.step(p); return search.position(); });
    search.restore(saved);
    const replay = [..."hijk"].map(p => { search.step(p); return search.position(); });
    expect(replay).toEqual(first);
    search.restore(saved);
    expect(search.save()).toEqual(saved);
  });

  it("retains both finite latent explanations after a mismatch", () => {
    const search = createLocalSearch(simpleIndex(), start, model);
    search.step("e");
    const saved = search.save() as { lo: number; cost: Float64Array };
    const offset = search.position().offset;
    const cell = (offset - saved.lo) * 2;
    expect(Number.isFinite(saved.cost[cell])).toBe(true);
    expect(Number.isFinite(saved.cost[cell + 1])).toBe(true);
  });
});
