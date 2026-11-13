import { describe, expect, it } from "vitest";
import { createLocalSearch } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";
import { createTracker } from "../src/tracker.js";
import { trajectoryFeatures, TRAJECTORY_FEATURES, validateTrajectoryModel } from "../src/trajectory.js";
import type { TrajectoryFrame, TrajectoryModel } from "../src/trajectory.js";
import { trainingLabels, transcriptPath, wordBoundaries } from "../train/location-data.js";
import type { TrainingLabel } from "../train/location-data.js";

const index = loadReferenceIndex({ version: "test", text: "test", profile: {}, inventory: { id: "test", tokens: "ابتثجح" },
  ayahs: Array.from({ length: 7 }, (_, i) => ({ s: 1, a: i + 1, w: ["x", "y"], p: "ابتثجح", o: [0, 3], iso: ["ابت", "ثجح"] })) });
const start = { offset: 0, origin: 0, edits: 0 };
const zero: TrajectoryModel = { version: 1, features: TRAJECTORY_FEATURES, weights: TRAJECTORY_FEATURES.map(() => 0) };

describe("adversarial location correction", () => {
  it("zero potentials and graph observation preserve positions, costs, rewinds and recentering", () => {
    const baseline = createLocalSearch(index, start);
    const frames: TrajectoryFrame[] = [];
    const audited = createLocalSearch(index, start, { observe: f => frames.push(f), model: zero });
    let lastLo = -1, recentered = false;
    for (const phone of "ابتثجح".repeat(6) + "ثجحابت") {
      const a = baseline.save(), b = audited.save();
      baseline.step(phone); audited.step(phone);
      expect(audited.position()).toEqual(baseline.position());
      baseline.restore(a); audited.restore(b);
      baseline.step(phone); audited.step(phone);
      expect(audited.position()).toEqual(baseline.position());
      if (lastLo >= 0 && frames.at(-1)!.lo !== lastLo) recentered = true;
      lastLo = frames.at(-1)!.lo;
    }
    expect(recentered).toBe(true);
    expect(frames.every(f => f.arcs.every(arcs => arcs[0] && arcs.filter(Boolean).every(a => a!.x.length === 24)))).toBe(true);
  });

  it("learned restart potential changes trajectory selection while preserving the available graph", () => {
    const model = { ...zero, weights: zero.weights.map((_, i) => i === 2 ? -4 : 0) };
    const base = createLocalSearch(index, start), learned = createLocalSearch(index, start, { model });
    base.step("ث"); learned.step("ث");
    expect(learned.position().origin).toBe(3);
    expect(learned.position().origin).not.toBe(base.position().origin);
    const first = learned.save();
    learned.step("ج"); learned.restore(first); learned.step("ج");
    expect(learned.position().offset).toBeGreaterThan(0);
  });

  it("zero or explicit null models preserve the whole tracker event stream", () => {
    const a = createTracker(index, { minPhonemes: 3, minWords: 1 });
    const b = createTracker(index, { minPhonemes: 3, minWords: 1, locationModel: zero });
    const c = createTracker(index, { minPhonemes: 3, minWords: 1, locationModel: null });
    const tokens = "ابتثجح".repeat(4);
    for (let i = 0; i < tokens.length; i++) {
      const e = { inventory: "test", frameMs: 40, phonemes: [{ p: tokens[i], start: i, end: i, conf: 1 }], decodedTo: i + 1 };
      const events = a.push(e);
      expect(b.push(e)).toEqual(events); expect(c.push(e)).toEqual(events);
    }
    const events = a.end(); expect(b.end()).toEqual(events); expect(c.end()).toEqual(events);
  });

  it("features depend on phones and relative movement, not absolute passage coordinates", () => {
    const a = trajectoryFeatures(index, 2, 6, 10, "ث", 0);
    const b = trajectoryFeatures(index, 2, 12, 16, "ث", 0);
    expect(a).toEqual(b);
    expect(a).toHaveLength(TRAJECTORY_FEATURES.length);
  });

  it("rejects schema drift and non-finite weights", () => {
    expect(() => validateTrajectoryModel({ ...zero, weights: [NaN] })).toThrow();
    expect(() => validateTrajectoryModel({ ...zero, features: [...zero.features].reverse() })).toThrow();
  });

  it("keeps a session's potential snapshot when the caller mutates its weight array", () => {
    const weights = [...zero.weights];
    const local = createLocalSearch(index, start, { model: { ...zero, weights } });
    weights[2] = -4;
    const baseline = createLocalSearch(index, start);
    baseline.step("ث"); local.step("ث");
    expect(local.position()).toEqual(baseline.position());
  });
});

describe("training-only route supervision", () => {
  const label: TrainingLabel = { id: "a", speaker: "train-a", split: "train", kind: "none", ayahs: ["1:1"], wav: "wav/a.wav" };
  it("excludes held-out and mixed-split speakers, ambiguous labels and unscored takes", () => {
    const rows = [label, { ...label, id: "b", split: "eval" }, { ...label, id: "c", speaker: "train-b" },
      { ...label, id: "d", speaker: "train-c", events: [{ errorClass: "wrong_word", location: { surah: 1, ayah: 1, word: 1 } }] },
      { ...label, id: "e", speaker: "train-d", scored: false }];
    expect(trainingLabels(rows).map(r => r.id)).toEqual(["c"]);
  });
  it("retains repeat occurrences and removes only scripted omitted words", () => {
    const r = { ...label, kind: "scripted", ayahs: ["1:1", "1:2"],
      events: [{ errorClass: "omitted_word", location: { surah: 1, ayah: 1, word: 2 } }],
      non_errors: [{ kind: "repeat", location: { surah: 1, ayah: 2, word: 1 } }] };
    expect(transcriptPath(index, r)).toEqual([0, 2, 2, 3]);
    expect(wordBoundaries(index, [0, 1], "ابتثجح")).toEqual([0, 3, 6]);
  });
});
