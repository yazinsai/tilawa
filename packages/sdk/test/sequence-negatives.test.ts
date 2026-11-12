import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "@tilawa/core";
import type { CheckerEvent, ReferenceAsset } from "@tilawa/core";
import { hasTranscript, mineSequenceNegatives, trainingTakes, trainingFeatures } from "../src/training/sequence-negatives.js";
import { createListener } from "../src/listener.js";
import type { LoadedModel } from "../src/model.js";
import type { TrainingTake } from "../src/training/sequence-negatives.js";

const take = (id: string, ayahs = ["1:1"]): TrainingTake =>
  ({ id, speaker: id, split: "train", kind: "none", ayahs, wav: `${id}.wav` });
const index = loadReferenceIndex({ version: "test", text: "test", profile: {}, inventory: { id: "test", tokens: "بتا" }, ayahs: [
  { s: 1, a: 1, w: ["one"], o: [0], p: "با", iso: ["ب"] },
  { s: 1, a: 2, w: ["two"], o: [0], p: "تا", iso: ["تا"] },
] } satisfies ReferenceAsset);
const warning = (observed: string): CheckerEvent => ({ type: "checker", v: 1, id: "x", timeMs: 80,
  state: "confirmed", errorClass: "wrong_word", location: { surah: 1, ayah: 1, word: 1 }, observed, confidence: 1 });
const tokens = [{ p: "ت", start: 1, end: 1, conf: 1 }, { p: "ا", start: 2, end: 3, conf: 1 }];

describe("training sequence negatives", () => {
  it("excludes whole speakers across splits and never trusts mistake scripts", () => {
    expect(trainingTakes([take("a"), { ...take("b"), speaker: "a", split: "eval" }, take("c")])
      .map(t => t.id)).toEqual(["c"]);
    expect(hasTranscript({ ...take("a"), events: [warning("تا")] })).toBe(false);
    expect(hasTranscript({ ...take("a"), non_errors: [{}] })).toBe(false);
    expect(hasTranscript({ ...take("a"), kind: "wrong_word" })).toBe(false);
    expect(() => trainingTakes([take("a"), take("a")])).toThrow("Duplicate");
  });
  it("keeps the whole mistaken sequence and requires independently recorded positives", () => {
    const found = mineSequenceNegatives(index, take("a"), [warning("تا")], tokens, [take("b", ["1:2"])]);
    expect(found).toEqual([{ recording: "a", speaker: "a", word: 0, from: 1, to: 4,
      written: "با", heard: "تا", positives: ["b"] }]);
    expect(mineSequenceNegatives(index, take("a"), [warning("تا")], tokens, [])).toEqual([]);
    expect(mineSequenceNegatives(index, take("a"), [warning("تا"), warning("تا")], tokens,
      [take("b", ["1:2"])])).toHaveLength(1);
    expect(mineSequenceNegatives(index, take("a"), [warning("تا")], tokens,
      [{ ...take("b", ["1:2"]), split: "eval" }])).toEqual([]);
    expect(mineSequenceNegatives(index, take("a"), [warning("ت")], tokens,
      [take("b", ["1:2"])])).toEqual([]); // positive CTC transcript is تا, not the partial ت
  });
  it("excludes lawful stop forms, unknown symbols and ambiguous token witnesses", () => {
    expect(mineSequenceNegatives(index, take("a"), [warning("ب")], tokens, [take("b")])).toEqual([]);
    expect(mineSequenceNegatives(index, take("a"), [warning("ج")], tokens, [take("b")])).toEqual([]);
    expect(mineSequenceNegatives(index, take("a"), [warning("تا")], [...tokens, ...tokens],
      [take("b", ["1:2"])])).toEqual([]);
  });
  it("matches actual listener inputs, including real pauses and the resampler tail", async () => {
    for (const rate of [16000, 48000]) {
      const samples = Float32Array.from({ length: rate }, (_, i) =>
        i < rate / 2 ? .1 * Math.sin(i * .17) : 0);
      const calls: Float32Array[] = [];
      const model: LoadedModel = {
        manifest: { schema: 1, id: "synthetic", format: "synthetic", arch: "zipformer2-streaming-ctc",
          inventory: "synthetic", tokens: "ب", blank: 0, sampleRate: 16000, frameMs: 40,
          fbank: { mels: 80, padValue: -23. }, window: 45, shift: 32, states: [],
          file: "synthetic", bytes: 0, sha256: "synthetic" }, supported: [],
        backend: { name: "capture", async release() {}, async run(feeds) {
          calls.push(Float32Array.from(feeds.x.data as Float32Array));
          return { log_probs: { type: "float32", dims: [1, 8, 2], data: new Float32Array(16).fill(-Math.log(2)) } };
        } },
      };
      const listener = createListener(model);
      const step = Math.round(rate * .32);
      for (let at = 0; at < samples.length; at += step)
        await listener.push(samples.subarray(at, at + step), rate);
      await listener.end();
      const actual = trainingFeatures(samples, rate);
      expect(actual.length).toBe(listener.frames);
      calls.forEach((window, n) => {
        for (let row = 0; row < 45; row++) expect(window.subarray(row * 80, (row + 1) * 80))
          .toEqual(actual[n * 32 + row] ?? new Float32Array(80).fill(-23.));
      });
      expect(actual.length).toBeGreaterThan(90);
      expect([...actual.at(-1)!]).toEqual(Array(80).fill(Math.fround(Math.log(1.1920928955078125e-7))));
    }
  });
});
