import { describe, expect, it } from "vitest";
import { createSeamTeacher, teacherFeatures } from "../src/seam-teacher.js";
import type { InferenceBackend } from "../src/model.js";

const wave = Float32Array.from({ length: 16000 }, (_, i) => Math.sin(i * .1) * (.1 + i / 160000));
describe("fp32 seam teacher", () => {
  it("normalizes each mel bin with unbiased variance before stacking and masks the padded pair", () => {
    const unpaired = wave.subarray(0, 15840);
    const features = teacherFeatures(unpaired);
    const n = 1 + Math.floor((unpaired.length - 400) / 160);
    expect(features.frames).toBe(Math.ceil(n / 2));
    for (let m = 0; m < 80; m++) {
      const values = Array.from({ length: n }, (_, i) => features.data[i * 80 + m]);
      const mean = values.reduce((a, b) => a + b, 0) / n;
      const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1);
      expect(Math.abs(mean)).toBeLessThan(1e-5);
      expect(variance).toBeCloseTo(1, 3);
    }
    expect(features.mask.at(-1)).toBe(n % 2 ? 0n : 1n);
  });

  it("collapses a held phone while preserving a second occurrence after blank and its posterior rows", async () => {
    const rows = [[.9, .1, 0], [.1, .8, .1], [.1, .85, .05], [.9, .1, 0], [.1, .8, .1]];
    const backend: InferenceBackend = { name: "test", release: async () => {}, run: async feeds => {
      expect(feeds.input_features.type).toBe("float32");
      expect(feeds.attention_mask.type).toBe("int64");
      return { log_probs: { type: "float32", data: Float32Array.from(rows.flat(), Math.log), dims: [1, rows.length, 3] } };
    } };
    const evidence = await createSeamTeacher(backend, "بت", "test").decode(wave, 3);
    expect(evidence.phonemes.map(p => [p.p, p.start, p.end])).toEqual([["ب", 1, 2], ["ب", 4, 4]]);
    expect(evidence.posteriors?.nFrames).toBe(rows.length);
    expect(evidence.phonemes[0].conf).toBeCloseTo(.85);
  });
});
