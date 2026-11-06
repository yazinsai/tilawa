import { expect, it } from "vitest";
import { createOfflineStudent } from "../src/offline-student.js";
import type { InferenceBackend, Tensor } from "../src/model.js";

it("uses the entire variable-length crop with no cache and collapses CTC blanks", async () => {
  const feeds: Record<string, Tensor>[] = [];
  const backend: InferenceBackend = { name: "offline-test", release: async () => {}, async run(f) {
    feeds.push(f);
    return { log_probs: { type: "float32", dims: [1, 5, 3],
      data: Float32Array.from([.01,.98,.01, .01,.98,.01, .98,.01,.01, .01,.98,.01, .01,.01,.98], Math.log) },
      out_lens: { type: "int64", dims: [1], data: BigInt64Array.of(5n) } };
  } };
  const ear = createOfflineStudent(backend, "بت", "test");
  const e = await ear.decode(new Float32Array(16000), 3);
  expect(e.phonemes.map(p => [p.p, p.start, p.end])).toEqual([["ب",0,1],["ب",3,3],["ت",4,4]]);
  expect(e.posteriors?.nFrames).toBe(5);
  await ear.decode(new Float32Array(32000), 3);
  expect(Object.keys(feeds[0]).sort()).toEqual(["x", "x_lens"]);
  expect(feeds[1].x.dims[1]).toBeGreaterThan(feeds[0].x.dims[1]);
  expect(Number(feeds[1].x_lens.data[0])).toBe(feeds[1].x.dims[1]);
});

it("rejects a crop too short for the subsampling layers", async () => {
  const backend: InferenceBackend = { name: "unused", release: async () => {}, run: async () => { throw new Error("called"); } };
  await expect(createOfflineStudent(backend, "ب", "test").decode(new Float32Array(400), 2)).rejects.toThrow("shorter");
});
