import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { createListener } from "../src/listener.js";
import type { InferenceBackend, ModelManifest } from "../src/model.js";

it("carries a model's raw-band history once per window and resets it for a new stream", async () => {
  const base = JSON.parse(readFileSync(new URL("./fixtures/tiny/manifest.json", import.meta.url), "utf8")) as ModelManifest;
  const manifest: ModelManifest = { ...base, states: [{ name: "band_history", shape: [1, 4, 80], dtype: "float32" }] };
  const histories: number[][] = [];
  const inputs: Float32Array[] = [];
  const backend: InferenceBackend = {
    name: "band-state-contract",
    async release() {},
    async run(feeds) {
      histories.push(Array.from(feeds.band_history.data as Float32Array));
      inputs.push((feeds.x.data as Float32Array).slice());
      const newHistory = inputs.at(-1)!.slice((manifest.shift - 4) * 80, manifest.shift * 80);
      for (let i = 0; i < newHistory.length; i++) newHistory[i] -= manifest.fbank.padValue;
      const vocab = manifest.tokens.length + 1;
      const lp = new Float32Array((manifest.shift / 4) * vocab).fill(-10);
      for (let i = 0; i < manifest.shift / 4; i++) lp[i * vocab] = 0;
      return {
        log_probs: { type: "float32", data: lp, dims: [1, manifest.shift / 4, vocab] },
        new_band_history: { type: "float32", data: newHistory, dims: [1, 4, 80] },
      };
    },
  };
  const model = { manifest, backend, supported: [] };
  const listener = createListener(model, { posteriors: true });
  const wave = Float32Array.from({ length: 16000 * 3 }, (_, i) => Math.sin(i * 0.2) * 0.1);
  await listener.push(wave, 16000);
  await listener.end();
  expect(histories.length).toBeGreaterThan(3);
  expect(histories[0].every(x => x === 0)).toBe(true);
  for (let call = 1; call < histories.length; call++) {
    const committed = inputs[call - 1].slice((manifest.shift - 4) * 80, manifest.shift * 80);
    for (let i = 0; i < committed.length; i++) committed[i] -= manifest.fbank.padValue;
    expect(histories[call]).toEqual(Array.from(committed));
  }
  const previousCalls = histories.length;
  const next = createListener(model);
  await next.push(wave.subarray(0, 16000), 16000);
  expect(histories[previousCalls].every(x => x === 0)).toBe(true);
  expect(listener.frames).toBe(Math.floor((wave.length - 400) / 160) + 1);
});
