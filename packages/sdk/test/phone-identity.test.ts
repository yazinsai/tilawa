import { describe, expect, it } from "vitest";
import { conditionalPhoneRows } from "../src/phone-identity.js";
import { createListener } from "../src/listener.js";
import type { LoadedModel, Tensor } from "../src/model.js";

describe("conditional non-blank identity", () => {
  it("changes identity while preserving each blank and total speech probability", () => {
    const base = Float32Array.from([.91, .08, .01, .2, .3, .5], Math.log);
    const got = conditionalPhoneRows(base, new Float32Array([-2, 2, 3, -3]), 3);
    for (let t = 0; t < 2; t++) {
      expect(got[t * 3]).toBe(base[t * 3]);
      const mass = Math.exp(got[t * 3 + 1]) + Math.exp(got[t * 3 + 2]);
      expect(mass).toBeCloseTo(-Math.expm1(base[t * 3]), 7);
    }
    expect(got[2]).toBeGreaterThan(got[1]);
    expect(got[4]).toBeGreaterThan(got[5]);
  });

  it("cannot emit at exact silence and retains very small speech mass", () => {
    const got = conditionalPhoneRows(new Float32Array([0, -Infinity, -Infinity, -1e-8, -20, -20]),
      new Float32Array([50, -50, 0, 0]), 3);
    expect([...got.subarray(0, 3)]).toEqual([0, -Infinity, -Infinity]);
    const mass = Math.exp(got[4]) + Math.exp(got[5]);
    // The reconstructed log probabilities are float32, even at tiny mass.
    expect(Math.abs(mass / 1e-8 - 1)).toBeLessThan(1e-6);
  });

  it("is invariant to a common identity-logit shift", () => {
    const base = Float32Array.from([.5, .1, .4], Math.log);
    expect([...conditionalPhoneRows(base, new Float32Array([10, 12]), 3)])
      .toEqual([...conditionalPhoneRows(base, new Float32Array([-10, -8]), 3)]);
  });

  it("rejects incompatible or invalid head outputs", () => {
    expect(() => conditionalPhoneRows(new Float32Array(6), new Float32Array(3), 3)).toThrow(/geometry/);
    expect(() => conditionalPhoneRows(new Float32Array([.1, -1, -1]), new Float32Array(2), 3)).toThrow(/blank/);
    expect(() => conditionalPhoneRows(new Float32Array([-1, -1, -1]), new Float32Array([NaN, 0]), 3)).toThrow(/identity/);
  });

  it("uses the new head in streaming decode and posterior output, preserving cache state", async () => {
    let calls = 0;
    const model: LoadedModel = {
      supported: [],
      manifest: { schema: 1, id: "synthetic", format: "onnx", arch: "zipformer2-streaming-ctc",
        inventory: "synthetic", tokens: "نم", blank: 0, sampleRate: 16000, frameMs: 40,
        fbank: { mels: 80, padValue: -23 }, window: 45, shift: 32,
        states: [{ name: "cache", shape: [1], dtype: "int64" }], file: "unused", bytes: 0, sha256: "" },
      backend: {
        name: "synthetic",
        async release() {},
        async run(feeds) {
          expect((feeds.cache.data as BigInt64Array)[0]).toBe(BigInt(calls++));
          const rows = Float32Array.from(Array.from({ length: 8 }, () => [.2, .75, .05]).flat(), Math.log);
          const identity = Float32Array.from(Array.from({ length: 8 }, () => [-10, 10]).flat());
          const result: Record<string, Tensor> = {
            log_probs: { type: "float32", dims: [1, 8, 3], data: rows },
            phone_identity_logits: { type: "float32", dims: [1, 8, 2], data: identity },
            new_cache: { type: "int64", dims: [1], data: new BigInt64Array([BigInt(calls)]) },
          };
          return result;
        },
      },
    };
    const listener = createListener(model, { posteriors: true, topK: 3 });
    const first = await listener.push(new Float32Array(16000), 16000);
    const last = await listener.end();
    const chunks = [first, last].filter(e => e !== null);
    expect(chunks.flatMap(e => e.phonemes).map(p => p.p)).toEqual(["م"]);
    let frame = 0;
    for (const c of chunks) {
      const p = c.posteriors!;
      expect(p.startFrame).toBe(frame);
      frame += p.nFrames;
      for (let i = 0; i < p.nFrames; i++) {
        const blank = [0, 1, 2].find(j => p.ids[i * 3 + j] === 0)!;
        expect(p.probs[i * 3 + blank]).toBe(.2);
        expect(p.ids[i * 3]).toBe(2);
        expect(p.probs[i * 3]).toBe(.8);
      }
    }
    expect(calls).toBeGreaterThan(1);
  });
});
