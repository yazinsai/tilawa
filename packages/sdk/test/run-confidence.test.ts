import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { PhonemeEvidence } from "@tilawa/core";
import { createListener } from "../src/listener.js";
import { loadNodeModel } from "../src/node.js";
import { decodeWav } from "../src/wav.js";
import { occurrenceConfidence, runCorrelation, validateRunConfidence } from "../src/run-confidence.js";
import type { RunConfidenceCalibration } from "../src/run-confidence.js";
import type { LoadedModel } from "../src/model.js";

const fit: RunConfidenceCalibration = {
  schema: 1, modelSha256: "test", inventory: "test", mode: "selection", prior: 0.8,
  correct: { mean: 4, sd: 2 }, incorrect: { mean: 1, sd: 2 },
  dependence: { correlation: 0, shrinkage: 4 },
};

describe("selection-adjusted occurrence confidence", () => {
  it("discounts extra chances to select the same maximum and retains highly distinguishing peaks", () => {
    const single = occurrenceConfidence([0.99], fit);
    expect(occurrenceConfidence([0.99, 0.8], fit)).toBeLessThan(single);
    expect(occurrenceConfidence([0.99999], fit)).toBeGreaterThan(0.99);
    expect(occurrenceConfidence([0.5], fit)).toBeLessThan(0.6);
  });

  it("uses fewer opportunities for dependent frames; the peak-only control ignores length", () => {
    const independent = occurrenceConfidence([0.99, 0.8], fit);
    const dependent = occurrenceConfidence([0.99, 0.8], { ...fit, dependence: { correlation: 0.9, shrinkage: 4 } });
    expect(dependent).toBeGreaterThan(independent);
    const control = { ...fit, mode: "peak" as const };
    expect(occurrenceConfidence([0.99, 0.8], control)).toBe(occurrenceConfidence([0.99], control));
    expect(runCorrelation([0.5], 0.4, 4)).toBe(0.4);
    expect(runCorrelation([0.7, 0.7, 0.7], 0.4, 4)).toBeCloseTo(0.4);
  });

  it("handles extreme peaks and rejects invalid or mismatched fits", () => {
    for (const p of [0, 1e-20, 0.5, 1 - 1e-15, 1]) {
      const probability = occurrenceConfidence([p], fit);
      expect(Number.isFinite(probability)).toBe(true);
      expect(probability).toBeGreaterThanOrEqual(0);
      expect(probability).toBeLessThanOrEqual(1);
    }
    expect(() => occurrenceConfidence([], fit)).toThrow();
    expect(validateRunConfidence(fit, "test", "test")).toBe(fit);
    for (const value of [null, { ...fit, prior: 1 }, { ...fit, correct: { mean: 0, sd: 0 } }, { ...fit, mode: "other" }]) {
      expect(() => validateRunConfidence(value, "test", "test")).toThrow();
    }
    expect(() => validateRunConfidence(fit, "different", "test")).toThrow();
  });

  it("calibrates the fragment emitted at a window boundary without re-emitting its continuation", async () => {
    const manifest = JSON.parse(readFileSync(new URL("./fixtures/tiny/manifest.json", import.meta.url), "utf8"));
    let frame = 0;
    const model: LoadedModel = {
      manifest, supported: [], runConfidence: fit,
      backend: {
        name: "synthetic",
        async run(feeds) {
          const vocab = manifest.tokens.length + 1;
          const data = new Float32Array(8 * vocab).fill(-20);
          for (let t = 0; t < 8; t++, frame++) {
            // The first articulation spans windows 1 and 2, then a blank and a different phone.
            const id = frame < 11 ? 1 : frame === 11 ? 0 : 2;
            data[t * vocab + id] = Math.log(frame === 0 ? .99 : .8);
          }
          return { ...Object.fromEntries(manifest.states.map((s: { name: string }) => [`new_${s.name}`, feeds[s.name]])),
            log_probs: { type: "float32" as const, data, dims: [1, 8, vocab] } };
        },
        async release() {},
      },
    };
    const listener = createListener(model);
    const first = await listener.push(new Float32Array(16_000), 16_000);
    const last = await listener.end();
    const phones = [...(first?.phonemes ?? []), ...(last?.phonemes ?? [])];
    expect(phones.map(p => p.p)).toEqual([manifest.tokens[0], manifest.tokens[1]]);
    expect(phones[0]).toEqual({ p: manifest.tokens[0], start: 0, end: 7,
      conf: Math.round(occurrenceConfidence([.99, ...Array(7).fill(.8)], fit) * 1e4) / 1e4 });
  });

  it("preserves primary token identities, frame bounds and posterior rows at every push size", async () => {
    const wav = decodeWav(readFileSync(new URL("./fixtures/tlog-clean-00010-630.wav", import.meta.url)));
    async function decode(calibrated: boolean, step: number) {
      const model = await loadNodeModel(new URL("./fixtures/tiny", import.meta.url).pathname);
      if (calibrated) model.runConfidence = fit;
      const listener = createListener(model, { posteriors: true });
      const chunks: PhonemeEvidence[] = [];
      for (let i = 0; i < wav.samples.length; i += step) {
        const e = await listener.push(wav.samples.subarray(i, i + step), wav.sampleRate);
        if (e) chunks.push(e);
      }
      const end = await listener.end();
      if (end) chunks.push(end);
      await model.backend.release();
      return chunks;
    }
    const base = await decode(false, 4000);
    const small = await decode(true, 4000);
    const large = await decode(true, 48000);
    const phones = (es: PhonemeEvidence[]) => es.flatMap(e => e.phonemes);
    expect(phones(small)).toEqual(phones(large));
    expect(phones(small).some((p, i) => p.conf !== phones(base)[i].conf)).toBe(true);
    const identity = (es: PhonemeEvidence[]) => phones(es).map(({ conf, ...p }) => p);
    expect(identity(small)).toEqual(identity(base));
    expect(small.map(e => e.posteriors)).toEqual(base.map(e => e.posteriors));
    // Sequence-support scores depend only on these unchanged rows.
  });
});
