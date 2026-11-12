import { describe, expect, it } from "vitest";
import { carryAngularEvidence, featureBoundaryScore, featureComparison, featureObservations, featureReversal, identityRows, magnitudeProbe, validateAngularModel } from "../src/angular-phone.js";
import type { AngularModel } from "../src/angular-phone.js";
import type { InferenceBackend, ModelManifest, Tensor } from "../src/model.js";
import { createListener } from "../src/listener.js";

const model = (): AngularModel => ({ schema: 1, inventory: "test", tokens: "بت", dimensions: 2, primarySha256: "a".repeat(64),
  featureOutput: "encoder", normRange: [1, 10], angular: { vectors: [[1, 0], [-1, 0]], concentration: 4 },
  control: { weights: [[0, 0], [.6, 0]], bias: [2, 0] } });
const candidate = { word: { surah: 1, ayah: 1, word: 1 }, written: "ب", rival: "ت", fromFrame: 4, toFrame: 8 };
const features = Array.from({ length: 4 }, () => [5, 0]);
const primary = Array.from({ length: 4 }, () => [.7, .1, .2].map(Math.log));

describe("angular identity comparison", () => {
  it("keeps primary blank and speech mass exactly while changing identity", () => {
    for (const arm of ["angular", "control"] as const) {
      const rows = identityRows(model(), features, primary, arm);
      for (let t = 0; t < rows.length; t++) {
        expect(rows[t][0]).toBe(primary[t][0]);
        expect(rows[t].slice(1).reduce((sum, v) => sum + Math.exp(v), 0)).toBeCloseTo(.3, 14);
      }
    }
    const allBlank = identityRows(model(), [[1, 0]], [[0, -Infinity, -Infinity]], "angular");
    expect(allBlank).toEqual([[0, -Infinity, -Infinity]]);
    const noBlank = identityRows(model(), [[1, 0]], [[-Infinity, 0, -Infinity]], "angular");
    expect(noBlank[0][0]).toBe(-Infinity);
    expect(noBlank[0].slice(1).reduce((s, v) => s + Math.exp(v), 0)).toBeCloseTo(1, 14);
  });
  it("is invariant to positive vector magnitudes, and isolates an affine preference reversal", () => {
    const rows = identityRows(model(), features, primary, "angular");
    const rescaled = features.map((x, i) => x.map(v => v * (i + 1) * .4));
    expect(identityRows(model(), rescaled, primary, "angular")).toEqual(rows);
    expect(identityRows(model(), rescaled, primary, "control")).not.toEqual(identityRows(model(), features, primary, "control"));
    const probe = magnitudeProbe(model(), features, primary, candidate)!;
    expect(probe.magnitudeSensitive).toBe(true);
    expect(probe.angularWritten).toBe(true);
    expect(probe.sweep.every(s => s.radius >= 1 && s.radius <= 10 + 1e-12)).toBe(true);
    expect(Math.max(...probe.sweep.map(s => s.angularMargin)) - Math.min(...probe.sweep.map(s => s.angularMargin))).toBeLessThan(1e-10);
  });
  it("scores complete candidates symmetrically and requires strict disagreement", () => {
    const c = featureComparison(model(), features, primary, candidate)!;
    expect(featureReversal(c)).toBe(true);
    const reverse = featureComparison(model(), features, primary, { ...candidate, written: candidate.rival, rival: candidate.written })!;
    expect(reverse.angularWritten).toBe(c.angularRival);
    expect(reverse.controlRival).toBe(c.controlWritten);
    expect(featureReversal(reverse)).toBe(false);
    expect(featureReversal({ ...c, controlWritten: c.controlRival })).toBe(false);
    expect(featureReversal({ ...c, angularWritten: NaN })).toBe(false);
  });
  it("matches exhaustive CTC path summation, including a genuine doubled phone", () => {
    const probs = [[.4, .5, .1], [.3, .2, .5], [.2, .4, .4], [.3, .3, .4]];
    for (const ids of [[1], [1, 2], [1, 1]]) {
      let mass = 0;
      for (let code = 0; code < 3 ** 4; code++) {
        let rest = code;
        const path = probs.map(() => { const p = rest % 3; rest = Math.floor(rest / 3); return p; });
        const collapsed = path.filter((p, i) => p !== 0 && (i === 0 || p !== path[i - 1]));
        if (JSON.stringify(collapsed) === JSON.stringify(ids)) mass += path.reduce((s, p, t) => s * probs[t][p], 1);
      }
      expect(featureBoundaryScore(probs.map(r => r.map(Math.log)), ids)).toBeCloseTo(Math.log(mass), 12);
    }
    expect(featureBoundaryScore([[0, -1, -1]], [1, 1])).toBe(-Infinity);
  });
  it("abstains on undefined directions, unknown phones and impossible candidates", () => {
    expect(featureComparison(model(), features, primary, { ...candidate, rival: "ث" })).toBeNull();
    expect(featureComparison(model(), [[0, 0]], primary.slice(0, 1), candidate)).toBeNull();
    expect(featureComparison(model(), [[NaN, 0]], primary.slice(0, 1), candidate)).toBeNull();
    expect(featureComparison(model(), features, primary.slice(1), candidate)).toBeNull();
    expect(featureComparison(model(), features.slice(0, 1), primary.slice(0, 1), { ...candidate, rival: "تت" })).toBeNull();
  });
  it("carries certificates without altering primary phones or posterior rows", () => {
    const chunks = [{ inventory: "test", frameMs: 40, phonemes: [{ p: "ت", start: 4, end: 7, conf: .99 }],
      posteriors: { k: 1, startFrame: 4, nFrames: 4, ids: [2, 2, 2, 2], probs: [.99, .99, .99, .99] } }];
    const c = featureComparison(model(), features, primary, candidate)!;
    const enriched = carryAngularEvidence(chunks, [c]);
    expect(enriched[0].posteriors).toBe(chunks[0].posteriors);
    expect(enriched[0].phonemes[0]).toEqual({ ...chunks[0].phonemes[0], angularPhoneVetoes: [c] });
    expect(chunks[0].phonemes[0]).not.toHaveProperty("angularPhoneVetoes");
  });
  it("rejects nonunit classes, unmatched inventory, wrong head size and invalid concentration", () => {
    const m = model(), manifest = { inventory: m.inventory, tokens: m.tokens, sha256: m.primarySha256 } as ModelManifest;
    expect(validateAngularModel(m, manifest)).toBe(m);
    expect(() => validateAngularModel({ ...m, primarySha256: "b".repeat(64) }, manifest)).toThrow();
    expect(() => validateAngularModel({ ...m, angular: { ...m.angular, vectors: [[2, 0], [-1, 0]] } }, manifest)).toThrow();
    expect(() => validateAngularModel({ ...m, angular: { ...m.angular, concentration: 0 } }, manifest)).toThrow();
    expect(() => validateAngularModel({ ...m, control: { ...m.control, bias: [0] } }, manifest)).toThrow();
    expect(() => validateAngularModel({ ...m, normRange: [0, 1] }, manifest)).toThrow();
  });
  it("replays the native streaming inputs and cache state with final frame truncation", async () => {
    const manifest = { inventory: "test", tokens: "بت", sampleRate: 16000, frameMs: 40, window: 45, shift: 32, chunk: 16,
      fbank: { mels: 80, padValue: -23.0259 }, states: [{ name: "count", shape: [1], dtype: "int64" }] } as ModelManifest;
    const captures: unknown[][] = [[], []];
    const backend = (which: number): InferenceBackend => ({ name: "test", async run(feeds) {
      captures[which].push(Object.fromEntries(Object.entries(feeds).map(([k, v]) => [k, Array.from(v.data, Number)])));
      const count = Number(feeds.count.data[0]), frames = 8;
      return { new_count: { type: "int64", dims: [1], data: new BigInt64Array([BigInt(count + 1)]) },
        encoder: { type: "float32", dims: [1, frames, 2], data: Float32Array.from({ length: frames * 2 }, (_, i) => count * frames * 2 + i) },
        log_probs: { type: "float32", dims: [1, frames, 3], data: Float32Array.from({ length: frames * 3 }, (_, i) => [-3, -1, -2][i % 3]) }
      } satisfies Record<string, Tensor>;
    } });
    const samples = Float32Array.from({ length: 14000 }, (_, i) => .2 * Math.sin(i * .08));
    const listener = createListener({ manifest, backend: backend(0), supported: [] }, { posteriors: true, topK: 3 });
    let nativeFrames = 0;
    for (let s = 0; s < samples.length; s += 1000) nativeFrames += (await listener.push(samples.slice(s, s + 1000), 16000))?.posteriors?.nFrames ?? 0;
    nativeFrames += (await listener.end())?.posteriors?.nFrames ?? 0;
    const observed = await featureObservations(backend(1), manifest, model(), samples);
    expect(captures[1]).toEqual(captures[0]);
    expect(observed.observations).toHaveLength(nativeFrames);
    expect(observed.observations.map(r => r[0])).toEqual(Array.from({ length: nativeFrames }, (_, i) => 2 * i));
  });
});
