import { describe, expect, it } from "vitest";
import { BASIN_ARMS, basinComparison, basinCtc, basinUniqueGain, carryBasinEvidence, basinVerdictRemovals, validateBasinReadouts, basinObservations } from "../src/connected-basin.js";
import type { BasinReadouts } from "../src/connected-basin.js";
import type { ModelManifest } from "../src/model.js";
import type { CheckerEvent } from "@tilawa/core";

const manifest = {inventory: "test", tokens: "ab", sha256: "parent"} as ModelManifest;
function model(): BasinReadouts {
  return {schema: 1, inventory: "test", tokens: "ab", primarySha256: "parent", featureOutput: "h", width: 1, hidden: 1,
    center: [0], scale: [1], heads: Object.fromEntries(BASIN_ARMS.map(arm => [arm,
      {"0.weight": [[1]], "0.bias": [0], "2.weight": [[0], [arm === "connected" ? 3 : -3], [0]], "2.bias": [0, 0, 0]}])) as BasinReadouts["heads"]};
}
const candidate = {word: {surah: 1, ayah: 1, word: 1}, written: "a", rival: "b", fromFrame: 2, toFrame: 4};

describe("connected readout complete pronunciation scoring", () => {
  it("agrees with exhaustive complete CTC paths, including repeated-label blank separation", () => {
    const rows = [[.3, .5, .2], [.4, .2, .4], [.3, .3, .4]];
    for (const ids of [[1], [1, 1], [1, 2]]) {
      let mass = 0;
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) for (let c = 0; c < 3; c++) {
        const path = [a, b, c], collapsed = path.filter((p, i) => p !== 0 && (!i || p !== path[i-1]));
        if (JSON.stringify(ids) === JSON.stringify(collapsed)) mass += rows[0][a]*rows[1][b]*rows[2][c];
      }
      expect(basinCtc(rows.map(r => r.map(Math.log)), ids)).toBeCloseTo(Math.log(mass), 12);
    }
    expect(basinCtc([[Math.log(.5), Math.log(.5)]], [1, 1])).toBe(-Infinity);
  });
  it("isolates an interior preference absent from endpoints and arithmetic averaging", () => {
    const c = basinComparison(model(), [[1], [1]], [[-3, -1, -1], [-3, -1, -1]], candidate)!;
    expect(basinUniqueGain(c)).toBe(true);
    expect(basinUniqueGain({...c, endpointAWritten: c.endpointARival+1})).toBe(false);
    expect(basinUniqueGain({...c, connectedWritten: c.connectedRival})).toBe(false);
  });
  it("abstains on incomplete windows, absent phones, nonfinite observations and incompatible assets", () => {
    const m = model(), rows = [[-3, -1, -1], [-3, -1, -1]];
    expect(basinComparison(m, [[1]], rows, candidate)).toBeNull();
    expect(basinComparison(m, [[1], [1]], rows, {...candidate, written: "ac"})).toBeNull();
    expect(basinComparison(m, [[NaN], [1]], rows, candidate)).toBeNull();
    // Identical run-normalized hypotheses must tie, so a length-only change cannot get a veto.
    const lengthOnly = basinComparison(m, [[1], [1]], rows, {...candidate, written: "aa", rival: "a"})!;
    expect(lengthOnly.connectedWritten).toBe(lengthOnly.connectedRival);
    expect(validateBasinReadouts(m, manifest)).toBe(m);
    expect(() => validateBasinReadouts({...m, primarySha256: "other"}, manifest)).toThrow();
    expect(() => validateBasinReadouts({...m, scale: [0]}, manifest)).toThrow();
  });
  it("carries certificates without changing primary phones, posteriors or existing insertion judgments", () => {
    const chunks = [{inventory: "test", frameMs: 40, phonemes: [{p: "a", start: 0, end: 3, conf: .9}],
      posteriors: {k: 1, startFrame: 0, nFrames: 1, ids: [1], probs: [.9]}}];
    const c = basinComparison(model(), [[1], [1]], [[-3, -1, -1], [-3, -1, -1]], candidate)!;
    const enriched = carryBasinEvidence(chunks, [c]);
    expect(enriched[0].posteriors).toBe(chunks[0].posteriors);
    const {connectedBasinVetoes, ...phone} = enriched[0].phonemes[0];
    expect(phone).toEqual(chunks[0].phonemes[0]);
    expect(connectedBasinVetoes).toEqual([c]);
    expect(carryBasinEvidence(chunks, [])).toBe(chunks);
  });
  it("retracts only matching live occurrences, preserving later repetitions and insertion events", () => {
    const e = {id: "issue", timeMs: 100, expected: "a", observed: "b", location: candidate.word,
      errorClass: "wrong_word", state: "confirmed"} as CheckerEvent;
    const later = {...e, timeMs: 200}, added = {...e, errorClass: "added_word" as const};
    expect(basinVerdictRemovals([e, later, added], [e], [])).toEqual([e]);
  });
  it("retains the native frame grid, warm context and separate cache, trimming final padding", async () => {
    let calls = 0;
    const geometry = {tokens: "ab", fbank: {mels: 80, padValue: -23}, window: 45, shift: 32,
      states: [{name: "cache", shape: [1], dtype: "float32"}]} as ModelManifest;
    const backend = {name: "side", async release() {}, async run(feeds: Record<string, import("../src/model.js").Tensor>) {
      expect(Number(feeds.cache.data[0])).toBe(calls);
      const offset = 100*calls++;
      return {
        log_probs: {type: "float32" as const, dims: [1, 8, 3], data: new Float32Array(24)},
        new_cache: {type: "float32" as const, dims: [1], data: new Float32Array([calls])},
        h: {type: "float32" as const, dims: [1, 8, 1], data: Float32Array.from({length: 8}, (_, i) => offset+i)},
      };
    }};
    const rows = await basinObservations({model: model(), backend}, geometry, new Float32Array(16000));
    expect(rows.features).toHaveLength(23); expect(rows.baseline).toHaveLength(23);
    expect(rows.features[0]).toEqual([0]); expect(rows.features[8]).toEqual([100]); expect(calls).toBe(3);
  });
});
