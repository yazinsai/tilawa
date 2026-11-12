import { describe, expect, it } from "vitest";
import { trackerInput } from "@tilawa/core";
import type { PhonemeEvidence } from "@tilawa/core";
import { createListener } from "../src/listener.js";
import type { LoadedModel } from "../src/model.js";

/** A deterministic graph with an inference advance independent of caller audio pushes. */
function model(chunkFrames: number): LoadedModel {
  return { supported: [], manifest: { schema: 1, id: "phase-fixture", format: "onnx", arch: "zipformer2-streaming-ctc",
    inventory: "phase-fixture", tokens: "بت", blank: 0, sampleRate: 16000, frameMs: 40,
    fbank: { mels: 80, padValue: 0 }, window: 4 * chunkFrames + 13, shift: 4 * chunkFrames,
    states: [], file: "unused", bytes: 0, sha256: "phase-fixture-hash" },
    backend: { name: "fixture", release: async () => {}, run: async () => {
      const lp = new Float32Array(chunkFrames * 3).fill(-20);
      for (let i = 0; i < chunkFrames; i++) lp[i * 3 + (i % 2 ? 2 : 1)] = 0;
      return { log_probs: { type: "float32", data: lp, dims: [1, chunkFrames, 3] } };
    } } };
}

async function run(push: number, chunkFrames: number) {
  const listener = createListener(model(chunkFrames), { posteriors: true });
  const samples = Float32Array.from({ length: 16000 }, (_, i) => .1 * Math.sin(i / 11));
  const out: PhonemeEvidence[] = [];
  for (let a = 0; a < samples.length; a += push) {
    const e = await listener.push(samples.subarray(a, a + push), 16000);
    if (e) out.push(e);
  }
  const end = await listener.end(); if (end) out.push(end);
  return out;
}

describe("listener inference coordinates", () => {
  it("keeps native phase identical for tiny audio pushes and a whole-clip push", async () => {
    const a = await run(320, 8), b = await run(16000, 8);
    expect(a.flatMap(e => e.phonemes)).toEqual(b.flatMap(e => e.phonemes));
    expect(a.flatMap(e => trackerInput(e).phonemes).map(p => p.chunkPhase))
      .toEqual(b.flatMap(e => trackerInput(e).phonemes).map(p => p.chunkPhase));
    for (const p of a.flatMap(e => e.phonemes)) {
      expect(p.inference?.model).toBe("phase-fixture-hash");
      expect(p.inference?.frame).toBe(p.start);
      expect(p.inference?.chunkStart).toBe(Math.floor(p.start / 8) * 8);
    }
  });

  it("gets chunk geometry from the inference output and keeps it on a partially drained final call", async () => {
    const chunks = await run(16000, 4), phones = chunks.flatMap(e => e.phonemes);
    expect(phones.at(-1)!.end % 4).toBe(2);
    for (const p of chunks.flatMap(e => trackerInput(e).phonemes)) {
      expect(p.inference?.chunkFrames).toBe(4);
      expect(p.chunkPhase).toBe((p.start % 4) / 4);
    }
  });
});
