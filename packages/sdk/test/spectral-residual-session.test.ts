import { describe, expect, it, vi } from "vitest";
import type { CheckerEvent, PhonemeEvidence, ReferenceIndex, SpectralResidualRequest } from "@tilawa/core";
import type { LoadedModel } from "../src/model.js";

const state = vi.hoisted(() => ({ emitted: false, calls: 0, requests: [] as SpectralResidualRequest[] }));
const evidence: PhonemeEvidence = { inventory: "test", frameMs: 40, phonemes: [{ p: "ت", start: 0, end: 1, conf: 1 }],
  posteriors: { startFrame: 0, nFrames: 2, k: 1, ids: [1, 1], probs: [1, 1] } };
vi.mock("../src/listener.js", () => ({ createListener: () => ({
  async push() { if (state.emitted) return null; state.emitted = true; return evidence; }, async end() { return null; }, decoded: 2,
}) }));
vi.mock("@tilawa/core", async original => ({
  ...await original<typeof import("@tilawa/core")>(),
  createTracker: () => ({ push: () => [], end: () => [] }),
  createChecker: (_index: unknown, options: any) => {
    const mistakes: CheckerEvent[] = [];
    return { push() {
      const r = { word: 0, written: "ب", accused: "ت", fromFrame: 0, toFrame: 2, slotFromFrame: 0, slotToFrame: 2 };
      options.spectralDecision?.("test", r);
      const e = { type: "checker", id: "test", state: options.spectralResidual?.(r) === false ? "uncertain" : "confirmed",
        errorClass: "wrong_word", location: { surah: 1, ayah: 1, word: 1 }, timeMs: 80 } as CheckerEvent;
      if (e.state === "confirmed") mistakes.push(e);
      return [e];
    }, end: () => [], mistakes: () => mistakes };
  },
}));
import { createEngine } from "../src/session.js";

function model(preferWritten: boolean): LoadedModel {
  return { manifest: { inventory: "test", id: "test", frameMs: 40, sampleRate: 16000 }, supported: ["wrong_word"],
    spectralResidual: { parameters: { schema: 1, tokens: "بت", projection: [new Array(256).fill(0)], predictor: Array.from({ length: 81 }, () => [0]),
      scale: [1], means: [[0], [0], [0]], variance: [1], covered: [true, true, true] },
      compare(_features, r) { state.calls++; state.requests.push(r); return { writtenLogLikelihood: preferWritten ? -1 : -3, accusedLogLikelihood: -2 }; } },
  } as LoadedModel;
}
const index = { inventoryId: "test", version: "test", words: [{ surah: 1, ayah: 1, word: 1 }], tokens: new Set("بت") } as unknown as ReferenceIndex;

describe("live spectral residual judgment", () => {
  it("judges on privately retained audio before publishing the final warning, even without evidence recording", async () => {
    state.emitted = false; state.calls = 0; state.requests = [];
    const session = createEngine({ reference: index, model: model(true) }).startFreeRecitation();
    const seen: any[] = []; session.subscribe(e => seen.push(e));
    const events = await session.pushAudio(new Float32Array(1600), 16000);
    expect(state.calls).toBe(1);
    expect(events).toMatchObject([{ state: "uncertain", errorClass: "wrong_word", timeMs: 80 }]);
    expect(seen.filter(e => e.type === "checker")).toEqual(events);
    expect(session.mistakes()).toEqual([]);
    expect(session.evidence()).toEqual([]);
  });

  it("keeps a rival-supported warning and abstains for evidence-only sessions without audio", async () => {
    state.emitted = false; state.calls = 0;
    const session = createEngine({ reference: index, model: model(false) }).startFreeRecitation();
    expect(await session.pushAudio(new Float32Array(1600), 16000)).toMatchObject([{ state: "confirmed" }]);
    expect(session.mistakes()).toHaveLength(1);
    state.calls = 0;
    const bypass = createEngine({ reference: index, model: model(true) }).startFreeRecitation();
    expect(bypass.pushEvidence(evidence)).toMatchObject([{ state: "confirmed" }]);
    expect(state.calls).toBe(0);
  });
});
