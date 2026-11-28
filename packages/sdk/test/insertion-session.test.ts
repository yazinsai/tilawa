import { describe, expect, it, vi } from "vitest";
import type { CheckerEvent, PhonemeEvidence, ReferenceIndex, SeamInsertionEvidence } from "@tilawa/core";
import type { LoadedModel } from "../src/model.js";

const fixtures = vi.hoisted(() => ({ audio: [] as number[], evidence: [] as PhonemeEvidence[] }));
vi.mock("../src/listener.js", () => ({ createListener: () => ({
  push: async () => null, end: async () => fixtures.evidence[0], decoded: 2,
}) }));
vi.mock("@tilawa/core", async original => ({
  ...await original<typeof import("@tilawa/core")>(),
  createTracker: () => ({ push: () => [{ type: "test-position" }], shown: () => [], silence: () => [], end: () => [] }),
  createChecker: () => {
    const found: CheckerEvent[] = [];
    return { push(e: PhonemeEvidence) {
      if (e.phonemes.some(p => p.seamInsertions?.length)) found.push({ type: "checker", state: "confirmed", errorClass: "added_word",
        location: { surah: 1, ayah: 2, word: 1 }, observed: "بت" } as CheckerEvent);
      return [];
    }, end: () => [], mistakes: () => found };
  },
}));
vi.mock("../src/seam-insertion.js", async original => ({
  ...await original<typeof import("../src/seam-insertion.js")>(),
  addInsertionEvidence: async (model: LoadedModel, audio: Float32Array, chunks: PhonemeEvidence[]) => {
    fixtures.audio = Array.from(audio);
    if (!model.insertionTeacher) return chunks;
    const certificate: SeamInsertionEvidence = { word: { surah: 1, ayah: 2, word: 1 }, observed: "بت", fromFrame: 0, toFrame: 1, marginal: .8 };
    return chunks.map(e => ({ ...e, phonemes: e.phonemes.map(p => ({ ...p, seamInsertions: [certificate] })) }));
  },
}));
import { createEngine } from "../src/session.js";

describe("auxiliary seam session completion", () => {
  it("retains private audio with recording disabled, publishes the addition and preserves primary events", async () => {
    fixtures.evidence = [{ inventory: "test", frameMs: 40, phonemes: [{ p: "ب", start: 0, end: 0, conf: 1 }],
      posteriors: { startFrame: 0, nFrames: 2, k: 1, ids: [1,0], probs: [1,1] } }];
    const index = { inventoryId: "test", version: "test", words: [], tokens: new Set("بت") } as unknown as ReferenceIndex;
    const model = { manifest: { inventory: "test", id: "test", frameMs: 40, sampleRate: 16000 }, supported: ["added_word"] } as unknown as LoadedModel;
    const run = async (teacher: boolean) => {
      const session = createEngine({ reference: index, model: { ...model, ...(teacher && { insertionTeacher: { decode: vi.fn() } }) } }).startFreeRecitation();
      await session.pushAudio(new Float32Array([.25,.5]), 16000);
      const events = await session.end();
      return { session, events };
    };
    const baseline = await run(false), judged = await run(true);
    expect(fixtures.audio).toEqual([.25,.5]);
    expect(judged.events.filter(e => e.type !== "checker")).toEqual(baseline.events);
    expect(judged.session.mistakes()).toMatchObject([{ state: "confirmed", errorClass: "added_word" }]);
    expect(judged.events).toContainEqual(judged.session.mistakes()[0]);
    expect(judged.session.evidence()).toEqual([]);
  });
});
