import { describe, expect, it, vi } from "vitest";
import type { CheckerEvent, PhonemeEvidence, ReferenceIndex, SeamInsertionEvidence } from "@tilawa/core";
import type { LoadedModel } from "../src/model.js";

const fixtures = vi.hoisted(() => ({ audio: [] as number[] }));
const certificate = (word: number, observed: string): SeamInsertionEvidence => ({
  word: { surah: 1, ayah: 2, word }, observed, fromFrame: 0, toFrame: 1, marginal: .9,
});
const warning = (id: string, word: number, observed: string): CheckerEvent => ({
  type: "checker", v: 1, id, timeMs: 40, state: "confirmed", errorClass: "added_word",
  location: { surah: 1, ayah: 2, word }, expected: "ب", observed, confidence: .9,
});

vi.mock("../src/listener.js", () => ({ createListener: () => ({
  push: async () => null, decoded: 2, end: async (): Promise<PhonemeEvidence> => ({
    inventory: "test", frameMs: 40, phonemes: [{ p: "ب", start: 0, end: 0, conf: .9 }],
    posteriors: { startFrame: 0, nFrames: 2, k: 1, ids: [1, 0], probs: [.9, 1] },
  }),
}) }));
vi.mock("@tilawa/core", async original => ({
  ...await original<typeof import("@tilawa/core")>(),
  createTracker: () => ({ push: () => [], end: () => [] }),
  createChecker: () => {
    const found: CheckerEvent[] = [];
    return {
      push(e: PhonemeEvidence) {
        for (const p of e.phonemes) {
          for (const c of p.seamInsertions ?? []) found.push(warning("folded", c.word.word, c.observed));
          for (const s of p.pauseMaskEvidence ?? []) for (const c of s.seamInsertions ?? [])
            found.push(warning("pause-mask/side", c.word.word, c.observed));
        }
        return [];
      }, end: () => [], mistakes: () => found,
    };
  },
}));
vi.mock("../src/seam-insertion.js", async original => ({
  ...await original<typeof import("../src/seam-insertion.js")>(),
  addInsertionEvidence: async (_model: LoadedModel, _audio: Float32Array, chunks: PhonemeEvidence[]) =>
    chunks.map(e => ({ ...e, phonemes: e.phonemes.map(p => ({ ...p, seamInsertions: [certificate(1, "بت")] })) })),
}));
vi.mock("../src/pause-mask.js", () => ({
  addPauseEvidence: async (model: LoadedModel, audio: Float32Array, chunks: PhonemeEvidence[]) => {
    fixtures.audio = Array.from(audio);
    if (!model.pauseModel) return chunks;
    return chunks.map(e => ({ ...e, phonemes: e.phonemes.map(p => ({ ...p, pauseMaskEvidence: [{
      inventory: "test", frameMs: 40, fromFrame: 0, toFrame: 1,
      phonemes: [{ p: "ت", start: 0, end: 0, conf: .9 }],
      posteriors: { startFrame: 0, nFrames: 1, k: 1, ids: [2], probs: [.9] },
      seamInsertions: [certificate(2, "تب")],
    }] })) }));
  },
}));
import { createEngine } from "../src/session.js";

describe("pause ear session isolation", () => {
  it.each([false, true])("retains folded teacher wins and primary evidence with recording=%s", async recordEvidence => {
    const reference = { inventoryId: "test", version: "test", words: [], tokens: new Set("بت") } as unknown as ReferenceIndex;
    const primary = { manifest: { inventory: "test", id: "primary", frameMs: 40, sampleRate: 16000 },
      supported: ["added_word"], insertionTeacher: { decode: vi.fn() } } as unknown as LoadedModel;
    const run = async (side: boolean) => {
      const session = createEngine({ reference, model: { ...primary, ...(side && { pauseModel: primary }) }, recordEvidence }).startFreeRecitation();
      await session.pushAudio(Float32Array.of(.25, .5), 16000);
      return { events: await session.end(), mistakes: session.mistakes(), evidence: session.evidence() };
    };
    const baseline = await run(false), candidate = await run(true);
    expect(baseline.mistakes).toEqual([warning("folded", 1, "بت")]);
    expect(candidate.mistakes).toEqual([...baseline.mistakes, warning("pause-mask/side", 2, "تب")]);
    expect(candidate.events.filter(e => e.type !== "checker" || !e.id.startsWith("pause-mask/"))).toEqual(baseline.events);
    expect(candidate.evidence.map(e => ({ ...e, phonemes: e.phonemes.map(({ pauseMaskEvidence: _, ...p }) => p) }))).toEqual(baseline.evidence);
    expect(fixtures.audio).toEqual([.25, .5]);
  });
});
