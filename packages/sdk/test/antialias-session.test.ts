import { describe, expect, it, vi } from "vitest";
import type { AntialiasEvidence, CheckerEvent, PhonemeEvidence, ReferenceIndex } from "@tilawa/core";
import type { LoadedModel } from "../src/model.js";

const fixtures = vi.hoisted(() => ({ audio: [] as number[], evidence: null as PhonemeEvidence | null,
  graph: false }));
vi.mock("../src/listener.js", () => ({ createListener: () => ({
  push: async () => null, end: async () => fixtures.evidence, decoded: 2,
}) }));
vi.mock("@tilawa/core", async original => ({
  ...await original<typeof import("@tilawa/core")>(),
  createTracker: () => ({ push: () => [{ type: "test-position" }], end: () => [] }),
  createChecker: () => {
    const found: CheckerEvent[] = [];
    return { push(e: PhonemeEvidence) {
      if (!e.phonemes.length) return [];
      const common = { type: "checker", v: 1, state: "confirmed", timeMs: 40, confidence: 1 };
      const events = [{ ...common, id: "insertion", errorClass: "added_word",
        location: { surah: 1, ayah: 1, word: 2 }, observed: "نك" } as CheckerEvent];
      if (fixtures.graph || !e.phonemes.some(p => p.antialiasRechecks?.length)) events.push({
        ...common, id: "wrong", errorClass: "wrong_word", expected: "نت", observed: "نك",
        location: { surah: 1, ayah: 1, word: 1 },
      } as CheckerEvent);
      found.push(...events);
      return events;
    }, end: () => [], mistakes: () => found };
  },
}));
vi.mock("../src/antialias-recheck.js", () => ({
  addAntialiasEvidence: async (model: LoadedModel, audio: Float32Array, chunks: PhonemeEvidence[]) => {
    fixtures.audio = Array.from(audio);
    if (!model.antialiasModel) return chunks;
    const certificate: AntialiasEvidence = { word: { surah: 1, ayah: 1, word: 1 },
      written: "نت", accused: "نك", wordWritten: "نت", wordAccused: "نك", recovered: "نت",
      fromFrame: 0, toFrame: 2, slotFromFrame: 0, slotToFrame: 2, writtenNll: 1, accusedNll: 30 };
    return chunks.map(e => ({ ...e, phonemes: e.phonemes.map(p => ({ ...p, antialiasRechecks: [certificate] })) }));
  },
}));
import { createEngine } from "../src/session.js";

const index = { inventoryId: "test", version: "test", words: [], tokens: new Set("نتك") } as unknown as ReferenceIndex;
const model = { manifest: { inventory: "test", id: "test", frameMs: 40, sampleRate: 16000 },
  supported: ["wrong_word", "added_word"] } as unknown as LoadedModel;
async function run(side: boolean) {
  fixtures.evidence = { inventory: "test", frameMs: 40, phonemes: [{ p: "ن", start: 0, end: 0, conf: 1 }],
    posteriors: { startFrame: 0, nFrames: 2, k: 1, ids: [1, 0], probs: [1, 1] } };
  const session = createEngine({ reference: index,
    model: { ...model, ...(side && { antialiasModel: model }) } }).startFreeRecitation();
  await session.pushAudio(new Float32Array([.25, .5]), 16000);
  const events = await session.end();
  return { session, events };
}

describe("antialias session completion", () => {
  it("retains private audio, withdraws the certified string warning and preserves insertion and tracker events", async () => {
    fixtures.graph = false;
    const primary = await run(false), side = await run(true);
    expect(fixtures.audio).toEqual([.25, .5]);
    expect(side.session.mistakes()).toEqual(primary.session.mistakes().filter(e => e.errorClass === "added_word"));
    expect(side.events.filter(e => e.type !== "checker")).toEqual(primary.events.filter(e => e.type !== "checker"));
    expect(side.events).toContainEqual(expect.objectContaining({ id: "wrong", state: "uncertain" }));
    expect(side.session.evidence()).toEqual([]);
  });
  it("retains a warning at the same word when ordinary replay still confirms the graph", async () => {
    fixtures.graph = true;
    const primary = await run(false), side = await run(true);
    expect(side.session.mistakes()).toEqual(primary.session.mistakes());
    expect(side.events).toEqual(primary.events);
  });
});
