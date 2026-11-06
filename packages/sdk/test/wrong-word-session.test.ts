import { describe, expect, it, vi } from "vitest";
import type { CheckerEvent, CheckerOptions, PhonemeEvidence, ReferenceIndex, WrongWordAccusation } from "@tilawa/core";
import type { LoadedModel } from "../src/model.js";
const state = vi.hoisted(() => ({ audio: [] as number[], agree: false }));
const location = { surah: 1, ayah: 2, word: 1 };
const accusation: WrongWordAccusation = { word: location, last: location, fromFrame: 0, toFrame: 1, accusation: { bar: 20 } };
const warning = (id: string, errorClass: "wrong_word" | "added_word"): CheckerEvent => ({
  type: "checker", v: 1, id, timeMs: 40, state: "confirmed", errorClass,
  location, expected: "ب", observed: "ت", confidence: .9,
});
vi.mock("../src/listener.js", () => ({ createListener: () => ({
  push: async () => null, decoded: 2, end: async (): Promise<PhonemeEvidence> => ({
    inventory: "test", frameMs: 40, phonemes: [{ p: "ت", start: 0, end: 0, conf: .9 }],
    posteriors: { startFrame: 0, nFrames: 2, k: 1, ids: [1, 0], probs: [.9, 1] },
  }),
}) }));
vi.mock("@tilawa/core", async original => ({
  ...await original<typeof import("@tilawa/core")>(),
  createTracker: () => ({ push: () => [], end: () => [] }),
  createChecker: (_index: ReferenceIndex, options: CheckerOptions) => {
    const found: CheckerEvent[] = [];
    return { push(e: PhonemeEvidence) {
      if (e.phonemes.length && !found.some(e => e.id === "primary")) {
        options.wrongWord?.(accusation, "primary", false); found.push(warning("primary", "wrong_word"));
      }
      for (const p of e.phonemes) if (p.seamInsertions?.length) found.push(warning("folded", "added_word"));
      return [];
    }, end: () => [], mistakes: () => found };
  },
}));
vi.mock("../src/seam-insertion.js", async original => ({
  ...await original<typeof import("../src/seam-insertion.js")>(),
  addInsertionEvidence: async (_m: LoadedModel, _a: Float32Array, chunks: PhonemeEvidence[]) => chunks.map(e => ({ ...e,
    phonemes: e.phonemes.map(p => ({ ...p, seamInsertions: [{ word: location, observed: "ت", fromFrame: 0, toFrame: 1, marginal: .9 }] })) })),
}));
vi.mock("../src/wrong-word-agreement.js", () => ({
  addWrongWordAgreements: async (model: LoadedModel, audio: Float32Array, chunks: PhonemeEvidence[]) => {
    state.audio = Array.from(audio);
    if (!model.pauseModel) return chunks;
    return chunks.map(e => ({ ...e, phonemes: e.phonemes.map(p => ({ ...p, wrongWordAgreements: [{
      acquireMs: 1000, primary: accusation, rejudged: state.agree ? [accusation] : [], evidence: [],
    }] })) }));
  },
}));
import { createEngine } from "../src/session.js";
const reference = { inventoryId: "test", version: "test", words: [], tokens: new Set("بت") } as unknown as ReferenceIndex;
const model = { manifest: { inventory: "test", id: "primary", frameMs: 40, sampleRate: 16000 },
  supported: ["wrong_word", "added_word"], insertionTeacher: { decode: vi.fn() } } as unknown as LoadedModel;
async function run(side: boolean, recordEvidence: boolean) {
  const session = createEngine({ reference, model: { ...model, ...(side && { pauseModel: model }) }, recordEvidence }).startFreeRecitation();
  await session.pushAudio(Float32Array.of(.25, .5), 16000);
  return { events: await session.end(), mistakes: session.mistakes(), evidence: session.evidence() };
}
describe("wrong-word agreement session completion", () => {
  it.each([false, true])("retracts only the wrong word and keeps the folded teacher addition with recording=%s", async record => {
    state.agree = false;
    expect((await run(false, record)).mistakes).toEqual([warning("primary", "wrong_word"), warning("folded", "added_word")]);
    const result = await run(true, record);
    expect(result.mistakes).toEqual([warning("folded", "added_word")]);
    expect(result.events).toContainEqual({ ...warning("primary", "wrong_word"), state: "uncertain" });
    expect(state.audio).toEqual([.25, .5]);
    if (!record) expect(result.evidence).toEqual([]);
  });
  it("retains the primary warning when the side ear agrees", async () => {
    state.agree = true;
    const result = await run(true, true);
    expect(result.mistakes).toEqual([warning("primary", "wrong_word"), warning("folded", "added_word")]);
    expect(result.events.some(e => e.type === "checker" && e.state === "uncertain")).toBe(false);
  });
});
