import { describe, expect, it, vi } from "vitest";
import type { CheckerEvent, CheckerOptions, PhonemeEvidence, ReferenceIndex } from "@tilawa/core";
import type { LoadedModel } from "../src/model.js";
import type { CurvatureModel } from "../src/spectral-curvature.js";

vi.mock("../src/listener.js", () => ({ createListener: () => ({
  push: async () => null, decoded: 4,
  end: async (): Promise<PhonemeEvidence> => ({ inventory:"test", frameMs:40,
    phonemes:[{p:"a", start:0, end:0, conf:1}],
    posteriors:{startFrame:0, nFrames:4, k:1, ids:[1,0,0,0], probs:[1,1,1,1]} }),
}) }));
vi.mock("@tilawa/core", async original => ({
  ...await original<typeof import("@tilawa/core")>(),
  createTracker: () => ({push: () => [{type:"position", state:"searching", timeMs:0}], end: () => []}),
  createChecker: (_: unknown, options: CheckerOptions) => {
    const found: CheckerEvent[] = [];
    let decided = false;
    return { push(e: PhonemeEvidence) {
      if (decided) return [];
      decided = true;
      const permitted = options.sequenceEvidence?.({word:{surah:1, ayah:1, word:1}, written:"a", rival:"b", fromFrame:0, toFrame:4}) !== false;
      const event = {type:"checker", errorClass:"wrong_word", state:permitted ? "confirmed" : "uncertain",
        location:{surah:1, ayah:1, word:1}, timeMs:0} as CheckerEvent;
      if (permitted) found.push(event);
      return [event];
    }, end: () => [], mistakes: () => found };
  },
}));
import { createEngine } from "../src/session.js";

const arm = (written: number, rival: number) => ({center:Array(14).fill(0), scale:Array(14).fill(1),
  emissions:[0,written,rival].map(p => ({mean:Array(14).fill(p), variance:Array(14).fill(1)}))});
const scorer: CurvatureModel = {schema:1, inventory:"test", tokens:"ab", resolution:16, frameMs:40,
  phoneCounts:[1,1], stay:.8, maxFrames:2048, control:arm(100,0), curvature:arm(0,100)};
const index = {inventoryId:"test", version:"test", words:[], tokens:new Set("ab")} as unknown as ReferenceIndex;
const primary = {manifest:{inventory:"test", id:"test", tokens:"ab", frameMs:40, sampleRate:16000},
  supported:["wrong_word"]} as LoadedModel;
const audio = Float32Array.from({length:4000}, (_,i) => .2*Math.sin(i*.2));
const run = async (side: boolean, recording: boolean) => {
  const session = createEngine({reference:index, model:{...primary, ...(side && {spectralCurvature:scorer})}, recordEvidence:recording}).startFreeRecitation();
  const events = [...await session.pushAudio(audio,16000), ...await session.end()];
  return {session, events};
};

describe("streaming curvature session integration", () => {
  it("uses original audio observations with recording disabled", async () => {
    const baseline = await run(false,false), changed = await run(true,false);
    expect(baseline.session.mistakes()).toHaveLength(1);
    expect(changed.session.mistakes()).toEqual([]);
    expect(changed.events.filter(e => e.type === "checker")).toMatchObject([{state:"uncertain"}]);
    expect(changed.events.filter(e => e.type === "position")).toEqual(baseline.events.filter(e => e.type === "position"));
    expect(changed.session.evidence()).toEqual([]);
  });
  it("carries exact-window certificates without replacing primary tokens or posterior values", async () => {
    const baseline = await run(false,true), changed = await run(true,true);
    const certificates = changed.session.evidence().flatMap(e => e.phonemes.flatMap(p => p.curvatureVetoes ?? []));
    expect(certificates).toHaveLength(1);
    expect(certificates[0]).toMatchObject({written:"a", rival:"b", fromFrame:0, toFrame:4});
    const primaryEvidence = changed.session.evidence().map(e => ({...e, phonemes:e.phonemes.map(({curvatureVetoes, ...p}) => p)}));
    expect(primaryEvidence).toEqual(baseline.session.evidence());
  });
});
