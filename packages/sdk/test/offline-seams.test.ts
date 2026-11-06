import { beforeEach, expect, it, vi } from "vitest";
import { loadReferenceIndex } from "@tilawa/core";
import type { EngineEvent, PhonemeEvidence } from "@tilawa/core";

const mock = vi.hoisted(() => ({ events: [] as unknown[] }));
vi.mock("@tilawa/core", async original => ({ ...await original<typeof import("@tilawa/core")>(),
  createTracker: () => ({ push: () => mock.events.splice(0), end: () => [] }),
}));
import { offlineSeams } from "../src/offline-seams.js";

const index = loadReferenceIndex({ version: "test", text: "test", profile: {},
  inventory: { id: "test", tokens: "بتثجدحخز" }, ayahs: [
    { s:1, a:1, w:["بت","ثج"], p:"بتثج", o:[0,2], iso:["بت","ثج"] },
    { s:1, a:2, w:["دح","خز"], p:"دحخز", o:[0,2], iso:["دح","خز"] },
  ] });
const steps: EngineEvent[] = index.words.map((w,i) => ({ type:"position", v:1, timeMs:[0,400,1400,2400][i],
  state:"locked", expected:w, observed:[], referenceVersion:"test" }));
const evidence: PhonemeEvidence[] = [{ inventory:"test", frameMs:40,
  phonemes:[{p:"ج",start:11,end:11,conf:1},{p:"د",start:31,end:31,conf:1}],
  posteriors:{startFrame:0,nFrames:70,k:1,ids:Array.from({length:70},(_,i)=>i===11?4:i===31?5:0), probs:Array(70).fill(1)} }];
beforeEach(() => { mock.events = structuredClone(steps); });

it("uses a labelled primary pause and tracker word-two step without a word fit", () => {
  const original = structuredClone(evidence);
  expect(offlineSeams(evidence,index)).toEqual([{word:2,fromFrame:12,toFrame:68}]);
  expect(evidence).toEqual(original);
});

it("does not bridge a relocation or a skipped tracker word", () => {
  mock.events = [steps[0],steps[1],{ ...steps[2], type:"position", relocated:{from:index.words[0],to:index.words[2]} },steps[3]];
  expect(offlineSeams(evidence,index)).toEqual([]);
  mock.events = [steps[0],steps[2],steps[3]];
  expect(offlineSeams(evidence,index)).toEqual([]);
});

it("does not call unblank speech a pause or use evidence with no rows", () => {
  const speech = structuredClone(evidence);
  for (let f=12; f<31; f++) (speech[0].posteriors!.ids as number[])[f]=1;
  expect(offlineSeams(speech,index)).toEqual([]);
  expect(offlineSeams(evidence.map(({posteriors:_,...e})=>e),index)).toEqual([]);
});
