import { describe,expect,it,vi } from "vitest";
import type { EngineEvent,PhonemeEvidence,ReferenceIndex } from "@tilawa/core";
import type { LoadedModel } from "../src/model.js";

const state=vi.hoisted(() => ({ primary: [] as PhonemeEvidence[],masked: [] as PhonemeEvidence[],audio: [] as number[],inputs: [] as string[] }));
vi.mock("../src/listener.js",() => ({ createListener: () => ({ push: async () => null,end: async () => state.primary[0],decoded: 24 }) }));
vi.mock("../src/gap-decode.js",() => ({ spliceContextFreeGaps: async (_: unknown,_a: unknown,_r: unknown,chunks: PhonemeEvidence[]) => chunks }));
vi.mock("../src/pause-mask.js",async original => ({
  ...await original<typeof import("../src/pause-mask.js")>(),
  decodePauseMasked: async (_: LoadedModel,audio: Float32Array,spans: unknown,k: number) => {
    expect(spans).toEqual([{ fromFrame: 3,toFrame: 23 }]); expect(k).toBe(1);
    state.audio=Array.from(audio); return state.masked;
  },
}));
vi.mock("@tilawa/core",async original => ({
  ...await original<typeof import("@tilawa/core")>(),
  createChecker: () => ({ push: () => [],end: () => [],mistakes: () => [] }),
  createTracker: () => ({ end: () => [],push: (e: PhonemeEvidence) => {
    state.inputs.push(JSON.stringify(e.phonemes));
    return e.phonemes.map(p => ({ type: "position",state: "locked",timeMs: p.start*40,
      expected: p.start===0 ? { surah: 1,ayah: 1,word: 1 } : { surah: 1,ayah: 2,word: p.start===12 ? 1 : 2 },
    })) as EngineEvent[];
  } }),
}));
import { createEngine } from "../src/session.js";

describe("scoped pause session evidence",() => {
  it("keeps primary tracker calls and values intact and retains private audio when recording is off",async () => {
    state.primary=[{ inventory: "test",frameMs: 40,phonemes: [
      { p: "ب",start: 0,end: 0,conf: .9 },{ p: "ت",start: 12,end: 12,conf: .9 },{ p: "ن",start: 23,end: 23,conf: .9 },
    ],posteriors: { startFrame: 0,nFrames: 24,k: 1,
      ids: Array.from({ length: 24 },(_,t) => t>=3 && t<11 ? 0 : t===12 ? 2 : t===23 ? 3 : 1),probs: Array(24).fill(.9) } }];
    state.masked=[{ ...state.primary[0],phonemes: [{ p: "ن",start: 11,end: 12,conf: .9 },{ p: "ت",start: 15,end: 16,conf: .9 }],
      posteriors: { ...state.primary[0].posteriors!,ids: Array.from({ length: 24 },(_,t) =>
        t===11 || t===12 ? 3 : t===15 || t===16 ? 2 : state.primary[0].posteriors!.ids[t]) } }];
    const index={ inventoryId: "test",version: "test",tokens: new Set("بتن"),words: [
      { surah: 1,ayah: 1,word: 1,start: 0,end: 1 },{ surah: 1,ayah: 2,word: 1,start: 1,end: 2 },{ surah: 1,ayah: 2,word: 2,start: 2,end: 3 },
    ] } as ReferenceIndex;
    const model={ manifest: { id: "test",inventory: "test",sampleRate: 16000,frameMs: 40 },supported: [] } as unknown as LoadedModel;
    const audio=new Float32Array(16000).fill(.25);
    const run=async (paused: boolean,record: boolean) => {
      state.inputs=[];
      const session=createEngine({ reference: index,recordEvidence: record,model: { ...model,...(paused && { pauseModel: model }) } }).startFreeRecitation();
      await session.pushAudio(audio,16000); const events=await session.end();
      return { evidence: session.evidence(),events,inputs: [...state.inputs] };
    };
    const baseline=await run(false,true),paused=await run(true,true);
    expect(paused.events).toEqual(baseline.events); expect(paused.inputs).toEqual(baseline.inputs);
    expect(paused.evidence.flatMap(e => e.phonemes).map(({ pauseMaskEvidence: _,...p }) => p)).toEqual(baseline.evidence.flatMap(e => e.phonemes));
    expect(paused.evidence.map(e => e.posteriors)).toEqual(baseline.evidence.map(e => e.posteriors));
    expect(paused.evidence.flatMap(e => e.phonemes)[0].pauseMaskEvidence).toMatchObject([{ fromFrame: 3,toFrame: 23 }]);
    const privateRun=await run(true,false);
    expect(privateRun.evidence).toEqual([]); expect(privateRun.events).toEqual(baseline.events);
    expect(state.audio).toEqual(Array.from(audio));
  });
});
