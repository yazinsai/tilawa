import { describe, expect, it } from "vitest";
import { carryPauseEvidence,pauseFloor,pauseMaskSpans,decodePauseMasked } from "../src/pause-mask.js";
import type { PhonemeEvidence, ReferenceIndex } from "@tilawa/core";
import { createListener } from "../src/listener.js";
import type { LoadedModel, Tensor } from "../src/model.js";

const index = { words: [
  { surah: 1, ayah: 1, word: 1, start: 0, end: 1 },
  { surah: 1, ayah: 2, word: 1, start: 1, end: 2 },
  { surah: 1, ayah: 2, word: 2, start: 2, end: 3 },
] } as ReferenceIndex;
const locks = [{ wi: 0, timeMs: 0 }, { wi: 1, timeMs: 480 }, { wi: 2, timeMs: 960 }];
const evidence = (blank: number): PhonemeEvidence[] => [{ inventory: "test", frameMs: 40, phonemes: [],
  posteriors: { startFrame: 0, nFrames: 24, k: 2,
    ids: Array.from({ length: 24 }, () => [0,1]).flat(),
    probs: Array.from({ length: 24 }, (_, t) => t >= 3 && t < 11 ? [blank,1-blank] : [.1,.9]).flat() } }];

describe("pause-blocked auxiliary streaming", () => {
  it("carries only scoped rows, merges overlapping masks and preserves primary values", () => {
    const primary=evidence(.9); primary[0].phonemes=[{ p: "ب",start: 0,end: 0,conf: .8 }];
    const masked=structuredClone(primary); masked[0].phonemes=[{ p: "ت",start: 4,end: 5,conf: .9 }];
    const r=masked[0].posteriors!;
    for (let t=0;t<r.nFrames;t++) r.ids[t*2]=2;
    const original=structuredClone(primary);
    const carried=carryPauseEvidence(primary,masked,[{ fromFrame: 3,toFrame: 8 },{ fromFrame: 5,toFrame: 8 }]);
    expect(carried[0].posteriors).toBe(primary[0].posteriors);
    expect(carried[0].phonemes.map(({ pauseMaskEvidence: _,...p }) => p)).toEqual(primary[0].phonemes);
    expect(primary).toEqual(original);
    const scope=JSON.parse(JSON.stringify(carried))[0].phonemes[0].pauseMaskEvidence;
    expect(scope).toHaveLength(1);
    expect(scope[0]).toMatchObject({ fromFrame: 3,toFrame: 8,phonemes: masked[0].phonemes,
      posteriors: { startFrame: 3,nFrames: 5,k: 2 } });
    expect(scope[0].posteriors.ids).toHaveLength(10);
  });
  it("uses the labelled pause onset and ends at the primary step onto word 2", () => {
    expect(pauseMaskSpans(evidence(.9),locks,index)).toEqual([{ fromFrame: 3, toFrame: 24 }]);
    expect(pauseMaskSpans(evidence(.49),locks,index)).toEqual([]);
  });
  it("requires contiguous in-order primary steps across the seam", () => {
    expect(pauseMaskSpans(evidence(.9),[locks[0],locks[2],locks[1]],index)).toEqual([]);
  });
  it("masks every labelled seam pause instead of discarding earlier warm-up spans", () => {
    const chunks=evidence(.9), p=chunks[0].posteriors!;
    for (let t=12;t<20;t++) p.probs[t*2]=.9,p.probs[t*2+1]=.1;
    const spans=pauseMaskSpans(chunks,[locks[0],{ wi: 1,timeMs: 840 },locks[2]],index);
    expect(spans).toEqual([{ fromFrame: 3,toFrame: 24 },{ fromFrame: 12,toFrame: 24 }]);
    expect(Array.from(pauseFloor(spans,20,16))).toEqual([6n,6n,6n,6n,...Array(12).fill(24n)]);
  });
  it("keeps original positions and masks only query frames inside the half-open span", () => {
    const spans = [{ fromFrame: 3, toFrame: 8 }];
    expect(Array.from(pauseFloor(spans,0,16))).toEqual([-1n,-1n,-1n,-1n,-1n,-1n,6n,6n,6n,6n,6n,6n,6n,6n,6n,6n]);
    expect(Array.from(pauseFloor(spans,16,16))).toEqual(Array(16).fill(-1n));
  });
  it("skips encoder calls outside spans while keeping the original grid and positions", async () => {
    const feeds: Record<string, Tensor>[]=[];
    const model: LoadedModel={ manifest:{ schema:1,id:"side",format:"onnx-int8",arch:"zipformer2-streaming-ctc",
      inventory:"test",tokens:"ب",blank:0,sampleRate:16000,frameMs:40,fbank:{mels:80,padValue:-23.0259},window:45,shift:32,
      states:[{name:"processed_lens",shape:[1],dtype:"int64"}],file:"side",bytes:0,sha256:"",
      pauseMask:{input:"pause_floor",clockMs:20} }, supported:[],backend:{name:"stub",async release(){},async run(f){
        feeds.push(f);
        return {log_probs:{type:"float32",data:new Float32Array(16),dims:[1,8,2]},
          new_processed_lens:{type:"int64",data:BigInt64Array.of((f.processed_lens.data as BigInt64Array)[0]+16n),dims:[1]}};
      }} };
    await decodePauseMasked(model,new Float32Array(16000),[{fromFrame:9,toFrame:14}],2);
    expect(feeds).toHaveLength(1);
    expect(Array.from(feeds[0].processed_lens.data)).toEqual([16n]);
    expect(Array.from(feeds[0].pause_floor.data)).toEqual([-1n,-1n,...Array(10).fill(18n),-1n,-1n,-1n,-1n]);
    const ordinary=createListener(model);
    await ordinary.push(new Float32Array(16000),16000); await ordinary.end();
    expect(feeds.slice(1).every(f=>!f.pause_floor)).toBe(true);
  });
});
