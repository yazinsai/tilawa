import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ReferenceIndex, PhonemeEvidence, SeamInsertionWindow, ShadowInsertionInterval } from "@tilawa/core";
import type { LoadedModel } from "../src/model.js";

const state = vi.hoisted(()=>({ intervals: [] as ShadowInsertionInterval[], marginal:.8, lastEnd:114 }));
const shadow = vi.hoisted(()=>vi.fn());
const loop = vi.hoisted(()=>vi.fn());
vi.mock("@tilawa/core",async original=>({
  ...await original<typeof import("@tilawa/core")>(),
  createTracker:()=>{
    let sent=false;
    return { push:()=>{
      if(sent)return []; sent=true;
      return [125,130].map((frame,i)=>({type:"position",state:"locked",timeMs:frame*40,
        observed:[{word:{surah:1,ayah:2,word:i+1}}]}));
    },end:()=>[] };
  },
  createChecker:(_:unknown,options:{seamPhoneLoop(c:SeamInsertionWindow):void})=>({push:()=>[],end:()=>{
    options.seamPhoneLoop({word:2,seamFromFrame:100,seamToFrame:140,prefixFrame:90}); return [];
  }}),
  lexicalWords:()=>({fragmentOfSeam:()=>false}),
  shadowInsertionIntervals:(...args:unknown[])=>{shadow(...args);return state.intervals;},
  seamPhoneLoop:(...args:unknown[])=>{
    loop(...args);return {marginal:state.marginal,phones:[5,6],fromFrame:110,toFrame:115};
  },
}));
import { addInsertionEvidence } from "../src/seam-insertion.js";

const index={inventoryId:"test",tokens:new Set("بتثجحخ"),phonemes:"بتثج",
  words:[..."بتثج"].map((_,i)=>({surah:1,ayah:i<2?1:2,word:i%2+1,start:i,end:i+1}))} as ReferenceIndex;
function evidence(): PhonemeEvidence[] {
  return [{inventory:"test",frameMs:40,phonemes:[{p:"ب",start:100,end:100,conf:1}],
    posteriors:{startFrame:0,nFrames:160,k:1,ids:Array(160).fill(0),probs:Array(160).fill(1)}}];
}
function model() {
  return {manifest:{sampleRate:16000},insertionTeacher:{decode:vi.fn(async()=>({
    inventory:"test",frameMs:40,phonemes:[{p:"ح",start:110,end:110,conf:1},{p:"خ",start:114,end:state.lastEnd,conf:1}],
    posteriors:{startFrame:0,nFrames:148,k:1,ids:Array(148).fill(0),probs:Array(148).fill(1)},
  }))}} as unknown as LoadedModel;
}
beforeEach(()=>{
  vi.clearAllMocks();state.marginal=.8;state.lastEnd=114;
  state.intervals=[{fromFrame:110,toFrame:115,observed:"حخ",plain:5,inserted:2,gain:3,margin:1}];
});
describe("listener-stage shadow tracker certificates",()=>{
  it("keeps teacher speech inside student blanks and confines the loop to the exported interval",async()=>{
    const primary=evidence(),before=structuredClone(primary);
    const result=await addInsertionEvidence(model(),new Float32Array(160*640),primary,index);
    expect(shadow.mock.calls[0][2]).toMatchObject([{p:"ح",start:110},{p:"خ",start:114}]);
    expect(loop.mock.calls[0][8]).toEqual([110,115]);
    expect(result[0].phonemes[0].seamInsertions).toEqual([{
      kind:"phone-loop",word:{surah:1,ayah:2,word:1},observed:"حخ",fromFrame:110,toFrame:115,marginal:.8,
    }]);
    expect(result[0].posteriors).toBe(primary[0].posteriors);
    expect(primary).toEqual(before);
  });

  it("does not add a loop when the shadow continues in order",async()=>{
    state.intervals=[];
    const primary=evidence();
    expect(await addInsertionEvidence(model(),new Float32Array(160*640),primary,index)).toBe(primary);
    expect(loop).not.toHaveBeenCalled();
  });

  it("clips a teacher run at the primary step onto the next word",async()=>{
    state.lastEnd=144;
    await addInsertionEvidence(model(),new Float32Array(160*640),evidence(),index);
    expect(shadow.mock.calls[0][2][1]).toMatchObject({start:114,end:130});
  });

  it("requires a loop marginal strictly above one half",async()=>{
    state.marginal=.5;
    const primary=evidence();
    expect(await addInsertionEvidence(model(),new Float32Array(160*640),primary,index)).toBe(primary);
  });
});
