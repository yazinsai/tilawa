import {describe,expect,it,vi} from "vitest";
import type {CheckerEvent,PhonemeEvidence,ReferenceIndex} from "@tilawa/core";
import type {LoadedModel} from "../src/model.js";
const state=vi.hoisted(()=>({audio:[] as number[]}));
const evidence:PhonemeEvidence={inventory:"test",frameMs:40,phonemes:[{p:"ب",start:0,end:0,conf:1}],
  posteriors:{startFrame:0,nFrames:2,k:1,ids:[1,0],probs:[1,1]}};
vi.mock("../src/listener.js",()=>({createListener:()=>({push:async()=>null,end:async()=>evidence,decoded:2})}));
vi.mock("@tilawa/core",async original=>({...await original<typeof import("@tilawa/core")>(),
  createTracker:()=>({push:()=>[],end:()=>[]}),
  createChecker:()=>{
    const mistakes:CheckerEvent[]=[];
    return {push(e:PhonemeEvidence){
      if(e.phonemes.some(p=>p.pauseMaskEvidence?.length))mistakes.push({type:"checker",id:"pause-mask/1",state:"confirmed",errorClass:"added_word",
        location:{surah:1,ayah:2,word:1},observed:"بت"} as CheckerEvent);
      return [];
    },end:()=>[],mistakes:()=>mistakes};
  },
}));
vi.mock("../src/seam-insertion.js",async original=>({...await original<typeof import("../src/seam-insertion.js")>(),
  addInsertionEvidence:async (_m:LoadedModel,_a:Float32Array,e:PhonemeEvidence[])=>e,
}));
vi.mock("../src/pause-mask.js",()=>({addPauseEvidence:async(m:LoadedModel,a:Float32Array,chunks:PhonemeEvidence[])=>{
  state.audio=Array.from(a);if(!m.pauseModel)return chunks;
  return chunks.map(e=>({...e,phonemes:e.phonemes.map(p=>({...p,pauseMaskEvidence:[{
    inventory:"test",frameMs:40,fromFrame:0,toFrame:2,phonemes:[],posteriors:e.posteriors!,seamInsertions:[],
  }]}))}));
}}));
import {createEngine} from "../src/session.js";
describe("pause-ear session completion",()=>{
  it("retains private audio and publishes folded lexical additions without requiring a teacher certificate",async()=>{
    const index={inventoryId:"test",version:"test",words:[],tokens:new Set("بت")} as unknown as ReferenceIndex;
    const model={manifest:{inventory:"test",id:"primary",frameMs:40,sampleRate:16000},supported:["added_word"],pauseModel:{}} as LoadedModel;
    const session=createEngine({reference:index,model}).startFreeRecitation();
    await session.pushAudio(Float32Array.of(.25,.5),16000);
    const events=await session.end();
    expect(state.audio).toEqual([.25,.5]);expect(session.evidence()).toEqual([]);
    expect(session.mistakes()).toMatchObject([{id:"pause-mask/1",errorClass:"added_word",state:"confirmed"}]);
    expect(events).toContainEqual(session.mistakes()[0]);
  });
});
