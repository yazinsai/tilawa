import {describe,expect,it,vi} from "vitest";
import type {ReferenceIndex} from "@tilawa/core";
import type {LoadedModel} from "../src/model.js";
const state=vi.hoisted(()=>({seenRows:0,rows:[] as Float32Array[]}));
vi.mock("@tilawa/core",async original=>({...await original<typeof import("@tilawa/core")>(),
  createTracker:()=>({push:()=>[],end:()=>[]}),
  createChecker:(_index:unknown,options:any)=>({
    push:()=>{
      state.seenRows=state.rows.length;
      const read=options.omissionRead({word:1,fromFrame:0,toFrame:4});
      return [{type:"checker",v:1,id:"unit",timeMs:160,state:read?"uncertain":"confirmed",errorClass:"omitted_word",
        location:{surah:1,ayah:1,word:2},expected:"ba",confidence:1}];
    },end:()=>[],mistakes:()=>[],
  }),
}));
import {createEngine} from "../src/session.js";

describe("live segment judgment",()=>{
  it("rejects a declared scorer without a streaming feature provider",()=>{
    const reference={inventoryId:"test"} as ReferenceIndex;
    const model={manifest:{inventory:"test",segmentPresence:{file:"scorer.json"}}} as LoadedModel;
    expect(()=>createEngine({reference,model})).toThrow("streaming feature provider");
  });
  it("collects frozen features before the checker decides a primary evidence cell",async()=>{
    state.rows.splice(0);state.seenRows=0;
    const index={inventoryId:"test",tokens:new Set("ab"),phonemes:"abbaba",
      words:Array.from({length:3},(_,i)=>({surah:1,ayah:1,word:i+1,start:2*i,end:2*i+2}))} as ReferenceIndex;
    const step=vi.fn(async (_x,keep)=>{for(let i=0;i<keep;i++)state.rows.push(Float32Array.from([0]));});
    const run=vi.fn(async()=>({log_probs:{type:"float32" as const,dims:[1,16,3],
      data:Float32Array.from(Array.from({length:16},()=>[Math.log(.01),Math.log(.98),Math.log(.01)]).flat())}}));
    const model={manifest:{inventory:"test",tokens:"ab",sampleRate:16000,frameMs:40,window:83,shift:64,states:[],
      fbank:{mels:80,padValue:-23.025850929940457}},backend:{name:"unit",run,release:async()=>{}},supported:["omitted_word"],
      segmentPresence:{model:{schema:1,inventory:"test",tokens:"ab",hidden:1,mean:[0],std:[1],featureNode:"mid",encoderSha256:"unit",
        weights:{"acoustic.weight":[[0]],"acoustic.bias":[0],"text.weight":[Array(39).fill(0)],"text.bias":[0],duration:[0,0],skip_logit:-20}},
        stream:()=>({rows:state.rows,step}),features:async()=>state.rows}} as unknown as LoadedModel;
    const session=createEngine({reference:index,model,listener:{topK:3}}).startFreeRecitation();
    const events=await session.pushAudio(new Float32Array(20000),16000);
    expect(step).toHaveBeenCalledOnce();expect(state.seenRows).toBe(16);
    expect(events.some(e=>e.type==="checker"&&e.state==="uncertain")).toBe(true);
    expect(events.some(e=>e.type==="checker"&&e.state==="confirmed")).toBe(false);
    expect(run).toHaveBeenCalledOnce();
  });
});
