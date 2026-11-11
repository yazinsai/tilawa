import { describe, expect, it, vi } from "vitest";
import type { CheckerEvent, CheckerOptions, PhonemeEvidence, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { LoadedModel } from "../src/model.js";

const hooks=vi.hoisted(()=>({comparison:vi.fn(),enrich:vi.fn()}));
vi.mock("../src/listener.js",()=>({createListener:()=>({
  push:async()=>null, decoded:4, end:async()=>({inventory:"test",frameMs:40,
    phonemes:[{p:"a",start:0,end:0,conf:1},{p:"c",start:3,end:3,conf:1}],
    posteriors:{startFrame:0,nFrames:4,k:1,ids:[1,0,0,3],probs:[1,1,1,1]},
  }),
})}));
vi.mock("@tilawa/core",async original=>({
  ...await original<typeof import("@tilawa/core")>(),
  createTracker:()=>({push:()=>[{type:"test-position"}],end:()=>[]}),
  createChecker:(_:ReferenceIndex,options:CheckerOptions)=>{
    const mistakes:CheckerEvent[]=[];
    return {push(e:PhonemeEvidence) {
      if (!e.phonemes.some(p=>p.p==="c")) return [];
      const candidate:SequenceEvidenceCandidate={word:{surah:1,ayah:1,word:1},written:"ab",rival:"ac",fromFrame:0,toFrame:4};
      const permitted=options.sequenceEvidence?.(candidate)!==false;
      const event={type:"checker",id:"one",state:permitted ? "confirmed":"uncertain",errorClass:"wrong_word",location:candidate.word} as CheckerEvent;
      if (permitted) mistakes.push(event);
      return [event];
    },end:()=>[],mistakes:()=>mistakes};
  },
}));
vi.mock("../src/gesture-emission.js",async original=>({
  ...await original<typeof import("../src/gesture-emission.js")>(),
  gestureComparison:hooks.comparison,
  addGestureEvidence:hooks.enrich,
}));
import { createEngine } from "../src/session.js";

describe("live gesture session verification",()=>{
  it("vetoes before publication with recording disabled and keeps primary events",async()=>{
    hooks.comparison.mockReset().mockImplementation((_m,_x,c)=>({...c,
      controlWritten:-2,controlRival:-1,overlapWritten:-1,overlapRival:-2}));
    hooks.enrich.mockReset().mockImplementation((_m,_audio,chunks)=>chunks);
    const reference={inventoryId:"test",version:"test",words:[],tokens:new Set("abc")} as unknown as ReferenceIndex;
    const model={manifest:{inventory:"test",id:"test",sampleRate:16000,frameMs:40},supported:["wrong_word"]} as LoadedModel;
    const wave=Float32Array.from({length:2800},(_,i)=>.2*Math.sin(i*.08));
    const run=async(scorer:boolean)=>{
      const s=createEngine({reference,model:{...model,...(scorer && {gesture:{} as NonNullable<LoadedModel["gesture"]>})}}).startFreeRecitation();
      for (let i=0;i<wave.length;i+=137) await s.pushAudio(wave.slice(i,i+137),16000);
      const events=await s.end();
      return {s,events};
    };
    const baseline=await run(false), judged=await run(true);
    expect(hooks.comparison).toHaveBeenCalledOnce();
    expect(hooks.comparison.mock.calls[0][1]).toHaveLength(16);
    expect(judged.events.filter(e=>e.type!=="checker")).toEqual(baseline.events.filter(e=>e.type!=="checker"));
    expect(judged.events.filter(e=>e.type==="checker")).toMatchObject([{state:"uncertain"}]);
    expect(judged.s.mistakes()).toEqual([]);
    expect(baseline.s.mistakes()).toHaveLength(1);
    expect(judged.s.evidence()).toEqual([]);
    expect(Array.from(hooks.enrich.mock.calls.at(-1)![1])).toEqual(Array.from(wave));
  });
});
