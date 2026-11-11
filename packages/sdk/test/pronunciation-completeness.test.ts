import { describe,expect,it } from "vitest";
import { carryCompletenessEvidence,completenessProbability,completenessRejects } from "../src/pronunciation-completeness.js";
import type { CompletenessModel } from "../src/pronunciation-completeness.js";
import type { InferenceBackend } from "../src/model.js";

describe("complete accused pronunciation recheck",()=>{
  it("marginalizes uncertain crop edges and submits every accused phone without a reference prompt",async()=>{
    const calls:Record<string,any>[]=[];
    const head:InferenceBackend={name:"fake",release:async()=>{},run:async(feeds)=>{
      calls.push(feeds);
      return {complete_probability:{type:"float32",dims:[1],data:new Float32Array([calls.length/4])}};
    }};
    const m={tokens:"بت",center:Array(32).fill(0),scale:Array(32).fill(1)} as CompletenessModel;
    expect(await completenessProbability(head,m,Array.from({length:5},()=>Array(32).fill(0)),"بتب")).toBe(.5);
    expect(calls.map(c=>c.features.dims[1])).toEqual([5,4,4]);
    expect(calls.every(c=>Object.keys(c).join(",")==="features,pronunciation")).toBe(true);
    expect(calls.every(c=>Array.from(c.pronunciation.data).join(",")==="1,2,1")).toBe(true);
    expect(await completenessProbability(head,m,[],"بت")).toBeNull();
    expect(await completenessProbability(head,m,[Array(32).fill(0)],"?")).toBeNull();
  });
  it("abstains on unavailable and invalid probabilities, and at the preservation boundary",()=>{
    expect(completenessRejects(.1,.2)).toBe(true);
    for(const [p,c] of [[.2,.2],[NaN,.2],[-1,.2],[.1,0],[.1,.6]])expect(completenessRejects(p,c)).toBe(false);
  });
  it("keeps primary tokens and posteriors intact when transporting certificates",()=>{
    const chunks=[{inventory:"test",frameMs:40,phonemes:[{p:"ب",start:0,end:1,conf:.9}]}];
    const c={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:0,toFrame:2,completeProbability:.1,cutoff:.2};
    const result=carryCompletenessEvidence(chunks,[c]);
    expect(chunks[0].phonemes[0]).not.toHaveProperty("completenessVetoes");
    expect(result[0].phonemes[0]).toMatchObject(chunks[0].phonemes[0]);
    expect(result[0].phonemes[0].completenessVetoes).toEqual([c]);
  });
});
