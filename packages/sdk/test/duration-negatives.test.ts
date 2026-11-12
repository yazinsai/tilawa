import { describe, expect, it } from "vitest";
import { compatibilityScore, marginalCompatibility, carryDurationNegativeEvidence, validateCompatibilityHead } from "../src/duration-negatives.js";
import type { CompatibilityHead, DurationNegativeModel } from "../src/duration-negatives.js";
import { durationNegativeVeto } from "@tilawa/core";

const head: CompatibilityHead = {
  embedding: Array.from({length:3}, (_,i) => [i-1,...new Array(15).fill(0)]),
  weight: Array.from({length:32}, (_,i) => Array.from({length:48}, (_,d) => i===0 && d===32 ? 1 : 0)),
  bias: new Array(32).fill(0), output: [1,...new Array(31).fill(0)], intercept:0,
};
const model = {tokens:"بت",center:new Array(32).fill(0),scale:new Array(32).fill(1)} as DurationNegativeModel;

describe("complete pronunciation compatibility", () => {
  it("integrates every boundary with the same normalized prior", () => {
    const e=[[1,2,3,4],[4,3,2,1]];
    const sums=[1+3+2+1,1+2+2+1,1+2+3+1].map(v=>Math.exp(v/4));
    expect(marginalCompatibility(e)).toBeCloseTo(Math.log(sums.reduce((a,b)=>a+b,0)/3),12);
    expect(marginalCompatibility(Array.from({length:3},()=>new Array(9).fill(2)))).toBeCloseTo(2,12);
  });
  it("cannot omit a candidate phone or consume nonexistent observations", () => {
    expect(marginalCompatibility([[1],[2]])).toBeNull();
    expect(marginalCompatibility([[1,2],[3,NaN]])).toBeNull();
    expect(compatibilityScore(head,model,[[0]],"ب")).toBeNull();
    expect(compatibilityScore(head,model,[new Array(32).fill(0)],"بت")).toBeNull();
    expect(compatibilityScore(head,model,[new Array(32).fill(0)],"ن")).toBeNull();
  });
  it("scores all candidate phones, averaging uncertain outer boundaries", () => {
    const x=Array.from({length:8},()=>new Array(32).fill(0));
    expect(compatibilityScore(head,model,x,"تت")).toBeCloseTo(Math.tanh(1),12);
    expect(compatibilityScore(head,model,x,"بب")).toBeCloseTo(0,12);
    expect(validateCompatibilityHead(head,3)).toBe(head);
    expect(()=>validateCompatibilityHead({...head,intercept:Infinity},3)).toThrow();
  });
  it("requires strict and finite opposite preferences from both heads", () => {
    expect(durationNegativeVeto({matchedMargin:1,randomMargin:-1})).toBe(true);
    for (const margins of [[0,-1],[1,0],[-1,-1],[1,1],[NaN,-1],[1,-Infinity]]) {
      expect(durationNegativeVeto({matchedMargin:margins[0],randomMargin:margins[1]})).toBe(false);
    }
  });
  it("carries certificates through serialization without changing primary tokens or posteriors", () => {
    const chunks=[{inventory:"test",frameMs:40,phonemes:[{p:"ب",start:0,end:1,conf:.9}],
      posteriors:{startFrame:0,nFrames:1,k:1,ids:[1],probs:[.9]}}];
    const certificates=[{word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:0,toFrame:2,matchedMargin:1,randomMargin:-1}];
    const result=carryDurationNegativeEvidence(chunks,certificates);
    expect(result[0].posteriors).toBe(chunks[0].posteriors);
    expect(chunks[0].phonemes[0]).not.toHaveProperty("durationNegativeVetoes");
    expect(JSON.parse(JSON.stringify(result))[0].phonemes[0].durationNegativeVetoes).toEqual(certificates);
    expect(result[0].phonemes.map(({durationNegativeVetoes,...p})=>p)).toEqual(chunks[0].phonemes);
  });
});
