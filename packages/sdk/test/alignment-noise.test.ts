import { describe, expect, it } from "vitest";
import { AlignmentObservations, alignmentFeatures, alignmentScores, alignmentReversal, boundaryLogP, validateAlignmentNoiseModel, type AlignmentNoiseModel } from "../src/alignment-noise.js";

function model(): AlignmentNoiseModel {
  const phone = (mean: number) => ({ mean:Array(8).fill(mean),variance:Array(8).fill(1),count:10 });
  return { schema:1,inventory:"test",tokens:"ab",frameMs:40,observationMs:10,dimensions:8,
    center:Array(8).fill(0),scale:Array(8).fill(1),control:[phone(0),phone(3),phone(0)],
    noise:[phone(0),phone(0),phone(3)],contamination:[0,.1,.2],stay:.8,maxFrames:2048 };
}
describe("alignment-label noise recheck", () => {
  it("integrates complete phone placements with a normalized timing prior", () => {
    expect(boundaryLogP(Array.from({length:4},()=>[0,0]),.8)).toBeCloseTo(0,12);
    expect(boundaryLogP([[0,0]],.8)).toBe(-Infinity);
    expect(boundaryLogP([[0,0],[0,0]],.8)).toBeCloseTo(0,12);
  });
  it("extracts identical observations across arbitrary PCM push boundaries", () => {
    const audio=Float32Array.from({length:4000},(_,i)=>.2*Math.sin(i*.071));
    const batched=alignmentFeatures(audio), streaming=new AlignmentObservations();
    for(let i=0;i<audio.length;i+=137) streaming.push(audio.slice(i,i+137));
    expect(streaming.frames).toEqual(batched);
  });
  it("requires a strict matched-control reversal and scores candidate order symmetrically", () => {
    const m=model(), x=Array.from({length:5},()=>Array(8).fill(0));
    const scores=alignmentScores(m,x,["a","b"])!;
    expect(scores.control[0]).toBeLessThan(scores.control[1]);
    expect(scores.noise[0]).toBeGreaterThan(scores.noise[1]);
    const reversed=alignmentScores(m,x,["b","a"])!;
    expect(reversed.control).toEqual(scores.control.slice().reverse());
    expect(reversed.noise).toEqual(scores.noise.slice().reverse());
    const c={word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:0,toFrame:1,
      controlWritten:scores.control[0],controlRival:scores.control[1],noiseWritten:scores.noise[0],noiseRival:scores.noise[1]};
    expect(alignmentReversal(c)).toBe(true);
    expect(alignmentReversal({...c,noiseWritten:c.noiseRival})).toBe(false);
    expect(alignmentReversal({...c,controlWritten:NaN})).toBe(false);
  });
  it("abstains for unknown, untrained, incomplete or unavailable observations", () => {
    const m=model(),x=[Array(8).fill(0)];
    expect(alignmentScores(m,x,["c"])).toBeNull();
    expect(alignmentScores(m,x,["ab"])).toBeNull();
    expect(alignmentScores(m,[],["a"])).toBeNull();
    m.noise[1].count=0;
    expect(alignmentScores(m,x,["a"])).toBeNull();
  });
  it("rejects mismatched inventory and nonfinite or nonpositive emission parameters", () => {
    expect(validateAlignmentNoiseModel(model(),"test","ab")).toEqual(model());
    expect(()=>validateAlignmentNoiseModel(model(),"other","ab")).toThrow();
    const m=model();m.noise[1].variance[0]=0;
    expect(()=>validateAlignmentNoiseModel(m,"test","ab")).toThrow();
  });
});
