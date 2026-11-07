import { describe, expect, it } from "vitest";
import { segmentPartition, segmentTextFeatures } from "../src/segment-presence.js";

describe("weak-label latent segmentation", () => {
  it("sums all nonempty monotone segmentations exactly once", () => {
    const e=[[.2,-.1,.5,.3],[-.3,.1,.4,-.2]], stay=[Math.log(.6),Math.log(.7)], exit=[Math.log(.4),Math.log(.3)];
    const paths=[1,2,3].map(b => e[0].slice(0,b).reduce((s,v)=>s+v,0)+e[1].slice(b).reduce((s,v)=>s+v,0)+
      (b-1)*stay[0]+(3-b)*stay[1]+exit[0]+exit[1]);
    expect(segmentPartition(e,stay,exit,-1)).toBeCloseTo(Math.log(paths.reduce((s,v)=>s+Math.exp(v),0)),12);
  });
  it("requires a zero-frame scripted omission and consumes every other frame", () => {
    const e=[[.1,.5,.2],[-.2,.3,-.1]], stay=[Math.log(.6),Math.log(.7)], exit=[Math.log(.4),Math.log(.3)];
    for (const omitted of [0,1]) {
      const other=1-omitted;
      const expected=e[other].reduce((s,v)=>s+v,0)+2*stay[other]+exit[other]-.8-Math.log(2);
      expect(segmentPartition(e,stay,exit,-.8,omitted)).toBeCloseTo(expected,12);
    }
  });
  it("cannot read more words than frames", () => {
    expect(segmentPartition([[0,0],[0,0],[0,0]],[-.5,-.5,-.5],[-.9,-.9,-.9],0)).toBe(-Infinity);
  });
  it("describes unseen words by phones and preserves boundary and bigram order", () => {
    const q=segmentTextFeatures(["abab","baba"],"ab");
    expect(q[0].slice(0,2)).toEqual(q[1].slice(0,2));
    expect(q[0]).not.toEqual(q[1]);
    expect(()=>segmentTextFeatures([""],"ab")).toThrow("Empty word");
  });
});
