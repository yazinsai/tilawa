import { describe, expect, it } from "vitest";
import { consistentDiphones, diphoneBoundaryScore, diphoneLabels, featureBoundaryScore, singlePhoneRows } from "../src/diphone.js";

describe("overlapping articulation evidence",()=>{
  it("requires the right articulation of one pair to be the left of the next",()=>{
    const labels=diphoneLabels([1,2,3],3);
    expect(labels).toEqual([2,6]);
    expect(consistentDiphones(labels,3)).toBe(true);
    const lp=Array.from({length:4},()=>Array(10).fill(Math.log(.1)));
    expect(Number.isFinite(diphoneBoundaryScore(lp,labels,3))).toBe(true);
    expect(diphoneBoundaryScore(lp,[2,9],3)).toBe(-Infinity);
    expect(diphoneBoundaryScore(lp,[],3)).toBe(-Infinity);
  });
  it("marginalizes all complete placements without rewarding pronunciation length",()=>{
    const lp=Array.from({length:6},()=>Array(10).fill(Math.log(.1)));
    expect(featureBoundaryScore(lp,[2])).toBeCloseTo(6*Math.log(.1),10);
    expect(featureBoundaryScore(lp,[2,6])).toBeCloseTo(6*Math.log(.1),10);
    expect(featureBoundaryScore(lp,[2,6,7])).toBeCloseTo(6*Math.log(.1),10);
    expect(featureBoundaryScore(lp.slice(0,1),[2,6])).toBe(-Infinity);
  });
  it("the matched control ignores partner identity and retains total probability",()=>{
    const rows=[[.1,.2,.3,.15,.25].map(Math.log)];
    const control=singlePhoneRows(rows,2)[0].map(Math.exp);
    expect(control[0]).toBeCloseTo(.1);
    expect(control[1]).toBeCloseTo(.5);
    expect(control[2]).toBeCloseTo(.4);
    const swapped=[[.1,.3,.2,.25,.15].map(Math.log)];
    singlePhoneRows(swapped,2)[0].forEach((v,i)=>expect(v).toBeCloseTo(Math.log(control[i])));
  });
});
