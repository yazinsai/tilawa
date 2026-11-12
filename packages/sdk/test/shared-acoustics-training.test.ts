import { describe, expect, it } from "vitest";
import { degradedReplicate } from "../src/shared-acoustics-training.js";

describe("independent frozen-encoder training replicates",()=> {
  it("preserves source samples, length and timing while independently degrading the copies",()=> {
    const samples=Float32Array.from({length:16000},(_,i)=>Math.sin(2*Math.PI*440*i/16000));
    const saved=samples.slice(),a=degradedReplicate(samples,1),b=degradedReplicate(samples,2);
    expect(samples).toEqual(saved);
    expect(a).toHaveLength(samples.length);expect(b).toHaveLength(samples.length);
    expect(a).toEqual(degradedReplicate(samples,1));expect(a).not.toEqual(b);
    expect(a.every(Number.isFinite)).toBe(true);expect(b.every(Number.isFinite)).toBe(true);
    const impulse=new Float32Array(2000);impulse[1000]=1;
    const result=degradedReplicate(impulse,1);
    expect(result.indexOf(Math.max(...result))).toBe(1000);
  });
  it("keeps empty and silent recordings silent",()=> {
    expect(degradedReplicate(new Float32Array(),1)).toHaveLength(0);
    expect(degradedReplicate(new Float32Array(100),1)).toEqual(new Float32Array(100));
  });
});
