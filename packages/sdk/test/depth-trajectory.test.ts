import { describe, expect, it } from "vitest";
import { carryDepthEvidence, depthBoundaryScore, depthComparison, depthObservations, depthReversal, depthVerdictRemovals, type DepthModel } from "../src/depth-trajectory.js";

const eye=Array.from({length:16},(_,i)=>Array.from({length:16},(_,j)=>i===j?1:0));
const arm=(reverse=false)=>({center:Array(16).fill(0),scale:Array(16).fill(1),
  phones:[0,1,2].map(p=>({mean:Array(16).fill(p===0?10:(reverse?3-p:p)),precision:eye,logDet:0,count:10}))});
const model={dimensions:16,tokens:"بت",trajectory:arm(),control:arm(true)} as DepthModel;
const c={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:0,toFrame:4};
describe("depth trajectory evidence",()=> {
  it("conditions out path count for constant evidence, preserving repeated articulations",()=> {
    for (const ids of [[1],[1,2],[1,1]]) expect(depthBoundaryScore(Array.from({length:7},()=>[-.75,-.75,-.75]),ids)).toBeCloseTo(-5.25,12);
    expect(depthBoundaryScore([[0,0,0]],[1,1])).toBe(-Infinity);
  });
  it("requires opposing finite complete-pronunciation preferences",()=> {
    const x=Array.from({length:4},()=>Array(16).fill(1));
    const result=depthComparison(model,x,x,c)!;
    expect(depthReversal(result)).toBe(true);
    expect(depthReversal({...result,controlWritten:result.controlRival})).toBe(false);
    expect(depthReversal({...result,trajectoryWritten:NaN})).toBe(false);
    expect(depthReversal({...result,trajectoryWritten:result.trajectoryRival})).toBe(false);
  });
  it("is antisymmetric when the two complete candidates are exchanged",()=> {
    const x=Array.from({length:4},()=>Array(16).fill(1));
    const a=depthComparison(model,x,x,c)!,b=depthComparison(model,x,x,{...c,written:c.rival,rival:c.written})!;
    expect(a.trajectoryWritten).toBe(b.trajectoryRival);
    expect(a.controlRival).toBe(b.controlWritten);
    expect(depthReversal(b)).toBe(false);
  });
  it("abstains on missing phones, mismatched clocks or nonfinite features",()=> {
    const x=Array.from({length:4},()=>Array(16).fill(1));
    expect(depthComparison(model,x,x.slice(1),c)).toBeNull();
    expect(depthComparison(model,x,x,{...c,rival:"ز"})).toBeNull();
    expect(depthComparison(model,[Array(16).fill(NaN)],x,c)).toBeNull();
  });
  it("retracts only the exact live occurrence removed by the certificate replay",()=> {
    const warning={type:"checker",v:1,id:"one",timeMs:4000,state:"confirmed",errorClass:"wrong_word",
      location:c.word,expected:c.written,observed:c.rival,confidence:1} as import("@tilawa/core").CheckerEvent;
    const repeated={...warning,id:"two",timeMs:8000};
    const laterGraph={...warning,timeMs:9000};
    const insertion={...warning,id:"three",errorClass:"added_word" as const};
    expect(depthVerdictRemovals([warning,repeated,laterGraph,insertion],
      [warning,repeated,laterGraph,insertion],[repeated,laterGraph,insertion])).toEqual([warning]);
    expect(depthVerdictRemovals([warning],[{...warning,timeMs:4001}],[])).toEqual([]);
  });
  it("carries certificates without changing primary phones or posterior rows",()=> {
    const chunks=[{inventory:"test",frameMs:40,phonemes:[{p:"ب",start:0,end:1,conf:.9}],
      posteriors:{startFrame:0,nFrames:2,k:1,ids:[1,1],probs:[.9,.9]}}];
    const certificate={...c,controlWritten:-2,controlRival:-1,trajectoryWritten:1,trajectoryRival:0};
    const enriched=carryDepthEvidence(chunks,[certificate]);
    expect(enriched[0].posteriors).toBe(chunks[0].posteriors);
    expect(enriched[0].phonemes[0]).toEqual({...chunks[0].phonemes[0],depthTrajectoryVetoes:[certificate]});
    expect(chunks[0].phonemes[0]).not.toHaveProperty("depthTrajectoryVetoes");
  });
});


it("keeps depth differences and final-layer observations on the native streaming clock",async()=> {
  let calls=0;
  const manifest={tokens:"بت",fbank:{mels:80,padValue:-23},window:39,shift:32,
    states:[{name:"cache",shape:[1],dtype:"float32"}]} as import("../src/model.js").ModelManifest;
  const outputs=["a","b","c","d"];
  const projection=Array.from({length:4},()=>[1,0]);
  const extractor={name:"frozen",async release(){},async run(feeds:Record<string,import("../src/model.js").Tensor>) {
    expect(Number(feeds.cache.data[0])).toBe(calls);
    const offset=1000*calls++;
    const out:Record<string,import("../src/model.js").Tensor>={
      log_probs:{type:"float32",dims:[1,8,3],data:new Float32Array(24)},
      new_cache:{type:"float32",dims:[1],data:new Float32Array([calls])}};
    outputs.forEach((name,l)=> {
      const n=l===3?8:16;
      out[name]={type:"float32",dims:[n,1,2],data:Float32Array.from({length:n*2},(_,i)=>
        i%2?0:offset+100*l+Math.floor(i/2))};
    });
    return out;
  }};
  const fitted={...model,outputs,projections:outputs.map(()=>projection),controlProjection:Array.from({length:16},()=>[1,0])};
  const result=await depthObservations(extractor,manifest,fitted,new Float32Array(16000));
  expect(result.trajectory).toHaveLength(23);
  expect(result.control).toHaveLength(23);
  expect(result.trajectory[0]).toEqual([...Array(4).fill(.5),...Array(8).fill(100),...Array(4).fill(99.5)]);
  expect(result.control[0]).toEqual(Array(16).fill(300));
  expect(result.control[8][0]).toBe(1300);
  expect(calls).toBe(3);
});
