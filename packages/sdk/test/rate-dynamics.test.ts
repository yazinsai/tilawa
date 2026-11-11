import { describe, expect, it } from "vitest";
import { logadd, rateComparison, rateFeatures, rateReversal, RateObservations, scoreRateSequence, validateRateModel } from "../src/rate-dynamics.js";
import type { RateModel } from "../src/rate-dynamics.js";

const model: RateModel = {
  schema:1,inventory:"test",tokens:"ab",frameMs:40,observationMs:10,dimensions:20,maxFrames:8192,
  mean:Array(20).fill(0),scale:Array(20).fill(1),innovation:Array(20).fill(1),
  dynamic:{targets:[Array(20).fill(0),Array(20).fill(1),Array(20).fill(-1)],counts:[0,1,1],shared:Array(20).fill(.6)},
  control:{targets:[Array(20).fill(0),Array(20).fill(1),Array(20).fill(-1)],counts:[0,1,1],shared:Array(20).fill(.6)},
};
const rows=[.2,.5,.8,-.1,-.4].map(v=>Array(20).fill(v));
const candidate={word:{surah:1,ayah:1,word:1},written:"ab",rival:"aa",fromFrame:0,toFrame:5};

describe("spectral target dynamics",()=>{
  it("matches exhaustive complete segmentations for both models",()=>{
    for (const dynamic of [false,true]) {
      const p=dynamic ? model.dynamic : model.control;
      const density=(t:number,s:number)=> {
        const rho=dynamic ? p.shared[0] : 0;
        const mean=rho*(t ? rows[t-1][0] : 0)+(1-rho)*p.targets[s+1][0];
        const variance=dynamic ? 1 : Math.exp(p.shared[0]);
        return -10*(Math.log(2*Math.PI*variance)+(rows[t][0]-mean)**2/variance);
      };
      const paths=[1,2,3,4].map(cut=>rows.reduce((sum,_,t)=>sum+density(t,t<cut?0:1),0));
      const expected=paths.reduce(logadd,-Infinity)-Math.log(paths.length);
      expect(scoreRateSequence(model,rows,"ab",dynamic)).toBeCloseTo(expected,10);
    }
  });
  it("carries the observed state across the interval boundary only in the dynamics arm",()=>{
    expect(scoreRateSequence(model,rows,"ab",false,Array(20).fill(1))).toBe(scoreRateSequence(model,rows,"ab",false));
    expect(scoreRateSequence(model,rows,"ab",true,Array(20).fill(1))).not.toBe(scoreRateSequence(model,rows,"ab",true));
  });
  it("recovers a short undershot target relative to an independent realization",()=>{
    const targets=[Array(20).fill(0),Array(20).fill(1),Array(20).fill(.4)];
    const m={...model,dynamic:{...model.dynamic,targets},control:{...model.control,targets}};
    const c={...candidate,written:"a",rival:"b",toFrame:1};
    expect(rateReversal(rateComparison(m,[Array(20).fill(.4)],c)!)).toBe(true);
    // A longer rendition reaches the written target; independent acoustics now
    // prefer the written word too, so the reversal-only certificate abstains.
    const long=Array.from({length:16},(_,t)=>Array(20).fill(1-.6**(t+1)));
    expect(rateReversal(rateComparison(m,long,{...c,toFrame:16})!)).toBe(false);
  });
  it("requires complete repeated pronunciations and abstains on unsupported observations",()=>{
    expect(scoreRateSequence(model,rows.slice(0,1),"aa",true)).toBeNull();
    expect(scoreRateSequence(model,rows,"c",true)).toBeNull();
    expect(scoreRateSequence(model,[Array(20).fill(NaN)],"a",true)).toBeNull();
    expect(scoreRateSequence({...model,dynamic:{...model.dynamic,counts:[0,0,1]}},rows,"ab",true)).toBeNull();
  });
  it("issues a certificate only for a strict finite preference reversal",()=>{
    expect(rateComparison(model,rows,candidate)).toMatchObject(candidate);
    const c={...candidate,controlWritten:-2,controlRival:-1,dynamicWritten:-1,dynamicRival:-2};
    expect(rateReversal(c)).toBe(true);
    expect(rateReversal({...c,dynamicWritten:c.dynamicRival})).toBe(false);
    expect(rateReversal({...c,controlWritten:NaN})).toBe(false);
  });
  it("rejects unstable transitions, bad inventories and invalid covariance",()=>{
    expect(validateRateModel(model,"test","ab")).toBe(model);
    expect(()=>validateRateModel(model,"wrong","ab")).toThrow();
    expect(()=>validateRateModel({...model,innovation:Array(20).fill(0)},"test","ab")).toThrow();
    expect(()=>validateRateModel({...model,dynamic:{...model.dynamic,shared:Array(20).fill(1)}},"test","ab")).toThrow();
  });
  it("extracts the same native spectral observations for any audio push size",()=>{
    const wave=Float32Array.from({length:3500},(_,i)=>.2*Math.sin(i*.08));
    const observation=new RateObservations();
    for (let i=0;i<wave.length;i+=137) observation.push(wave.slice(i,i+137));
    expect(observation.frames).toEqual(rateFeatures(wave));
    for (const row of observation.frames) expect(row.reduce((sum,v)=>sum+Math.exp(v),0)).toBeCloseTo(1,10);
  });
});
