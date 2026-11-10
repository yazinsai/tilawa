import { describe, expect, it } from "vitest";
import { Fbank } from "../src/fbank.js";
import { RidgeObservations, ridgeFeatures, ridgePatch, ridgeComparison, ridgeReversal, ridgeLogadd, scoreRidgeSequence, validateRidgeModel } from "../src/spectral-ridge.js";
import type { RidgeModel } from "../src/spectral-ridge.js";

const model=():RidgeModel=>({schema:1,inventory:"test",tokens:"ab",frameMs:40,observationMs:10,dimensions:56,controlDimensions:32,
  radius:1,center:Array(56).fill(0),scale:Array(56).fill(1),counts:[0,10,10],stay:.8,maxFrames:2048,
  means:Array.from({length:3},(_,p)=>Array.from({length:56},(_,d)=>d===0?(p===1?1:0):d===32?(p===1?2:0):0)),
  variances:Array.from({length:3},()=>Array(56).fill(1))});
const row=(energy:number,connection:number)=>Array.from({length:56},(_,d)=>d===0?energy:d===32?connection:0);
const spectrum=(peaks:number[])=>Array.from({length:64},(_,i)=>peaks.includes(i)?4:-1);

describe("spectral ridge connectivity evidence",()=> {
  it("retains identical local ridge locations and energies while distinguishing interrupted connections",()=> {
    const a=spectrum([8]),b=spectrum([40]);
    const continuous=ridgePatch([a,a,a],1,1),interrupted=ridgePatch([b,a,b],1,1);
    expect(continuous.slice(0,32)).toEqual(interrupted.slice(0,32));
    expect(continuous[35]).toBeGreaterThan(interrupted[35]);
    expect(interrupted[37]).toBeGreaterThan(continuous[37]);
  });
  it("distinguishes a ridge that branches across time from an uninterrupted single ridge",()=> {
    const broad=spectrum([8,9,10]),split=spectrum([8,10]);
    const straight=ridgePatch([broad,broad,broad],1,1),branching=ridgePatch([split,broad,split],1,1);
    expect(straight.slice(0,32)).toEqual(branching.slice(0,32));
    expect(branching[36]).toBeGreaterThan(straight[36]);
  });
  it("requires the matched control to prefer the rival and connectivity to prefer the written sequence",()=> {
    const c={word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:0,toFrame:1};
    const certificate=ridgeComparison(model(),[row(0,2),row(0,2)],c)!;
    expect(ridgeReversal(certificate)).toBe(true);
    expect(ridgeReversal({...certificate,controlWritten:certificate.controlRival})).toBe(false);
    expect(ridgeReversal(ridgeComparison(model(),[row(0,0),row(0,0)],c)!)).toBe(false);
  });
  it("equals exhaustive segmentation with a normalized completion prior",()=> {
    const m=model(),raw=[row(1,2),row(.5,1),row(0,0)];
    let mass=-Infinity,prior=-Infinity;
    for (const path of [[0,0,1],[0,1,1]]) {
      let p=0,acoustic=0;
      for (let t=0;t<path.length;t++) {
        if (t) p+=path[t]===path[t-1]?Math.log(m.stay):Math.log1p(-m.stay);
        acoustic+=raw[t].reduce((v,x,d)=>v-.5*(Math.log(2*Math.PI)+(x-m.means[path[t]+1][d])**2),0);
      } mass=ridgeLogadd(mass,p+acoustic);prior=ridgeLogadd(prior,p);
    }
    expect(scoreRidgeSequence(m,raw,"ab",true)).toBeCloseTo(mass-prior,10);
  });
  it("requires every phone including repeats, and abstains on unsupported or incomplete input",()=> {
    expect(scoreRidgeSequence(model(),[row(0,2)],"aa",true)).toBeNull();
    expect(scoreRidgeSequence(model(),[row(0,2),row(0,2)],"aa",true)).not.toBeNull();
    expect(scoreRidgeSequence(model(),[row(0,2)],"c",true)).toBeNull();
    expect(scoreRidgeSequence(model(),[row(NaN,2)],"a",true)).toBeNull();
    const sparse=model();sparse.counts[1]=0;
    expect(scoreRidgeSequence(sparse,[row(0,2)],"a",true)).toBeNull();
  });
  it("preserves primary frontend output and observes the same geometry under irregular audio pushes",()=> {
    const wave=Float32Array.from({length:5000},(_,i)=>.2*Math.sin(i*.08)+.1*Math.cos(i*.15));
    const native=new Fbank().push(wave),power:Float64Array[]=[];
    expect(new Fbank({spectrum:p=>power.push(p)}).push(wave)).toEqual(native);
    expect(power.length).toBe(native.length);
    for (const radius of [1,3] as const) {
      const streamed=new RidgeObservations(radius);
      for (let i=0;i<wave.length;i+=137) streamed.push(wave.slice(i,i+137));
      streamed.end();
      expect(streamed.frames).toEqual(ridgeFeatures(wave,radius));
      expect(streamed.frames.length).toBe(native.length);
    }
  });
  it("validates all model shapes, geometry, clocks and variances",()=> {
    const m=model();expect(validateRidgeModel(m,"test","ab")).toBe(m);
    expect(()=>validateRidgeModel(m,"other","ab")).toThrow();
    expect(()=>validateRidgeModel({...m,stay:1},"test","ab")).toThrow();
    m.variances[1][0]=0;
    expect(()=>validateRidgeModel(m,"test","ab")).toThrow();
  });
});
