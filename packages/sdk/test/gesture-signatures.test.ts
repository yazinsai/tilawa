import { describe, expect, it } from "vitest";
import { GestureObservations, gestureDimensions, gestureFeatures, pathSignature, timeTrajectory,
  scoreGestureSequence, validateGestureModel, type GestureModel } from "../src/gesture-signatures.js";

const near=(a:number[],b:number[])=>a.forEach((v,i)=>expect(v).toBeCloseTo(b[i],10));
const path=[[0,0],[1,0],[1,1]];
const model=():GestureModel=>({schema:1,inventory:"test",tokens:"ab",frameMs:40,bands:2,order:2,
  center:[0,0],scale:[1,1],maxSegment:4,maxFrames:2048,counts:[1,1,1],
  signature:{means:Array.from({length:3},()=>Array(8).fill(0)),variances:Array.from({length:3},()=>Array(8).fill(1))},
  control:{means:Array.from({length:3},()=>Array(8).fill(0)),variances:Array.from({length:3},()=>Array(8).fill(1))}});

describe("tempo-invariant acoustic path signatures",()=>{
  it("computes ordered cross-band iterated integrals",()=>{
    near(pathSignature(path,2),[0,0,1,1,.5,1,0,.5]);
    expect(pathSignature([[0,0],[0,1],[1,1]],2)).not.toEqual(pathSignature(path,2));
  });
  it("preserves all levels under monotone changes in traversal speed",()=>{
    const slower=[[0,0],[.1,0],[.3,0],[1,0],[1,1]];
    for(const order of [2,3]) {
      near(pathSignature(path,order),pathSignature(slower,order));
      expect(timeTrajectory(path,order)).not.toEqual(timeTrajectory(slower,order));
      expect(timeTrajectory(path,order).length).toBe(gestureDimensions(2,order));
    }
  });
  it("counts a closed loop's signed area and keeps stationary paths finite",()=>{
    near(pathSignature([[0,0],[1,0],[1,1],[0,1],[0,0]],2),[0,0,0,0,0,1,-1,0]);
    near(pathSignature([[2,3],[2,3]],3),[2,3,...Array(14).fill(0)]);
  });
  it("marginalizes complete phone boundaries with optional silence",()=>{
    const m=model(), raw=Array.from({length:3},()=>Array(80).fill(0));
    // Every segment has the same Gaussian log density, but longer segmentations have more factors.
    const density=-4*Math.log(2*Math.PI);
    const partitions:number[]=[];
    for(let a=0;a<=3;a++)for(let b=1;b<=3;b++)for(let c=0;c<=3;c++) if(a+b+c===3)
      partitions.push(density*(1+Number(a>0)+Number(c>0)));
    const expected=Math.log(partitions.reduce((s,p)=>s+Math.exp(p),0)/partitions.length);
    expect(scoreGestureSequence(m,raw,"a",true)).toBeCloseTo(expected,10);
    expect(scoreGestureSequence(m,raw,"z",true)).toBeNull();
    expect(scoreGestureSequence(m,raw,"aaaa",true)).toBeNull();
  });
  it("matches the native observation clock for arbitrary input pushes",()=>{
    const audio=Float32Array.from({length:2400},(_,i)=>.2*Math.sin(i/8));
    const o=new GestureObservations();
    for(let a=0;a<audio.length;a+=113)o.push(audio.subarray(a,a+113));
    expect(o.frames).toEqual(gestureFeatures(audio));
  });
  it("rejects incompatible or incomplete declared assets",()=>{
    expect(validateGestureModel(model(),"test","ab")).toEqual(model());
    expect(()=>validateGestureModel({...model(),order:4},"test","ab")).toThrow();
    const broken=model(); broken.control.variances[0][1]=0;
    expect(()=>validateGestureModel(broken,"test","ab")).toThrow();
    expect(()=>validateGestureModel(model(),"other","ab")).toThrow();
  });
});
