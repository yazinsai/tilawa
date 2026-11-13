import { describe, expect, it } from "vitest";
import { Fbank } from "../src/fbank.js";
import { MovingZeroObservations, movingZeroFeatures, rootPairs, scoreMovingZeroSequence, movingZeroComparison, movingZeroReversal, validateMovingZeroModel } from "../src/moving-spectral-zero.js";
import { rootFrames, zeroModel, reversalModel } from "./fixtures/moving-zero.js";

describe("moving pole-zero pronunciation evidence",()=> {
  it("preserves known conjugate roots without assigning polynomial root indices",()=> {
    const radius=.8,angle=.3*Math.PI;
    const values=rootPairs([1,-2*radius*Math.cos(angle),radius**2]);
    expect(values[0][0]).toBeCloseTo(.3,9);
    expect(values[0][1]).toBeCloseTo(-Math.log(radius)/Math.PI,9);
    expect(()=>rootPairs([0,1,1])).toThrow();
  });
  it("extracts chunk-independent rational roots and preserves primary fbank",()=> {
    const audio=Float32Array.from({length:1600},(_,i)=>.2*Math.sin(i*.13)+.02*Math.sin(i*.71));
    const features=movingZeroFeatures(audio),stream=new MovingZeroObservations();
    for(let i=0;i<audio.length;i+=137) stream.push(audio.subarray(i,i+137));
    expect(stream.frames).toEqual(features);
    expect(features).toHaveLength(2);expect(features.every(f=>f.valid)).toBe(true);
    const plain=new Fbank().push(audio);
    const observing=new Fbank({onSpectrum:p=>p.fill(0)}).push(audio);
    expect(observing).toEqual(plain);
  });
  it("marginalizes zero correspondence across swaps in root extraction order",()=> {
    const rows=rootFrames(),m=zeroModel();
    const permuted=rows.map((r,i)=>({...r,zeros:i%2?r.zeros.slice().reverse():[r.zeros[1],r.zeros[2],r.zeros[0]]}));
    for(const moving of [false,true]) expect(scoreMovingZeroSequence(m,permuted,"ab",moving)).toBeCloseTo(scoreMovingZeroSequence(m,rows,"ab",moving)!,10);
  });
  it("distinguishes smooth articulation from the same shuffled static zeros",()=> {
    const rows=rootFrames(10),m=zeroModel();
    const shuffled=[0,9,1,8,2,7,3,6,4,5].map(j=>rows[j]);
    // Every state has identical parameters here: static evidence cannot use phone
    // occupancy or the order of frames to answer this trajectory question.
    expect(scoreMovingZeroSequence(m,rows,"a",false)).toBeCloseTo(scoreMovingZeroSequence(m,shuffled,"a",false)!,10);
    expect(scoreMovingZeroSequence(m,rows,"a",true)!).toBeGreaterThan(scoreMovingZeroSequence(m,shuffled,"a",true)!);
  });
  it("requires full pronunciation acceptance and abstains on unavailable evidence",()=> {
    const m=zeroModel(),r=rootFrames();
    expect(scoreMovingZeroSequence(m,r.slice(0,1),"ab",true)).toBeNull();
    expect(scoreMovingZeroSequence(m,r.slice(0,2),"aa",true)).toBeNull();
    expect(scoreMovingZeroSequence(m,r,"z",false)).toBeNull();
    expect(scoreMovingZeroSequence(m,r.map(x=>({...x,valid:false})),"a",true)).toBeNull();
    m.phoneCounts[1]=0;expect(scoreMovingZeroSequence(m,r,"ab",true)).toBeNull();
    expect(()=>validateMovingZeroModel(zeroModel(),"wrong","ab")).toThrow();
    const corrupt=zeroModel();corrupt.moving.alpha[0][0]=1;expect(()=>validateMovingZeroModel(corrupt,"test","ab")).toThrow();
  });
  it("requires finite written preference and strict rival preference in the matched static control",()=> {
    const m=reversalModel(),rows=rootFrames(),c={word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:0,toFrame:6};
    const comparison=movingZeroComparison(m,rows,c)!;
    expect(movingZeroReversal(comparison)).toBe(true);
    expect(movingZeroReversal({...comparison,controlWritten:comparison.controlRival})).toBe(false);
    expect(movingZeroReversal({...comparison,movingWritten:comparison.movingRival})).toBe(false);
    expect(movingZeroReversal({...comparison,movingWritten:NaN})).toBe(false);
    expect(JSON.parse(JSON.stringify(comparison))).toEqual(comparison);
    expect(movingZeroComparison(m,rows,{...c,toFrame:7})).toBeNull();
  });
});
