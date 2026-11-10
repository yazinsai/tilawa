import { describe, expect, it } from "vitest";
import { AmFmObservations, amFmFeatures, amFmEmission, amFmComparison, amFmReversal, logadd, scoreAmFmSequence, validateAmFmModel } from "../src/am-fm.js";
import type { AmFmModel } from "../src/am-fm.js";

const model=():AmFmModel=>({schema:1,inventory:"test",tokens:"ab",frameMs:40,observationMs:10,bands:[0],center:[[0,0]],scale:[[1,1]],
  means:[[0,0],[0,0],[0,0]],variances:[[1,1],[1,1],[.8,.8]],covariance:[[0],[.9],[-.7]],counts:[0,10,10],stay:.7,maxFrames:4096});
const row=(a:number,f:number,q=1)=>[a,f,q,...Array(21).fill(0)];
const candidate={word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:0,toFrame:1};

describe("joint amplitude and instantaneous-frequency evidence",()=> {
  it("uses coupling beyond the independent marginals and retains a genuine rival",()=> {
    const forward=amFmComparison(model(),[row(.4,.4),row(-.4,-.4)],candidate)!;
    const scrambled=amFmComparison(model(),[row(.4,-.4),row(-.4,.4)],candidate)!;
    expect(amFmReversal(forward)).toBe(true);
    expect(scrambled.controlWritten).toBeCloseTo(forward.controlWritten,12);
    expect(scrambled.controlRival).toBeCloseTo(forward.controlRival,12);
    expect(scrambled.jointRival).toBeGreaterThan(scrambled.jointWritten);
    expect(amFmReversal(scrambled)).toBe(false);
  });
  it("marginalizes frequency exactly at unreliable amplitude and mixes partial reliability",()=> {
    const m=model(), a=row(.4,1000,0), b=row(.4,-999,0);
    expect(amFmEmission(m,a,1,true)).toBe(amFmEmission(m,b,1,true));
    expect(amFmEmission(m,a,1,true)).toBe(amFmEmission(m,a,1,false));
    const partial=amFmEmission(m,row(.4,.4,.25),1,true);
    const expected=logadd(Math.log(.25)+amFmEmission(m,row(.4,.4,1),1,true),Math.log(.75)+amFmEmission(m,a,1,true));
    expect(partial).toBeCloseTo(expected,12);
  });
  it("agrees with exhaustive marginalization over complete phone boundaries",()=> {
    const m=model(),raw=[row(.1,.2),row(.3,.1),row(-.2,.5),row(.4,-.3)];
    let mass=-Infinity,prior=-Infinity;
    for (let boundary=1;boundary<raw.length;boundary++) {
      const timing=(raw.length-2)*Math.log(m.stay)+Math.log1p(-m.stay);
      const acoustic=raw.reduce((v,r,t)=>v+amFmEmission(m,r,t<boundary?1:2,true),0);
      mass=logadd(mass,timing+acoustic);prior=logadd(prior,timing);
    }
    expect(scoreAmFmSequence(m,raw,"ab",true)).toBeCloseTo(mass-prior,12);
  });
  it("requires every phone, treats repetitions as separate articulations and abstains on missing support",()=> {
    const m=model();
    expect(scoreAmFmSequence(m,[row(0,0)],"aa",true)).toBeNull();
    expect(scoreAmFmSequence(m,[row(0,0),row(0,0)],"aa",true)).not.toBeNull();
    expect(scoreAmFmSequence(m,[row(0,0)],"c",true)).toBeNull();
    expect(scoreAmFmSequence(m,[row(NaN,0)],"a",true)).toBeNull();
    expect(scoreAmFmSequence(m,[row(0,0,-1)],"a",true)).toBeNull();
    m.counts[1]=0;expect(scoreAmFmSequence(m,[row(0,0)],"a",true)).toBeNull();
  });
  it("preserves filter state and the snipped 25 ms / 10 ms clock under arbitrary audio chunks",()=> {
    const wave=Float32Array.from({length:5000},(_,i)=>.2*Math.sin(i*.08)+.1*Math.cos(i*.15));
    const full=amFmFeatures(wave), streamed=new AmFmObservations();
    for(let i=0;i<wave.length;i+=137)streamed.push(wave.subarray(i,i+137));
    expect(streamed.frames).toEqual(full);
    expect(full.length).toBe(Math.floor((wave.length-400)/160)+1);
    expect(full.flat().every(Number.isFinite)).toBe(true);
    expect(full.every(r=>Array.from({length:8},(_,b)=>r[b*3+2]).every(q=>q>=0&&q<=1))).toBe(true);
    expect(amFmFeatures(new Float32Array(800)).every(r=>r.every(v=>v===0))).toBe(true);
  });
  it("checks positive definite covariance, inventory and model dimensions",()=> {
    const m=model();expect(validateAmFmModel(m,"test","ab")).toBe(m);
    expect(()=>validateAmFmModel(m,"other","ab")).toThrow();
    m.covariance[1][0]=1;expect(()=>validateAmFmModel(m,"test","ab")).toThrow();
  });
});
