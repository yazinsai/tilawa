import { describe, expect, it } from "vitest";
import { Fbank } from "../src/fbank.js";
import { ReverseObservations, reverseFeatures, reverseComparison, reverseReversal, predictiveLogadd, scorePredictiveSequence, validateReverseModel } from "../src/reverse-predictive.js";
import type { ReverseModel } from "../src/reverse-predictive.js";

const row=(v: number)=>Array.from({length:80},(_,d)=>d ? 0 : v);
const model = (): ReverseModel => {
  const arm = () => ({intercept:[row(0),row(0),row(1)],coefficient:[row(0),row(.5),row(.5)],
    variance:[row(1).map(()=>1),row(1).map(()=>1),row(1).map(()=>1)],counts:[10,10,10]});
  return {schema:1,inventory:"test",tokens:"ab",frameMs:40,observationMs:10,dimensions:80,
    center:row(0),scale:row(1).map(()=>1),forward:arm(),reverse:arm(),maxFrames:8192};
};

describe("matched reverse-predictive acoustic evidence",()=> {
  it("reverses acoustic observations and the complete pronunciation together",()=> {
    const m=model(), x=[0,.3,1,1.8,2,1].map(row);
    expect(scorePredictiveSequence(m,x,"ab",true)).toBeCloseTo(scorePredictiveSequence(m,x.slice().reverse(),"ba",false)!,10);
    expect(scorePredictiveSequence(m,x,"ab",true)).not.toBeCloseTo(scorePredictiveSequence(m,x,"ab",false)!,5);
  });
  it("equals exhaustive complete-path integration including backgrounds and adjacent repeated phones",()=> {
    const m=model(), raw=[0,1,.5,1,2].map(row);
    for (const text of ["ab","aa"]) {
      const labels=[0,...[...text].flatMap(p=>[m.tokens.indexOf(p)+1,0])];
      let mass=-Infinity, prior=-Infinity;
      const walk=(path: number[],p: number) => {
        if (path.length===raw.length) {
          if (path.at(-1)!<labels.length-2) return;
          let acoustic=0;
          for (let t=0;t<path.length;t++) {
            const id=labels[path[t]];
            acoustic+=raw[t].reduce((v,x,d)=> {
              const variance=m.forward.variance[id][d];
              const mean=m.forward.intercept[id][d]+m.forward.coefficient[id][d]*(t ? raw[t-1][d] : 0);
              return v-.5*(Math.log(2*Math.PI*variance)+(x-mean)**2/variance);
            },0);
          }
          mass=predictiveLogadd(mass,p+acoustic);prior=predictiveLogadd(prior,p);return;
        }
        const s=path.at(-1)!;
        walk([...path,s],p+Math.log(.8));
        const canSkip=s+2<labels.length && labels[s]!==0 && labels[s]!==labels[s+2];
        if (s+1<labels.length) walk([...path,s+1],p+Math.log(canSkip ? .1 : .2));
        if (canSkip) walk([...path,s+2],p+Math.log(.1));
      };
      walk([0],-Math.log(2));walk([1],-Math.log(2));
      expect(scorePredictiveSequence(m,raw,text,false)).toBeCloseTo(mass-prior,9);
    }
  });
  it("requires finite strict directional disagreement rather than either arm's uncertainty",()=> {
    const c={word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:0,toFrame:1};
    const result={...c,forwardWritten:0,forwardRival:1,reverseWritten:1,reverseRival:0};
    expect(reverseReversal(result)).toBe(true);
    expect(reverseReversal({...result,forwardWritten:1})).toBe(false);
    expect(reverseReversal({...result,reverseWritten:0})).toBe(false);
    expect(reverseReversal({...result,reverseWritten:NaN})).toBe(false);
    expect(reverseComparison(model(),[row(0),row(1)],{...c,written:"unknown"})).toBeNull();
  });
  it("abstains on impossible, missing or unsupported complete pronunciations",()=> {
    expect(scorePredictiveSequence(model(),[row(0)],"ab",true)).toBeNull();
    expect(scorePredictiveSequence(model(),[row(0),row(1)],"aa",true)).toBeNull();
    expect(scorePredictiveSequence(model(),[row(NaN)],"a",true)).toBeNull();
    const sparse=model();sparse.reverse.counts[1]=0;
    expect(scorePredictiveSequence(sparse,[row(0)],"a",true)).toBeNull();
  });
  it("uses the native frontend identically under different push sizes",()=> {
    const wave=Float32Array.from({length:5000},(_,i)=>.2*Math.sin(i*.08)+.1*Math.cos(i*.15));
    const native=new Fbank().push(wave).map(r=>Array.from(r));
    const streamed=new ReverseObservations();
    for (let i=0;i<wave.length;i+=137) streamed.push(wave.slice(i,i+137));
    expect(reverseFeatures(wave)).toEqual(native);
    expect(streamed.frames).toEqual(native);
  });
  it("validates both matched predictors and the bundle inventory",()=> {
    const m=model();
    expect(validateReverseModel(m,"test","ab")).toBe(m);
    expect(()=>validateReverseModel(m,"other","ab")).toThrow();
    m.reverse.variance[1][0]=0;
    expect(()=>validateReverseModel(m,"test","ab")).toThrow();
  });
});
