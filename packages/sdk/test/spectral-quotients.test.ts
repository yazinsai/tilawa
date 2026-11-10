import { describe, expect, it } from "vitest";
import { Fbank } from "../src/fbank.js";
import { QuotientObservations, quotientFeatures, quotientComparison, quotientReversal, quotientUnits,
  quotientGaussian, quotientLogadd, scoreQuotientSequence, validateQuotientModel } from "../src/spectral-quotients.js";
import type { QuotientModel } from "../src/spectral-quotients.js";

const row=(v:number) => Array.from({length:32},(_,d) => d ? 0 : v);
const emission=(values:number[]) => ({mean:values.flatMap(row),variance:Array(values.length*32).fill(1)});
const model=():QuotientModel => ({schema:1,inventory:"test",tokens:"بتَ",frameMs:40,dimensions:32,
  scale:Array(32).fill(1),phoneCounts:[10,10,20],maxPart:2,maxFrames:1024,
  quotient:{"بَ":emission([0]),"تَ":emission([-2])},
  absolute:{"بَ":emission([5]),"تَ":emission([0])},
});

describe("local consonant / vowel spectral quotients",()=> {
  it("requires the written quotient preference to reverse a rival-preferring absolute control",()=> {
    const c={word:{surah:1,ayah:1,word:1},written:"بَ",rival:"تَ",fromFrame:0,toFrame:2};
    const comparison=quotientComparison(model(),[row(0),row(0)],c)!;
    expect(quotientReversal(comparison)).toBe(true);
    expect(quotientReversal({...comparison,controlWritten:comparison.controlRival})).toBe(false);
    expect(quotientReversal({...comparison,quotientWritten:NaN})).toBe(false);
    expect(quotientReversal(quotientComparison(model(),[row(0),row(2)],c)!)).toBe(false);
  });
  it("cancels arbitrary stationary per-frequency log response in both latent intervals",()=> {
    const raw=[row(0),row(.5),row(1),row(2)];
    const coloured=raw.map(r => r.map((x,d) => x+Math.sin(d)*8));
    for (const p of ["بَ","تَ"]) expect(scoreQuotientSequence(model(),coloured,p,true))
      .toBeCloseTo(scoreQuotientSequence(model(),raw,p,true)!,10);
  });
  it("matches exhaustive summation over every unit boundary and internal CV cut",()=> {
    const m=model(), raw=[0,.5,1,2,2,1].map(row), us=["بَ","تَ"];
    const mean=(a:number,b:number) => row(raw.slice(a,b).reduce((s,r) => s+r[0],0)/(b-a));
    for (const relative of [false,true]) {
      let mass=-Infinity,prior=-Infinity;
      const walk=(i:number,a:number,acoustic:number,timing:number):void => {
        if (i===us.length) { if (a===raw.length) { mass=quotientLogadd(mass,acoustic+timing);prior=quotientLogadd(prior,timing); } return; }
        for (let b=a+2;b<=Math.min(a+4,raw.length);b++) {
          const lo=Math.max(a+1,b-2),hi=Math.min(b-1,a+2);
          for (let c=lo;c<=hi;c++) {
            const left=mean(a,c),right=mean(c,b),x=relative ? left.map((v,d) => v-right[d]) : left;
            walk(i+1,b,acoustic+quotientGaussian(x,(relative?m.quotient:m.absolute)[us[i]]),timing-Math.log(3)-Math.log(hi-lo+1));
          }
        }
      };
      walk(0,0,0,0);
      expect(scoreQuotientSequence(m,raw,"بَتَ",relative)).toBeCloseTo(mass-prior,10);
    }
  });
  it("keeps every phone and abstains on unavailable coverage or impossible complete placements",()=> {
    expect(quotientUnits("ببََتَ" )).toEqual(["ب","بَ","َ","تَ"]);
    expect(scoreQuotientSequence(model(),[row(0),row(1),row(0)],"بَبَ",true)).toBeNull();
    expect(scoreQuotientSequence(model(),[row(0),row(1)],"ب",true)).toBeNull();
    expect(scoreQuotientSequence(model(),[row(0),row(1)],"دَ",true)).toBeNull();
    const m=model();m.phoneCounts[0]=0;
    expect(scoreQuotientSequence(m,[row(0),row(1)],"بَ",true)).toBeNull();
  });
  it("preserves the native fbank and makes spectral observations independent of push sizes",()=> {
    const wave=Float32Array.from({length:5000},(_,i) => .2*Math.sin(i*.08)+.1*Math.cos(i*.15));
    const native=new Fbank().push(wave), powers:Float64Array[]=[];
    expect(new Fbank({onSpectrum:p => powers.push(p)}).push(wave)).toEqual(native);
    const streamed=new QuotientObservations();
    for (let i=0;i<wave.length;i+=137) streamed.push(wave.slice(i,i+137));
    const features=quotientFeatures(wave);
    expect(streamed.frames).toEqual(features);
    expect(features.length).toBe(Math.floor(native.length/4));
    expect(features[0][0]).toBe(powers.slice(0,4).reduce((s,p) => s+Math.log(Math.max(p[4],1e-12)),0)/4);
  });
  it("rejects mismatched clocks, corrupt emission shapes and invalid variances",()=> {
    const m=model(); expect(validateQuotientModel(m,"test","بتَ")).toBe(m);
    expect(()=>validateQuotientModel(m,"other","بتَ")).toThrow();
    m.absolute["بَ"].variance[0]=0;
    expect(()=>validateQuotientModel(m,"test","بتَ")).toThrow();
  });
});
