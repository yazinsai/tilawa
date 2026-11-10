import { describe, expect, it } from "vitest";
import { Fbank } from "../src/fbank.js";
import { boundaryLogP, boundaryWithNoise, logI0Scaled, logPowerControl, marginalEmission, noisyPowerLogP, PowerObservations, powerComparison, POWER_BINS, validateNoisePowerModel } from "../src/noise-power.js";

describe("explicit noisy FFT-power likelihood",()=> {
  it("reduces to exponential noise power when clean magnitude is zero",()=> {
    for (const y of [0, .1, 1, 100]) expect(noisyPowerLogP(y,0,2)).toBeCloseTo(-Math.log(2)-y/2,12);
  });
  it("has finite high-SNR values without subtracting nearly equal large terms",()=> {
    expect(Number.isFinite(noisyPowerLogP(1,1,1e-14))).toBe(true);
    expect(logI0Scaled(1)).toBeCloseTo(-.7640856414928213,12);
    expect(logI0Scaled(100)).toBeCloseTo(-3.220267310057417,10);
  });
  it("integrates the density over clean-power uncertainty rather than averaging log densities",()=> {
    const quadrature={nodes:[-1,1],weights:[.5,.5]};
    const terms=[-1,1].map(z=>Math.exp(noisyPowerLogP(.3,Math.exp(z),.2)));
    expect(marginalEmission([.3],[.2],[0],[1],quadrature,true)).toBeCloseTo(Math.log((terms[0]+terms[1])/2),12);
  });
  it("can reverse a low-SNR log-control preference",()=> {
    // Noise power makes y near n plausible even for a small clean magnitude.
    const y=1,n=1,written=.01,rival=2;
    expect(logPowerControl(y,written,n)).toBeLessThan(logPowerControl(y,rival,n));
    expect(noisyPowerLogP(y,written,n)).toBeGreaterThan(noisyPowerLogP(y,rival,n));
  });
  it("sums all complete boundaries with a normalized timing prior",()=> {
    const e=[[1,5],[2,6],[3,7]];
    const expected=Math.log((Math.exp(1+6+7)+Math.exp(1+2+7))/2);
    expect(boundaryLogP(e)).toBeCloseTo(expected,12);
    expect(boundaryLogP([[0,0]])).toBe(-Infinity);
    expect(boundaryLogP([[0,0],[0,0],[0,0]])).toBe(0);
  });
  it("marginalizes optional noise without allowing phones to disappear",()=> {
    expect(boundaryWithNoise([[0,0],[0,0],[0,0]],[0,0,0])).toBeCloseTo(0,12);
    expect(boundaryWithNoise([[0,0]],[0])).toBe(-Infinity);
    expect(boundaryWithNoise([[5],[5]],[-100,-100])).toBeLessThan(10);
  });
  it("returns a whole-pronunciation reversal on the same frozen pause estimate",()=> {
    const quadrature={nodes:[-Math.sqrt(3),0,Math.sqrt(3)],weights:[1/6,2/3,1/6]};
    const model={schema:1 as const,inventory:"phones",tokens:"بت",frameMs:40 as const,bins:POWER_BINS,counts:[0,1,1],maxFrames:512,
      means:[Array(16).fill(0),Array(16).fill(Math.log(.01)),Array(16).fill(Math.log(2))],
      deviations:Array.from({length:3},()=>Array(16).fill(.05)),quadrature,refined:quadrature};
    const o=new PowerObservations();o.frames.push(...Array.from({length:80},()=>POWER_BINS.map(()=>1)));
    const phones=[{p:"ب",start:0,end:0,conf:1},{p:"ت",start:10,end:11,conf:1}];
    const c={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:12,toFrame:14};
    const result=powerComparison(model,o,phones,c)!;
    expect(result.controlWritten).toBeLessThan(result.controlRival);
    expect(result.powerWritten).toBeGreaterThan(result.powerRival);
    const reverse=powerComparison(model,o,phones,{...c,written:c.rival,rival:c.written})!;
    expect(reverse.powerWritten).toBe(result.powerRival);
    expect(reverse.controlWritten).toBe(result.controlRival);
  });
  it("observes the unchanged frontend and is independent of audio push sizes",()=> {
    const audio=Float32Array.from({length:4000},(_,i)=>Math.sin(.1*i)*.01);
    const baseline=new Fbank().push(audio),spectra:Float64Array[]=[];
    expect(new Fbank().push(audio,p=>spectra.push(p))).toEqual(baseline);
    const a=new PowerObservations(),b=new PowerObservations();a.push(audio);
    for (let i=0;i<audio.length;i+=173) b.push(audio.subarray(i,i+173));
    expect(a.frames).toEqual(b.frames);
    expect(a.frames[0]).toEqual(POWER_BINS.map(k=>spectra[0][k]));
  });
  it("freezes only preceding completed pauses and abstains without them",()=> {
    const o=new PowerObservations();o.frames.push(...Array.from({length:200},()=>POWER_BINS.map(()=>2)));
    const phones=[{p:"ب",start:0,end:1,conf:1},{p:"ت",start:12,end:13,conf:1},{p:"ب",start:40,end:41,conf:1}];
    expect(o.noiseBefore(11,phones)).toBeNull();
    expect(o.noiseBefore(14,phones)).toEqual(POWER_BINS.map(()=>2));
    o.frames.slice(56).forEach(r=>r.fill(100));
    expect(o.noiseBefore(14,phones)).toEqual(POWER_BINS.map(()=>2));
  });
  it("requires compatible assets, full phones and acoustic coverage",()=> {
    const model={schema:1,inventory:"phones",tokens:"بت",frameMs:40,bins:POWER_BINS,counts:[0,1,1],maxFrames:512,
      means:Array.from({length:3},()=>Array(16).fill(0)),deviations:Array.from({length:3},()=>Array(16).fill(1)),
      quadrature:{nodes:Array(17).fill(0),weights:Array(17).fill(1/17)},refined:{nodes:Array(33).fill(0),weights:Array(33).fill(1/33)}};
    const m=validateNoisePowerModel(model,"phones","بت");
    expect(()=>validateNoisePowerModel({...model,bins:[0]},"phones","بت")).toThrow();
    const o=new PowerObservations();o.frames.push(...Array.from({length:60},()=>POWER_BINS.map(()=>1)));
    const c={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:12,toFrame:14};
    expect(powerComparison(m,o,[],c)).toBeNull();
    expect(powerComparison(m,o,[{p:"ب",start:0,end:0,conf:1},{p:"ت",start:10,end:11,conf:1}],{...c,written:"ز"})).toBeNull();
  });
});
