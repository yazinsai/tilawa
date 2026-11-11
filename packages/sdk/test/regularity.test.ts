import { describe,expect,it } from "vitest";
import { RegularityObservations,regularityFeatures,waveletObservation,regularityComparison,regularityReversal,
  regularityLogadd,scoreRegularitySequence,validateRegularityModel } from "../src/regularity.js";
import type { EmissionModel,RegularityModel } from "../src/regularity.js";

const scales=[2,4,8,16,32,64,128];
const emission=(reverse=false): EmissionModel=>({
  center:Array(12).fill(0),scale:Array(12).fill(1),counts:[0,10,10],
  means:[Array(12).fill(0),Array(12).fill(reverse?1:0),Array(12).fill(reverse?0:1)],
  variances:Array.from({length:3},()=>Array(12).fill(1)),
});
const model=(): RegularityModel=>({schema:1,inventory:"test",tokens:"ab",frameMs:40,observationMs:10,dimensions:12,
  scales,stay:.7,maxFrames:2048,regularity:emission(),energy:emission(true)});
const row=(v: number)=>({regularity:Array(12).fill(v),energy:Array(12).fill(v)});

describe("multiscale waveform regularity",()=> {
  it("is invariant to recording gain, polarity and DC level",()=> {
    const wave=Float32Array.from({length:400},(_,i)=>.1*Math.sin(i*.18)+.03*Math.cos(i*.77));
    const baseline=waveletObservation(wave,scales),changed=waveletObservation(wave.map(v=>v*-3+.2),scales);
    for (const arm of ["energy","regularity"] as const) for (let i=0;i<12;i++) expect(changed[arm][i]).toBeCloseTo(baseline[arm][i],5);
    expect(waveletObservation(new Float32Array(400),scales)).toEqual({regularity:Array(12).fill(0),energy:Array(12).fill(0)});
  });
  it("retains different local scaling patterns for periodic, interrupted and turbulent signals",()=> {
    let seed=5;
    const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32-.5;};
    const waves=[Float32Array.from({length:400},(_,i)=>Math.sin(i*.15)),
      Float32Array.from({length:400},(_,i)=>i>150&&i<210?0:Math.sin(i*.15)),
      Float32Array.from({length:400},random)];
    const observations=waves.map(w=>waveletObservation(w,scales));
    expect(observations.every(o=>o.regularity.every(Number.isFinite))).toBe(true);
    for (let i=0;i<3;i++) for (let j=i+1;j<3;j++) {
      expect(observations[i].regularity.reduce((s,v,k)=>s+(v-observations[j].regularity[k])**2,0)).toBeGreaterThan(.01);
    }
  });
  it("pools scale identity out of the equal-capacity energy control",()=> {
    const wave=Float32Array.from({length:400},(_,i)=>Math.sin(i*.15)+(i===180?3:0));
    const forward=waveletObservation(wave,scales),reverse=waveletObservation(wave,[...scales].reverse());
    expect(forward.energy.length).toBe(forward.regularity.length);
    for (let i=0;i<12;i++) expect(reverse.energy[i]).toBeCloseTo(forward.energy[i],10);
    expect(reverse.regularity).not.toEqual(forward.regularity);
  });
  it("preserves observations and frame count under arbitrary microphone push sizes",()=> {
    const wave=Float32Array.from({length:5000},(_,i)=>.2*Math.sin(i*.08)+.1*Math.cos(i*.15));
    const offline=regularityFeatures(wave,scales),streamed=new RegularityObservations(scales);
    for (let i=0;i<wave.length;i+=137) streamed.push(wave.slice(i,i+137));
    expect(streamed.frames).toEqual(offline);
    expect(offline.length).toBe(1+Math.floor((wave.length-400)/160));
  });
  it("sums all complete phone placements with the same normalized duration prior",()=> {
    const m=model(),raw=[0,.1,.9,1].map(row);
    let mass=-Infinity,prior=-Infinity;
    for (let boundary=1;boundary<raw.length;boundary++) {
      let score=0;
      const p=(raw.length-2)*Math.log(m.stay)+Math.log1p(-m.stay);
      for (let t=0;t<raw.length;t++) {
        const id=t<boundary?1:2;
        score+=raw[t].regularity.reduce((s,v,d)=>s-.5*(Math.log(2*Math.PI)+(v-m.regularity.means[id][d])**2),0);
      }
      mass=regularityLogadd(mass,p+score);prior=regularityLogadd(prior,p);
    }
    expect(scoreRegularitySequence(m,raw,"ab","regularity")).toBeCloseTo(mass-prior,10);
  });
  it("requires positive complete written evidence and the independently fitted energy control to prefer the rival",()=> {
    const candidate={word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:0,toFrame:1};
    expect(regularityReversal(regularityComparison(model(),[row(0)],candidate)!)).toBe(true);
    expect(regularityReversal(regularityComparison(model(),[row(1)],candidate)!)).toBe(false);
    expect(regularityReversal(regularityComparison(model(),[row(.5)],candidate)!)).toBe(false);
  });
  it("abstains on unsupported labels, nonfinite audio evidence and incomplete pronunciations",()=> {
    expect(scoreRegularitySequence(model(),[row(0)],"ab","regularity")).toBeNull();
    expect(scoreRegularitySequence(model(),[row(0)],"c","regularity")).toBeNull();
    expect(scoreRegularitySequence(model(),[row(NaN)],"a","regularity")).toBeNull();
    expect(scoreRegularitySequence(model(),[row(0),row(0)],"aa","regularity")).not.toBeNull();
    expect(scoreRegularitySequence(model(),[waveletObservation(new Float32Array(400).fill(NaN),scales)],"a","regularity")).toBeNull();
  });
  it("rejects wrong inventory, malformed scales and invalid emission or timing laws",()=> {
    expect(validateRegularityModel(model(),"test","ab").scales).toEqual(scales);
    expect(()=>validateRegularityModel(model(),"other","ab")).toThrow();
    const m=model();m.energy.variances[1][0]=0;
    expect(()=>validateRegularityModel(m,"test","ab")).toThrow();
    expect(()=>validateRegularityModel({...model(),stay:1},"test","ab")).toThrow();
    expect(()=>validateRegularityModel({...model(),scales:[2,4,8,16,32,64,64]},"test","ab")).toThrow();
  });
});
