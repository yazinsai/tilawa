import { describe, expect, it } from "vitest";
import { CycleObservations, cycleFeatures, cycleWindow, cycleEmission, cycleComparison, cycleReversal, logadd,
  scoreCycleSequence, validateCycleModel } from "../src/cycle-turbulence.js";
import type { CycleModel } from "../src/cycle-turbulence.js";

const model=():CycleModel=>({schema:1,inventory:"test",tokens:"ab",frameMs:40,observationMs:10,
  center:Array(16).fill(0),scale:Array(16).fill(1),means:[Array(16).fill(0),[0,0,...Array(14).fill(1)],[.5,.5,...Array(14).fill(-1)]],
  variances:Array.from({length:3},()=>Array(16).fill(1)),counts:[0,10,10],cycleCounts:[0,5,5],stay:.7,maxFrames:4096});
const row=(phase=1,q=1)=>[.5,.5,...Array(14).fill(phase),q];
const candidate={word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:0,toFrame:1};

describe("cycle-conditioned turbulence evidence",()=> {
  it("distinguishes phase distributions with identical pooled energy and retains an actual rival",()=> {
    const written=cycleComparison(model(),[row(1),row(1)],candidate)!;
    const rival=cycleComparison(model(),[row(-1),row(-1)],candidate)!;
    expect(cycleReversal(written)).toBe(true);
    expect(rival.controlWritten).toBe(written.controlWritten);
    expect(rival.controlRival).toBe(written.controlRival);
    expect(rival.conditionedRival).toBeGreaterThan(rival.conditionedWritten);
    expect(cycleReversal(rival)).toBe(false);
  });
  it("integrates unreliable phase out exactly and abstains without any reliable cycle",()=> {
    const m=model();
    expect(cycleEmission(m,row(1000,0),1,true)).toBe(cycleEmission(m,row(-1000,0),1,true));
    expect(cycleEmission(m,row(1000,0),1,true)).toBe(cycleEmission(m,row(1000,0),1,false));
    expect(cycleComparison(m,[row(1,0)],candidate)).toBeNull();
    expect(cycleReversal({...candidate,controlWritten:-1,controlRival:0,conditionedWritten:NaN,conditionedRival:0})).toBe(false);
  });
  it("matches exhaustive integration over all complete phone boundaries",()=> {
    const m=model(),raw=[row(.1),row(.3),row(-.2),row(.4)];
    let mass=-Infinity,prior=-Infinity;
    for(let boundary=1;boundary<raw.length;boundary++) {
      const timing=(raw.length-2)*Math.log(m.stay)+Math.log1p(-m.stay);
      mass=logadd(mass,timing+raw.reduce((v,r,t)=>v+cycleEmission(m,r,t<boundary?1:2,true),0));
      prior=logadd(prior,timing);
    }
    expect(scoreCycleSequence(m,raw,"ab",true)).toBeCloseTo(mass-prior,10);
  });
  it("requires every repeated articulation and training coverage, and rejects invalid observations",()=> {
    const m=model();
    expect(scoreCycleSequence(m,[row()],"aa",true)).toBeNull();
    expect(scoreCycleSequence(m,[row(),row()],"aa",true)).not.toBeNull();
    expect(scoreCycleSequence(m,[row()],"c",true)).toBeNull();
    expect(scoreCycleSequence(m,[row(NaN)],"a",true)).toBeNull();
    expect(scoreCycleSequence(m,[row(1,.5)],"a",true)).toBeNull();
    m.cycleCounts[1]=0;expect(scoreCycleSequence(m,[row()],"a",true)).toBeNull();
  });
  it("measures aperiodic phase structure rather than high harmonic power",()=> {
    const low=Array.from({length:800},(_,i)=>Math.sin(2*Math.PI*i/80));
    let state=7;
    const noise=Array.from({length:800},(_,i)=>{state=(1664525*state+1013904223)>>>0;
      return (state/2**32-.5)*(i%80<20?1:.02);});
    const shifted=noise.map((_,i)=>noise[(i+40)%800]);
    const a=cycleWindow(low,[noise,noise],1),initial=cycleWindow(low,[shifted,shifted],1);
    const matched=shifted.map(v=>v*Math.exp((a[0]-initial[0])/2));
    const b=cycleWindow(low,[matched,matched],1);
    expect(a[16]).toBe(1);expect(b[16]).toBe(1);
    expect(a[0]).toBeCloseTo(b[0],12);expect(a[1]).toBeCloseTo(b[1],12);
    expect(a.slice(2,16)).not.toEqual(b.slice(2,16));
    const harmonic=low.map((_,i)=>Math.sin(2*Math.PI*i/10));
    expect(cycleWindow(low,[harmonic,harmonic],1)[16]).toBe(0);
  });
  it("preserves filters and the 50 ms / 10 ms grid under arbitrary audio chunks",()=> {
    const wave=Float32Array.from({length:5000},(_,i)=>.2*Math.sin(i*.08)+.1*Math.cos(i*.15));
    const full=cycleFeatures(wave),streamed=new CycleObservations();
    for(let i=0;i<wave.length;i+=137)streamed.push(wave.subarray(i,i+137));
    expect(streamed.frames).toEqual(full);
    expect(full.length).toBe(Math.floor((wave.length-800)/160)+1);
    expect(full.flat().every(Number.isFinite)).toBe(true);
    expect(cycleFeatures(new Float32Array(1000)).flat().every(v=>v===0)).toBe(true);
  });
  it("validates model geometry, variances, support counts and inventory",()=> {
    const m=model();expect(validateCycleModel(m,"test","ab")).toBe(m);
    expect(()=>validateCycleModel(m,"other","ab")).toThrow();
    m.variances[1][0]=0;expect(()=>validateCycleModel(m,"test","ab")).toThrow();
  });
});
