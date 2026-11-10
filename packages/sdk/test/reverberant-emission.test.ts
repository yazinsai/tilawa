import { describe, expect, it } from "vitest";
import { Fbank } from "../src/fbank.js";
import { decayWeights, logadd, ReverberantObservations, reverberantFeatures, reverberantComparison,
  reverberantReversal, scoreReverberantSequence, validateReverberantModel } from "../src/reverberant-emission.js";
import type { ReverberantModel } from "../src/reverberant-emission.js";

const model = (): ReverberantModel => {
  const a=Array(20).fill(.2/19), b=a.slice(); a[0]=.8; b[1]=.8;
  return {schema:1,inventory:"test",tokens:"abc",frameMs:40,dimensions:20,history:4,mix:.7,decay:.9,
    spectra:[Array(20).fill(.05),a,b,a.map((v,d)=>.7*v+.3*b[d])],
    variances:Array.from({length:4},()=>Array(20).fill(.1)),stay:[.6,.7,.8,.9],
    initial:[.1,.3,.4,.2],counts:[1,10,10,10],maxFrames:2048};
};
const observations=(m:ReverberantModel,ids:number[])=>ids.map(p=>m.spectra[p].map(Math.log));

describe("causal reverberant spectral evidence",()=> {
  it("reverses a lingering-spectrum rival and retains the rival's actual spectrum",()=> {
    const m=model(); m.initial=[0,1,0,0];
    const c={word:{surah:1,ayah:1,word:1},written:"ab",rival:"ac",fromFrame:0,toFrame:4};
    expect(reverberantReversal(reverberantComparison(m,observations(m,[1,1,1,3]),c)!)).toBe(true);
    const actual=reverberantComparison(m,observations(m,[1,1,3,3,3,3]),c)!;
    expect(reverberantReversal(actual)).toBe(false);
    expect(actual.controlRival).toBeGreaterThan(actual.controlWritten);
  });
  it("matches exhaustive boundary and incoming-memory enumeration beyond the memory horizon",()=> {
    const m=model(), raw=observations(m,[1,1,1,3,2,2,2]), ids=[1,2], weights=decayWeights(m);
    let mass=-Infinity, prior=-Infinity;
    for (let split=1;split<raw.length;split++) for (let seed=0;seed<m.initial.length;seed++) {
      const path=raw.map((_,t)=>t<split ? 0 : 1);
      let lp=Math.log(m.initial[seed]), acoustic=0;
      for (let t=0;t<raw.length;t++) {
        if (t) lp+=path[t]===path[t-1] ? Math.log(m.stay[ids[path[t-1]]]) : Math.log1p(-m.stay[ids[path[t-1]]]);
        const spectrum=m.spectra[ids[path[t]]].map(v=>weights[0]*v);
        for (let l=1;l<=m.history;l++) {
          const p=t<l ? seed : ids[path[t-l]];
          for (let d=0;d<20;d++) spectrum[d]+=weights[l]*m.spectra[p][d];
        }
        acoustic+=raw[t].reduce((sum,x,d)=> {
          const v=m.variances[ids[path[t]]][d];
          return sum-.5*(Math.log(2*Math.PI*v)+(x-Math.log(spectrum[d]))**2/v);
        },0);
      }
      mass=logadd(mass,lp+acoustic); prior=logadd(prior,lp);
    }
    expect(scoreReverberantSequence(m,raw,"ab",true)).toBeCloseTo(mass-prior,10);
    expect(weights.reduce((s,v)=>s+v,0)).toBeCloseTo(1,12);
  });
  it("has an exactly matched control at zero memory, including repeated articulations",()=> {
    const m=model(); m.mix=0;
    const raw=observations(m,[1,1,2,2,2,2]);
    for (const word of ["a","ab","aa","aba"]) expect(scoreReverberantSequence(m,raw,word,true))
      .toBeCloseTo(scoreReverberantSequence(m,raw,word,false)!,10);
    expect(scoreReverberantSequence(m,raw.slice(0,1),"aa",true)).toBeNull();
  });
  it("abstains on incomplete, non-finite, oversized or unsupported complete comparisons",()=> {
    const m=model(), raw=observations(m,[1,1,2]);
    expect(scoreReverberantSequence(m,raw,"ad",true)).toBeNull();
    expect(scoreReverberantSequence(m,[[NaN,...raw[0].slice(1)]],"a",true)).toBeNull();
    m.counts[1]=0;
    expect(scoreReverberantSequence(m,raw,"a",true)).toBeNull();
    expect(scoreReverberantSequence(model(),Array(2049).fill(raw[0]),"a",true)).toBeNull();
  });
  it("keeps spectral power and frame alignment invariant to microphone push sizes",()=> {
    const wave=Float32Array.from({length:7000},(_,i)=>.2*Math.sin(i*.08)+.1*Math.cos(i*.15));
    const frames=reverberantFeatures(wave), streamed=new ReverberantObservations();
    for (let i=0;i<wave.length;i+=137) streamed.push(wave.slice(i,i+137));
    expect(streamed.frames).toEqual(frames);
    const native=new Fbank().push(wave);
    expect(frames.length).toBe(Math.floor(native.length/4));
    expect(frames.every(row=>Math.abs(row.reduce((s,v)=>s+Math.exp(v),0)-1)<1e-10)).toBe(true);
    const power=Array.from({length:20},(_,d)=>native.slice(0,4).reduce((s,r)=>
      s+Math.exp(r[4*d])+Math.exp(r[4*d+1])+Math.exp(r[4*d+2])+Math.exp(r[4*d+3]),0));
    expect(frames[0][0]).toBeCloseTo(Math.log(power[0]/power.reduce((s,v)=>s+v,0)),12);
  });
  it("rejects malformed, unnormalized and wrong-inventory assets",()=> {
    const m=model(); expect(validateReverberantModel(m,"test","abc")).toBe(m);
    expect(()=>validateReverberantModel(m,"other","abc")).toThrow();
    m.spectra[1][0]=1;
    expect(()=>validateReverberantModel(m,"test","abc")).toThrow();
    const invalid=model(); invalid.stay[1]=1;
    expect(()=>validateReverberantModel(invalid,"test","abc")).toThrow();
  });
});
