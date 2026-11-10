import { describe, expect, it } from "vitest";
import { MicropitchObservations, micropitchComparison, micropitchReversal, pitchFrames, pitchResiduals, pitchStates, scoreMicropitch, validateMicropitchModel } from "../src/micropitch.js";
import type { MicropitchModel } from "../src/micropitch.js";

const model=():MicropitchModel=> {
  const ss=pitchStates("بَت");
  return {schema:1,inventory:"toy",tokens:"بَت",radius:.35,reliability:.8,shrinkage:16,scale:.02,variance:1,stay:[.65,.9,.65],
    means:Object.fromEntries(ss.map(s=>[s.key,0])),control:Object.fromEntries(ss.map(s=>[s.control,0])),counts:Object.fromEntries(ss.map(s=>[s.key,10]))};
};
const candidate={word:{surah:1,ayah:1,word:1},written:"بَت",rival:"تَب",fromFrame:0,toFrame:3};
describe("independent consonant micropitch evidence",()=> {
  it("estimates pitch without inventing it on silence and preserves input chunking",()=> {
    const audio=Float32Array.from({length:4800},(_,i)=>.2*Math.sin(2*Math.PI*160*i/16000));
    const raw=pitchFrames(audio);
    expect(raw.length).toBeGreaterThan(20);
    expect(Math.exp(raw[10].logPitch)).toBeCloseTo(160,0);
    expect(raw[10].reliability).toBeGreaterThan(.95);
    expect(pitchFrames(new Float32Array(1600)).every(f=>f.reliability===0)).toBe(true);
    const stream=new MicropitchObservations();
    for(let i=0;i<audio.length;i+=137) stream.push(audio.slice(i,i+137));
    expect(stream.frames).toEqual(raw);
  });
  it("removes a shared linear melody and retains an isolated disturbance",()=> {
    const frames=Array.from({length:150},(_,i)=>({logPitch:Math.log(150)+i*.0008,reliability:.99}));
    const baseline=pitchResiduals(frames,.35,.8);
    expect(Math.abs(baseline[70][0])).toBeLessThan(1e-9);
    frames[70].logPitch+=.04;
    expect(pitchResiduals(frames,.35,.8)[70][0]).toBeCloseTo(.04,9);
    expect(pitchResiduals(frames.map(f=>({...f,reliability:.5})),.35,.8).every(r=>r[1]===0)).toBe(true);
  });
  it("retains every length-expanded articulation and conditions only vowel edges",()=> {
    const ss=pitchStates("بَاات");
    expect(ss.length).toBe(11);
    expect(ss.filter(s=>s.adjacent).every(s=>s.phase!==1)).toBe(true);
    expect(ss.filter(s=>s.adjacent).map(s=>s.key)).toEqual(["َ:0:ب","ا:2:ت"]);
  });
  it("matches exhaustive boundary summation with the same normalized timing prior",()=> {
    const m=model(),raw=Array.from({length:7},(_,i)=>[i*.002,1]),ss=pitchStates(candidate.written);
    m.means[ss[1].key]=.7;
    let likelihood=0,prior=0;
    const enumerate=(path:number[])=> {
      if(path.length===raw.length) {
        if(path.at(-1)!==ss.length-1) return;
        let p=1,e=1;
        for(let t=0;t<path.length;t++) {
          const s=path[t],phase=ss[s].phase;
          if(t) p*=s===path[t-1]?m.stay[phase]:1-m.stay[ss[path[t-1]].phase];
          const y=raw[t][0]/m.scale-m.means[ss[s].key];
          e*=Math.exp(-.5*(Math.log(2*Math.PI)+y*y))/m.scale;
        }
        likelihood+=p*e;prior+=p;return;
      }
      const s=path.at(-1)!;
      enumerate([...path,s]);if(s+1<ss.length) enumerate([...path,s+1]);
    };
    enumerate([0]);
    expect(scoreMicropitch(m,raw,candidate.written,true)!.score).toBeCloseTo(Math.log(likelihood/prior),10);
  });
  it("abstains on missing training coverage, unreliable pitch and incomplete pronunciations",()=> {
    const m=model(),raw=Array.from({length:12},()=>[0,0]);
    expect(micropitchComparison(m,raw,candidate)).toBeNull();
    expect(scoreMicropitch(m,[[0,1]],candidate.written,true)).toBeNull();
    expect(scoreMicropitch(m,raw,"زَ",true)).toBeNull();
    m.counts["َ:0:ب"]=0;
    expect(scoreMicropitch(m,raw,candidate.written,true)).toBeNull();
  });
  it("requires strict preference reversal in both matched arms",()=> {
    const c={...candidate,controlWritten:-2,controlRival:-1,micropitchWritten:2,micropitchRival:1};
    expect(micropitchReversal(c)).toBe(true);
    expect(micropitchReversal({...c,controlWritten:-1})).toBe(false);
    expect(micropitchReversal({...c,micropitchWritten:1})).toBe(false);
    expect(micropitchReversal({...c,micropitchWritten:NaN})).toBe(false);
  });
  it("rejects corrupt trained parameters and mismatched inventories",()=> {
    expect(validateMicropitchModel(model(),"toy","بَت")).toEqual(model());
    expect(()=>validateMicropitchModel({...model(),variance:0},"toy","بَت")).toThrow();
    expect(()=>validateMicropitchModel(model(),"other","بَت")).toThrow();
  });
});
