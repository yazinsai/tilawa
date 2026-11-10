import { describe, expect, it } from "vitest";
import { Fbank } from "../src/fbank.js";
import { CueObservations, cueComparison, cueFeatures, cueGraph, cueLogadd, cueReversal, scoreCueSequence, validateCueModel } from "../src/synchronous-cues.js";
import type { CueModel } from "../src/synchronous-cues.js";

const model = (): CueModel => ({schema:1,inventory:"test",tokens:"ab",frameMs:40,observationMs:10,dimensions:20,
  center:Array(20).fill(0),scale:Array(20).fill(1),classes:[[0,0,0],[1,1,1],[2,2,2]],counts:[10,10,10],stay:.7,maxFrames:4096,
  means:Array.from({length:3},(_,h)=>Array.from({length:3},(_,c)=>Array.from({length:20},(_,d)=>c===2&&d===h?1:0))),
  variances:Array.from({length:3},(_,h)=>Array.from({length:3},(_,c)=>Array.from({length:20},(_,d)=>d===h?[.1,.5,.05][c]:100))),
});
const candidate={word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:0,toFrame:1};
const row=(values: number[])=>[...values,...Array(20-values.length).fill(0)];

describe("synchronized place, manner and voicing evidence",()=> {
  it("rejects a rival assembled from cues at three different times",()=> {
    const c=cueComparison(model(),[row([1,0,0]),row([0,1,0]),row([0,0,1])],candidate)!;
    expect(c.independentRival).toBeGreaterThan(c.independentWritten);
    expect(c.synchronizedWritten).toBeGreaterThan(c.synchronizedRival);
    expect(cueReversal(c)).toBe(true);
  });
  it("retains a spoken rival with simultaneous distinguishing cues",()=> {
    const c=cueComparison(model(),[row([0,0,0]),row([1,1,1]),row([0,0,0])],candidate)!;
    expect(c.synchronizedRival).toBeGreaterThan(c.synchronizedWritten);
    expect(cueReversal(c)).toBe(false);
  });
  it("matches exhaustive path sums and normalized timing priors in both arms",()=> {
    const m=model(), raw=[row([.2,.5,.7]),row([1,.1,.4]),row([0,.3,.8]),row([.1,.9,.2])];
    const {labels,incoming}=cueGraph([1,2],m.stay);
    const paths=(path: number[]):number[][]=> {
      if(path.length===raw.length) return path.at(-1)!>=labels.length-2 ? [path] : [];
      return labels.flatMap((_,to)=>incoming[to].some(([from])=>from===path.at(-1)) ? paths([...path,to]) : []);
    };
    const sum=(heads:number[])=> {
      let mass=-Infinity, prior=-Infinity;
      for(const path of [...paths([0]),...paths([1])]) {
        let timing=-Math.log(2), acoustic=0;
        for(let t=0;t<raw.length;t++) {
          const s=path[t];
          if(t) timing+=incoming[s].find(([p])=>p===path[t-1])![1];
          for(const h of heads) {
            const c=m.classes[labels[s]][h];
            acoustic+=raw[t].reduce((v,x,d)=>v-.5*(Math.log(2*Math.PI*m.variances[h][c][d])+
              (x-m.means[h][c][d])**2/m.variances[h][c][d]),0);
          }
        }
        mass=cueLogadd(mass,timing+acoustic);prior=cueLogadd(prior,timing);
      }
      return mass-prior;
    };
    expect(scoreCueSequence(m,raw,"ab",true)).toBeCloseTo(sum([0,1,2]),10);
    expect(scoreCueSequence(m,raw,"ab",false)).toBeCloseTo(sum([0])+sum([1])+sum([2]),10);
  });
  it("requires every original articulation even when features are identical",()=> {
    const m=model();m.classes[2]=m.classes[1].slice();
    expect(cueComparison(m,[row([0,0,0])],candidate)).toBeNull();
    expect(scoreCueSequence(m,[row([0,0,0]),row([0,0,0])],"aa",true)).toBeNull();
    expect(scoreCueSequence(m,[row([0,0,0]),row([0,0,0])],"ab",true)).not.toBeNull();
    expect(scoreCueSequence(m,[row([NaN,0,0])],"a",true)).toBeNull();
    expect(scoreCueSequence(m,[row([0,0,0])],"c",true)).toBeNull();
    m.counts[1]=0;
    expect(scoreCueSequence(m,[row([0,0,0])],"a",true)).toBeNull();
  });
  it("keeps native feature values and clocks under arbitrary audio chunking",()=> {
    const audio=Float32Array.from({length:5000},(_,i)=>.2*Math.sin(i*.08)+.1*Math.cos(i*.15));
    const native=new Fbank().push(audio), frames=cueFeatures(audio), streaming=new CueObservations();
    for(let i=0;i<audio.length;i+=137) streaming.push(audio.slice(i,i+137));
    expect(streaming.frames).toEqual(frames);
    expect(frames.length).toBe(native.length);
    expect(frames[0][0]).toBe((native[0][0]+native[0][1]+native[0][2]+native[0][3])/4);
  });
  it("validates class indices, shapes, inventory and emission variances",()=> {
    const m=model();
    expect(validateCueModel(m,"test","ab")).toBe(m);
    expect(()=>validateCueModel(m,"other","ab")).toThrow();
    m.classes[1][1]=10;
    expect(()=>validateCueModel(m,"test","ab")).toThrow();
    const invalid=model();invalid.variances[0][0][0]=0;
    expect(()=>validateCueModel(invalid,"test","ab")).toThrow();
  });
});
