import { describe, expect, it } from "vitest";
import { Fbank } from "../src/fbank.js";
import { PhaseObservations, phaseFeatures, phaseComparison, phaseReversal, phaseLogadd, scorePhaseSequence, validatePhaseModel } from "../src/intraphone-phase.js";
import type { PhaseModel } from "../src/intraphone-phase.js";

const model = (): PhaseModel => ({ schema:1,inventory:"test",tokens:"ab",frameMs:40,observationMs:10,dimensions:20,
  center:Array(20).fill(0),scale:Array(20).fill(1),counts:[0,10,10],stay:[.6,.8,.5],maxFrames:2048,
  means:Array.from({length:3},(_,p)=>Array.from({length:3},(_,q)=>Array.from({length:20},(_,d)=>d ? 0 : p===1 ? q : 2-q))),
  variances:Array.from({length:3},(_,p)=>Array.from({length:3},()=>Array.from({length:20},(_,d)=>d ? 1 : p===1 ? .05 : .04))),
});
const row=(v: number)=>Array.from({length:20},(_,d)=>d ? 0 : v);

describe("ordered intraphone acoustic evidence",()=> {
  it("distinguishes attack-to-release order from a matched phase-pooled mixture",()=> {
    const c={word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:0,toFrame:1};
    const forward=phaseComparison(model(),[row(0),row(1),row(2)],c)!;
    expect(phaseReversal(forward)).toBe(true);
    const backwards=phaseComparison(model(),[row(2),row(1),row(0)],c)!;
    expect(backwards.controlWritten).toBeCloseTo(forward.controlWritten,10);
    expect(backwards.controlRival).toBeCloseTo(forward.controlRival,10);
    expect(phaseReversal(backwards)).toBe(false);
    expect(backwards.orderedRival).toBeGreaterThan(backwards.orderedWritten);
  });
  it("equals exhaustive path enumeration with latent phase and phone boundaries",()=> {
    const m=model(), raw=[0,.5,1,2,2,1,0].map(row), ids=[1,2];
    const walk=(path: number[]): number[][] => {
      if (path.length===raw.length) return path.at(-1)===5 ? [path] : [];
      const s=path.at(-1)!;
      return [...walk([...path,s]),...(s<5 ? walk([...path,s+1]) : [])];
    };
    let mass=-Infinity, prior=-Infinity;
    for (const path of walk([0])) {
      let p=0, acoustic=0;
      for (let t=0;t<path.length;t++) {
        const s=path[t], q=s%3, id=ids[Math.floor(s/3)];
        if (t) p+=path[t-1]===s ? Math.log(m.stay[q]) : Math.log1p(-m.stay[path[t-1]%3]);
        acoustic+=raw[t].reduce((v,x,d)=> {
          const variance=m.variances[id][q][d], mu=m.means[id][q][d];
          return v-.5*(Math.log(2*Math.PI*variance)+(x-mu)**2/variance);
        },0);
      }
      mass=phaseLogadd(mass,p+acoustic); prior=phaseLogadd(prior,p);
    }
    expect(scorePhaseSequence(m,raw,"ab",true)).toBeCloseTo(mass-prior,10);
  });
  it("requires complete phases for every phone, including repeated phones, and abstains on unsupported inputs",()=> {
    expect(scorePhaseSequence(model(),[row(0),row(1),row(2)],"aa",true)).toBeNull();
    expect(scorePhaseSequence(model(),Array.from({length:6},(_,i)=>row(i%3)),"aa",true)).not.toBeNull();
    expect(scorePhaseSequence(model(),[row(0),row(1),row(2)],"c",true)).toBeNull();
    expect(scorePhaseSequence(model(),[row(NaN),row(1),row(2)],"a",true)).toBeNull();
    const sparse=model(); sparse.counts[1]=0;
    expect(scorePhaseSequence(sparse,[row(0),row(1),row(2)],"a",true)).toBeNull();
  });
  it("keeps the native frontend and observation clock identical under different audio push sizes",()=> {
    const wave=Float32Array.from({length:5000},(_,i)=>.2*Math.sin(i*.08)+.1*Math.cos(i*.15));
    const native=new Fbank().push(wave);
    const features=phaseFeatures(wave), streamed=new PhaseObservations();
    for (let i=0;i<wave.length;i+=137) streamed.push(wave.slice(i,i+137));
    expect(streamed.frames).toEqual(features);
    expect(features.length).toBe(native.length);
    expect(features[0][0]).toBe((native[0][0]+native[0][1]+native[0][2]+native[0][3])/4);
  });
  it("validates bundle identity, positive variances and proper transition probabilities",()=> {
    const m=model();
    expect(validatePhaseModel(m,"test","ab")).toBe(m);
    expect(()=>validatePhaseModel(m,"other","ab")).toThrow();
    m.stay[1]=1;
    expect(()=>validatePhaseModel(m,"test","ab")).toThrow();
  });
});
