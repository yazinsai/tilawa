import { describe, expect, it } from "vitest";
import { gestureMeans, GestureObservations, scoreGestureSequence, validateGestureModel } from "../src/gesture-emission.js";
import type { GestureModel, GestureParameters } from "../src/gesture-emission.js";

function parameters(): GestureParameters {
  const spectra=Array.from({length:9},(_,s)=> {
    const r=Array.from({length:20},(_,d)=>1+((s*7+d*3)%11));
    return r.map(v=>v/r.reduce((a,b)=>a+b,0));
  });
  return {spectra,variances:spectra.map(()=>Array(20).fill(.5)),counts:Array(9).fill(1),
    stay:Array.from({length:9},(_,s)=>.4+.05*s),weights:[.2,.4]};
}
function model(): GestureModel {
  return {schema:1,inventory:"test",tokens:"ab",frameMs:40,observationMs:10,dimensions:20,maxFrames:8192,
    overlap:parameters(),control:parameters()};
}
function exhaustive(m: GestureModel,x:number[][],overlap:boolean): number {
  const p=overlap ? m.overlap : m.control, states=[3,4,5,6,7,8], means=gestureMeans(p,[1,2],overlap);
  const add=(values:number[])=> {const max=Math.max(...values); return max+Math.log(values.reduce((s,v)=>s+Math.exp(v-max),0));};
  const values:number[]=[],priors:number[]=[];
  for (let mask=0;mask<2**(x.length-1);mask++) {
    const path=[0]; let prior=0;
    for (let t=1;t<x.length;t++) {
      const advance=(mask>>(t-1))&1, s=path[t-1];
      if (s>=states.length) break;
      prior+=Math.log(advance ? 1-p.stay[states[s]] : p.stay[states[s]]);
      path.push(s+advance);
    }
    if (path.length!==x.length || path.at(-1)!==5) continue;
    const density=x.reduce((sum,row,t)=>sum+row.reduce((v,y,d)=> {
      const r=y-means[path[t]][d];return v-.5*(Math.log(Math.PI)+r*r/.5);
    },0),0);
    priors.push(prior); values.push(prior+density);
  }
  return add(values)-add(priors);
}

describe("overlapping gesture sequence evidence",()=> {
  it("matches exhaustive summation over complete phase and phone boundaries in both arms",()=> {
    const m=model(), x=[3,4,4,5,6,7,7,8].map(s=>m.overlap.spectra[s].map(Math.log));
    for (const overlap of [true,false]) expect(scoreGestureSequence(m,x,"ab",overlap)).toBeCloseTo(exhaustive(m,x,overlap),10);
  });
  it("uses normalized shared kernels for preceding release and following attack",()=> {
    const m=parameters(),mean=gestureMeans(m,[1,2],true);
    expect(Math.exp(mean[2][0])).toBeCloseTo((m.spectra[5][0]+.4*m.spectra[6][0])/1.4,12);
    expect(Math.exp(mean[3][0])).toBeCloseTo((m.spectra[6][0]+.2*m.spectra[5][0])/1.2,12);
    for (const row of mean) expect(row.reduce((s,v)=>s+Math.exp(v),0)).toBeCloseTo(1,12);
    expect(gestureMeans(m,[1,2],false)).not.toEqual(mean);
  });
  it("abstains on missing phone coverage, unknown labels, incomplete sequences and invalid observations",()=> {
    const m=model(),x=Array.from({length:8},()=>Array(20).fill(-Math.log(20)));
    expect(scoreGestureSequence(m,x,"c",true)).toBeNull();
    expect(scoreGestureSequence(m,x.slice(0,5),"aa",true)).toBeNull();
    m.overlap.counts[4]=0;
    expect(scoreGestureSequence(m,x,"ab",true)).toBeNull();
    expect(scoreGestureSequence(model(),[[NaN]],"a",false)).toBeNull();
  });
  it("rejects assets with a different clock, inventory or unnormalized spectrum",()=> {
    expect(validateGestureModel(model(),"test","ab")).toEqual(model());
    expect(()=>validateGestureModel({...model(),observationMs:40},"test","ab")).toThrow();
    const m=model();m.control.spectra[0][0]=2;
    expect(()=>validateGestureModel(m,"test","ab")).toThrow();
    expect(()=>validateGestureModel(model(),"other","ab")).toThrow();
  });
  it("keeps spectral observations and their clock independent of input push size",()=> {
    const wave=Float32Array.from({length:3200},(_,i)=>.2*Math.sin(i*.08)), a=new GestureObservations(), b=new GestureObservations();
    a.push(wave);
    for (let i=0;i<wave.length;i+=137) b.push(wave.slice(i,i+137));
    expect(a.frames).toEqual(b.frames);
    expect(a.frames.length).toBe(18);
    for (const row of a.frames) expect(row.reduce((s,v)=>s+Math.exp(v),0)).toBeCloseTo(1,10);
  });
});
