import { describe, expect, it } from "vitest";
import { carryWavecodecEvidence, wavecodecBoundaryScore, wavecodecComparison, controlObservations, wavecodecReversal, wavecodecVerdictRemovals, type WavecodecModel } from "../src/wavecodec.js";

const eye=Array.from({length:16},(_,i)=>Array.from({length:16},(_,j)=>i===j?1:0));
const arm=(reverse=false)=>({center:Array(16).fill(0),scale:Array(16).fill(1),
  phones:[0,1,2].map(p=>({mean:Array(16).fill(p===0?10:(reverse?3-p:p)),precision:eye,logDet:0,count:10}))});
const model={dimensions:16,tokens:"بت",codec:arm(),control:arm(true)} as WavecodecModel;
const c={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:0,toFrame:4};
describe("waveform codec evidence",()=> {
  it("conditions out path count for constant evidence, preserving repeated articulations",()=> {
    for (const ids of [[1],[1,2],[1,1]]) expect(wavecodecBoundaryScore(Array.from({length:7},()=>[-.75,-.75,-.75]),ids)).toBeCloseTo(-5.25,12);
    expect(wavecodecBoundaryScore([[0,0,0]],[1,1])).toBe(-Infinity);
  });
  it("requires opposing finite complete-pronunciation preferences",()=> {
    const x=Array.from({length:4},()=>Array(16).fill(1));
    const result=wavecodecComparison(model,x,x,c)!;
    expect(wavecodecReversal(result)).toBe(true);
    expect(wavecodecReversal({...result,controlWritten:result.controlRival})).toBe(false);
    expect(wavecodecReversal({...result,codecWritten:NaN})).toBe(false);
    expect(wavecodecReversal({...result,codecWritten:result.codecRival})).toBe(false);
  });
  it("is antisymmetric when the two complete candidates are exchanged",()=> {
    const x=Array.from({length:4},()=>Array(16).fill(1));
    const a=wavecodecComparison(model,x,x,c)!,b=wavecodecComparison(model,x,x,{...c,written:c.rival,rival:c.written})!;
    expect(a.codecWritten).toBe(b.codecRival);
    expect(a.controlRival).toBe(b.controlWritten);
    expect(wavecodecReversal(b)).toBe(false);
  });
  it("abstains on missing phones, mismatched clocks or nonfinite features",()=> {
    const x=Array.from({length:4},()=>Array(16).fill(1));
    expect(wavecodecComparison(model,x,x.slice(1),c)).toBeNull();
    expect(wavecodecComparison(model,x,x,{...c,rival:"ز"})).toBeNull();
    expect(wavecodecComparison(model,[Array(16).fill(NaN)],x,c)).toBeNull();
  });
  it("retracts only the exact live occurrence removed by the certificate replay",()=> {
    const warning={type:"checker",v:1,id:"one",timeMs:4000,state:"confirmed",errorClass:"wrong_word",
      location:c.word,expected:c.written,observed:c.rival,confidence:1} as import("@tilawa/core").CheckerEvent;
    const repeated={...warning,id:"two",timeMs:8000};
    const laterGraph={...warning,timeMs:9000};
    const insertion={...warning,id:"three",errorClass:"added_word" as const};
    expect(wavecodecVerdictRemovals([warning,repeated,laterGraph,insertion],
      [warning,repeated,laterGraph,insertion],[repeated,laterGraph,insertion])).toEqual([warning]);
    expect(wavecodecVerdictRemovals([warning],[{...warning,timeMs:4001}],[])).toEqual([]);
  });
  it("carries certificates without changing primary phones or posterior rows",()=> {
    const chunks=[{inventory:"test",frameMs:40,phonemes:[{p:"ب",start:0,end:1,conf:.9}],
      posteriors:{startFrame:0,nFrames:2,k:1,ids:[1,1],probs:[.9,.9]}}];
    const certificate={...c,controlWritten:-2,controlRival:-1,codecWritten:1,codecRival:0};
    const enriched=carryWavecodecEvidence(chunks,[certificate]);
    expect(enriched[0].posteriors).toBe(chunks[0].posteriors);
    expect(enriched[0].phonemes[0]).toEqual({...chunks[0].phonemes[0],wavecodecVetoes:[certificate]});
    expect(chunks[0].phonemes[0]).not.toHaveProperty("wavecodecVetoes");
  });
});


it("matches exhaustive weighted complete-boundary paths",()=> {
  const values=Array.from({length:5},(_,t)=>[-.2-t*.1,-1+t*.15,-.8-t*.05]);
  for (const ids of [[1],[1,2],[1,1]]) {
    const labels=[0,...ids.flatMap(p=>[p,0])];
    let mass=0,prior=0;
    const visit=(state:number,t:number,weight:number,likelihood:number)=> {
      if (t===values.length) {
        if (state>=labels.length-2) {prior+=weight;mass+=weight*Math.exp(likelihood);}
        return;
      }
      const next=[state,state+1,state+2].filter(s=>s<labels.length&&
        (s!==state+2||(labels[s]!==0&&labels[s]!==labels[state])));
      for (const s of next) visit(s,t+1,weight/next.length,likelihood+values[t][labels[s]]);
    };
    for (const state of [0,1]) visit(state,1,.5,values[0][labels[state]]);
    expect(wavecodecBoundaryScore(values,ids)).toBeCloseTo(Math.log(mass/prior),12);
  }
});

it("extracts shipped features with independent states on their native clock",async()=> {
  let calls=0;
  const manifest={tokens:"بت",fbank:{mels:80,padValue:-23},window:39,shift:32,
    states:[{name:"cache",shape:[1],dtype:"float32"}]} as import("../src/model.js").ModelManifest;
  const extractor={name:"frozen",async release(){},async run(feeds:Record<string,import("../src/model.js").Tensor>) {
    expect(Number(feeds.cache.data[0])).toBe(calls);
    const offset=1000*calls++;
    return {
      log_probs:{type:"float32" as const,dims:[1,8,3],data:new Float32Array(24)},
      new_cache:{type:"float32" as const,dims:[1],data:new Float32Array([calls])},
      encoder:{type:"float32" as const,dims:[8,1,2],data:Float32Array.from({length:16},(_,i)=>i%2?0:offset+Math.floor(i/2))}};
  }};
  const result=await controlObservations(extractor,manifest,{output:"encoder",controlProjection:Array.from({length:16},()=>[1,0])},new Float32Array(16000));
  expect(result).toHaveLength(23);
  expect(result[0]).toEqual(Array(16).fill(0));
  expect(result[8]).toEqual(Array(16).fill(1000));
  expect(result[16]).toEqual(Array(16).fill(2000));
  expect(calls).toBe(3);
});
