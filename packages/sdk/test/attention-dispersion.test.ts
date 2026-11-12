import { describe, expect, it } from "vitest";
import { carryAttentionEvidence, attentionBoundaryScore, attentionComparison, attentionObservations, attentionHeadRows, attentionReversal, attentionVerdictRemovals, type AttentionModel } from "../src/attention-dispersion.js";

const eye=Array.from({length:16},(_,i)=>Array.from({length:16},(_,j)=>i===j?1:0));
const arm=(reverse=false)=>({center:Array(16).fill(0),scale:Array(16).fill(1),
  phones:[0,1,2].map(p=>({mean:Array(16).fill(p===0?10:(reverse?3-p:p)),precision:eye,logDet:0,count:10}))});
const model={dimensions:16,tokens:"بت",dispersion:arm(),control:arm(true)} as AttentionModel;
const c={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:0,toFrame:4};
describe("attention dispersion evidence",()=> {
  it("conditions out path count for constant evidence, preserving repeated articulations",()=> {
    for (const ids of [[1],[1,2],[1,1]]) expect(attentionBoundaryScore(Array.from({length:7},()=>[-.75,-.75,-.75]),ids)).toBeCloseTo(-5.25,12);
    expect(attentionBoundaryScore([[0,0,0]],[1,1])).toBe(-Infinity);
  });
  it("requires opposing finite complete-pronunciation preferences",()=> {
    const x=Array.from({length:4},()=>Array(16).fill(1));
    const result=attentionComparison(model,x,x,c)!;
    expect(attentionReversal(result)).toBe(true);
    expect(attentionReversal({...result,controlWritten:result.controlRival})).toBe(false);
    expect(attentionReversal({...result,dispersionWritten:NaN})).toBe(false);
    expect(attentionReversal({...result,dispersionWritten:result.dispersionRival})).toBe(false);
  });
  it("is antisymmetric when the two complete candidates are exchanged",()=> {
    const x=Array.from({length:4},()=>Array(16).fill(1));
    const a=attentionComparison(model,x,x,c)!,b=attentionComparison(model,x,x,{...c,written:c.rival,rival:c.written})!;
    expect(a.dispersionWritten).toBe(b.dispersionRival);
    expect(a.controlRival).toBe(b.controlWritten);
    expect(attentionReversal(b)).toBe(false);
  });
  it("abstains on missing phones, mismatched clocks or nonfinite features",()=> {
    const x=Array.from({length:4},()=>Array(16).fill(1));
    expect(attentionComparison(model,x,x.slice(1),c)).toBeNull();
    expect(attentionComparison(model,x,x,{...c,rival:"ز"})).toBeNull();
    expect(attentionComparison(model,[Array(16).fill(NaN)],x,c)).toBeNull();
  });
  it("retracts only the exact live occurrence removed by the certificate replay",()=> {
    const warning={type:"checker",v:1,id:"one",timeMs:4000,state:"confirmed",errorClass:"wrong_word",
      location:c.word,expected:c.written,observed:c.rival,confidence:1} as import("@tilawa/core").CheckerEvent;
    const repeated={...warning,id:"two",timeMs:8000};
    const laterGraph={...warning,timeMs:9000};
    const insertion={...warning,id:"three",errorClass:"added_word" as const};
    expect(attentionVerdictRemovals([warning,repeated,laterGraph,insertion],
      [warning,repeated,laterGraph,insertion],[repeated,laterGraph,insertion])).toEqual([warning]);
    expect(attentionVerdictRemovals([warning],[{...warning,timeMs:4001}],[])).toEqual([]);
  });
  it("carries certificates without changing primary phones or posterior rows",()=> {
    const chunks=[{inventory:"test",frameMs:40,phonemes:[{p:"ب",start:0,end:1,conf:.9}],
      posteriors:{startFrame:0,nFrames:2,k:1,ids:[1,1],probs:[.9,.9]}}];
    const certificate={...c,controlWritten:-2,controlRival:-1,dispersionWritten:1,dispersionRival:0};
    const enriched=carryAttentionEvidence(chunks,[certificate]);
    expect(enriched[0].posteriors).toBe(chunks[0].posteriors);
    expect(enriched[0].phonemes[0]).toEqual({...chunks[0].phonemes[0],attentionDispersionVetoes:[certificate]});
    expect(chunks[0].phonemes[0]).not.toHaveProperty("attentionDispersionVetoes");
  });
});


it("reads a head's weighted statistics on the native clock with independent caches",async()=> {
  let calls=0;
  const manifest={tokens:"بت",fbank:{mels:80,padValue:-23},window:39,shift:32,
    states:[{name:"cache",shape:[1],dtype:"float32"}]} as import("../src/model.js").ModelManifest;
  const head={meanOutput:"mean",varianceOutput:"variance",head:1,heads:2,width:2};
  const extractor={name:"frozen",async release(){},async run(feeds:Record<string,import("../src/model.js").Tensor>) {
    expect(Number(feeds.cache.data[0])).toBe(calls);
    const offset=1000*calls++;
    return {
      log_probs:{type:"float32" as const,dims:[1,8,3],data:new Float32Array(24)},
      new_cache:{type:"float32" as const,dims:[1],data:new Float32Array([calls])},
      mean:{type:"float32" as const,dims:[2,8,2],data:Float32Array.from({length:32},(_,i)=>offset+i)},
      variance:{type:"float32" as const,dims:[2,8,2],data:new Float32Array(32).fill(2)},
    };
  }};
  const fitted={...model,heads:[head],selected:0,
    dispersionProjection:Array.from({length:16},()=>[1,0,1,0]),controlProjection:Array.from({length:16},()=>[1,0])};
  const result=await attentionObservations(extractor,manifest,fitted,new Float32Array(16000));
  expect(result.dispersion).toHaveLength(23);
  expect(result.control).toHaveLength(23);
  expect(result.dispersion[0]).toEqual(Array(16).fill(18));
  expect(result.control[0]).toEqual(Array(16).fill(16));
  expect(result.control[8][0]).toBe(1016);
  expect(calls).toBe(3);
});

it("keeps dispersion evidence when opposing values have the same weighted mean",()=> {
  const head={meanOutput:"mean",varianceOutput:"variance",head:0,heads:1,width:1};
  const read=(variance:number)=>attentionHeadRows({
    mean:{type:"float32",dims:[1,2,1],data:new Float32Array([0,0])},
    variance:{type:"float32",dims:[1,2,1],data:new Float32Array([variance,variance])},
  },head,4);
  expect(read(0).mean).toEqual(read(1).mean);
  expect(read(1).variance).toEqual([[1],[1],[1],[1]]);
  expect(()=>read(-1)).toThrow("Negative");
});
