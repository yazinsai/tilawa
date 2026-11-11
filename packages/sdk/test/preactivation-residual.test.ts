import { describe, expect, it } from "vitest";
import { carryResidualEvidence, residualBoundaryScore, residualComparison, residualObservations, residualReversal, residualVerdictRemovals, residualFeatures, eligibleResidualTake, validateResidualModel, type ActivationSite, type ResidualModel } from "../src/preactivation-residual.js";

const eye=Array.from({length:16},(_,i)=>Array.from({length:16},(_,j)=>i===j?1:0));
const arm=(reverse=false)=>({center:Array(16).fill(0),scale:Array(16).fill(1),
  phones:[0,1,2].map(p=>({mean:Array(16).fill(p===0?10:(reverse?3-p:p)),precision:eye,logDet:0,count:10}))});
const model={dimensions:16,tokens:"بت",residual:arm(),control:arm(true)} as ResidualModel;
const c={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:0,toFrame:4};
describe("preactivation residual evidence",()=> {
  it("conditions out path count for constant evidence, preserving repeated articulations",()=> {
    for (const ids of [[1],[1,2],[1,1]]) expect(residualBoundaryScore(Array.from({length:7},()=>[-.75,-.75,-.75]),ids)).toBeCloseTo(-5.25,12);
    expect(residualBoundaryScore([[0,0,0]],[1,1])).toBe(-Infinity);
  });
  it("requires opposing finite complete-pronunciation preferences",()=> {
    const x=Array.from({length:4},()=>Array(16).fill(1));
    const result=residualComparison(model,x,x,c)!;
    expect(residualReversal(result)).toBe(true);
    expect(residualReversal({...result,controlWritten:result.controlRival})).toBe(false);
    expect(residualReversal({...result,residualWritten:NaN})).toBe(false);
    expect(residualReversal({...result,residualWritten:result.residualRival})).toBe(false);
  });
  it("is antisymmetric when the two complete candidates are exchanged",()=> {
    const x=Array.from({length:4},()=>Array(16).fill(1));
    const a=residualComparison(model,x,x,c)!,b=residualComparison(model,x,x,{...c,written:c.rival,rival:c.written})!;
    expect(a.residualWritten).toBe(b.residualRival);
    expect(a.controlRival).toBe(b.controlWritten);
    expect(residualReversal(b)).toBe(false);
  });
  it("abstains on missing phones, mismatched clocks or nonfinite features",()=> {
    const x=Array.from({length:4},()=>Array(16).fill(1));
    expect(residualComparison(model,x,x.slice(1),c)).toBeNull();
    expect(residualComparison(model,x,x,{...c,rival:"ز"})).toBeNull();
    expect(residualComparison(model,[Array(16).fill(NaN)],x,c)).toBeNull();
  });
  it("retracts only the exact live occurrence removed by the certificate replay",()=> {
    const warning={type:"checker",v:1,id:"one",timeMs:4000,state:"confirmed",errorClass:"wrong_word",
      location:c.word,expected:c.written,observed:c.rival,confidence:1} as import("@tilawa/core").CheckerEvent;
    const repeated={...warning,id:"two",timeMs:8000};
    const laterGraph={...warning,timeMs:9000};
    const insertion={...warning,id:"three",errorClass:"added_word" as const};
    expect(residualVerdictRemovals([warning,repeated,laterGraph,insertion],
      [warning,repeated,laterGraph,insertion],[repeated,laterGraph,insertion])).toEqual([warning]);
    expect(residualVerdictRemovals([warning],[{...warning,timeMs:4001}],[])).toEqual([]);
  });
  it("carries certificates without changing primary phones or posterior rows",()=> {
    const chunks=[{inventory:"test",frameMs:40,phonemes:[{p:"ب",start:0,end:1,conf:.9}],
      posteriors:{startFrame:0,nFrames:2,k:1,ids:[1,1],probs:[.9,.9]}}];
    const certificate={...c,controlWritten:-2,controlRival:-1,residualWritten:1,residualRival:0};
    const enriched=carryResidualEvidence(chunks,[certificate]);
    expect(enriched[0].posteriors).toBe(chunks[0].posteriors);
    expect(enriched[0].phonemes[0]).toEqual({...chunks[0].phonemes[0],preactivationResidualVetoes:[certificate]});
    expect(chunks[0].phonemes[0]).not.toHaveProperty("preactivationResidualVetoes");
  });
});


const site:ActivationSite={pre:"pre",post:"post",width:2,kind:"Swoosh",scope:"encoder/activation",
  preProjection:Array.from({length:8},()=>[1,0]),postProjection:Array.from({length:16},()=>[1,0]),
  nonlinearProjection:Array.from({length:64},(_,i)=>i?[0,0]:[1,0])};
const predictor={center:[0,0],scale:[1,1],strength:1,
  coefficients:Array.from({length:8},()=>Array.from({length:67},(_,i)=>i===3?1:0))};
const fitted={...model,schema:1,inventory:"test",primarySha256:"hash",sites:[site],site:0,predictor} as ResidualModel;

it("uses all corresponding post channels to predict preactivation coordinates",()=> {
  const coefficients=Array.from({length:8},()=>Array.from({length:67},(_,i)=>i===2?2:0));
  const result=residualFeatures(site,{...predictor,coefficients},[6,0],[99,3]);
  expect(result.residual.slice(8)).toEqual(Array(8).fill(0));
  expect(result.control).toEqual(Array(16).fill(99));
  expect(()=>residualFeatures(site,predictor,[1],[1,0])).toThrow();
});

it("rejects malformed declared models and predictors",()=> {
  const manifest={inventory:"test",tokens:"بت",sha256:"hash"} as import("../src/model.js").ModelManifest;
  expect(validateResidualModel(fitted,manifest)).toBe(fitted);
  expect(()=>validateResidualModel({...fitted,site:3},manifest)).toThrow();
  expect(()=>validateResidualModel({...fitted,predictor:{...predictor,scale:[0,1]}},manifest)).toThrow();
  expect(()=>validateResidualModel({...fitted,sites:null},manifest)).toThrow();
  expect(()=>validateResidualModel({...fitted,predictor:{...predictor,coefficients:[[1]]}},manifest)).toThrow();
});

it("accepts only usable none-labelled training speakers with no edits or repeats",()=> {
  const label={speaker:"a",split:"train",kind:"none",scored:true,events:[],non_errors:[]};
  expect(eligibleResidualTake(label,new Set())).toBe(true);
  for(const changed of [{split:"eval"},{kind:"substituted_word"},{exclude:true},{excluded:true},{scored:false},
    {events:[{}]},{non_errors:[{kind:"repeat"}]}])expect(eligibleResidualTake({...label,...changed},new Set())).toBe(false);
  expect(eligibleResidualTake(label,new Set(["a"]))).toBe(false);
});

it("computes nonlinear residuals on actual activation pairs before native-clock pooling",async()=> {
  let calls=0;
  const manifest={tokens:"بت",fbank:{mels:80,padValue:-23},window:39,shift:32,
    states:[{name:"cache",shape:[1],dtype:"float32"}]} as import("../src/model.js").ModelManifest;
  const extractor={name:"frozen",async release(){},async run(feeds:Record<string,import("../src/model.js").Tensor>) {
    expect(Number(feeds.cache.data[0])).toBe(calls);
    const offset=100*calls++;
    return {
      log_probs:{type:"float32" as const,dims:[1,8,3],data:new Float32Array(24)},
      new_cache:{type:"float32" as const,dims:[1],data:new Float32Array([calls])},
      pre:{type:"float32" as const,dims:[16,1,2],data:Float32Array.from({length:32},(_,i)=>i%2?0:2*(offset+Math.floor(i/2)))},
      post:{type:"float32" as const,dims:[16,1,2],data:Float32Array.from({length:32},(_,i)=>i%2?0:offset+Math.floor(i/2))},
    };
  }};
  const result=await residualObservations(extractor,manifest,fitted,new Float32Array(16000));
  expect(result.residual).toHaveLength(23);
  expect(result.control).toHaveLength(23);
  expect(result.residual[0].slice(0,8)).toEqual(Array(8).fill(.5));
  for(const v of result.residual[0].slice(8))expect(v).toBeCloseTo(1-Math.tanh(1)/2,12);
  expect(result.residual[0][8]).not.toBeCloseTo(1-Math.tanh(.5),8);
  expect(result.control[8][0]).toBe(100.5);
  expect(calls).toBe(3);
});

it("repeats slower paired observations and rejects fractional clock ratios",async()=> {
  const manifest={tokens:"بت",fbank:{mels:80,padValue:-23},window:39,shift:32,states:[]} as import("../src/model.js").ModelManifest;
  const backend=(rows:number)=>({name:"frozen",async release(){},async run(){return {
    log_probs:{type:"float32" as const,dims:[1,8,3],data:new Float32Array(24)},
    pre:{type:"float32" as const,dims:[rows,1,2],data:Float32Array.from({length:rows*2},(_,i)=>i%2?0:Math.floor(i/2))},
    post:{type:"float32" as const,dims:[rows,1,2],data:Float32Array.from({length:rows*2},(_,i)=>i%2?0:Math.floor(i/2))},
  };}});
  const result=await residualObservations(backend(4),manifest,fitted,new Float32Array(16000));
  expect(result.control.slice(0,8).map(r=>r[0])).toEqual([0,0,1,1,2,2,3,3]);
  expect(result.residual[0]).toEqual(result.residual[1]);
  await expect(residualObservations(backend(6),manifest,fitted,new Float32Array(16000)))
    .rejects.toThrow("Invalid paired activation geometry");
});
