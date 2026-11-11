import { describe, expect, it } from "vitest";
import { carryPairwiseEvidence, featureBoundaryScore, featureComparison, featureObservations, featureReversal, validatePairwiseModel, reconcilePhones } from "../src/pairwise-phone.js";
import type { PairwiseModel, FeatureHead } from "../src/pairwise-phone.js";
import type { InferenceBackend, ModelManifest, Tensor } from "../src/model.js";
import { createListener } from "../src/listener.js";

const head=(bias:number[],width=1):FeatureHead=>({hidden:Array.from({length:width},()=>Array(32).fill(0)),hiddenBias:Array(width).fill(0),
  output:bias.map(()=>Array(width).fill(0)),outputBias:bias,outputHasBias:true});
// Binary: 3*34+1=103; dense three-class control: 3*36+3=111.
// This fixture is deliberately unequal until the validation test constructs its matched heads.
const model=():PairwiseModel=>({schema:1,inventory:"test",tokens:"بت",dimensions:32,primarySha256:"a".repeat(64),
  featureOutput:"encoder",projection:Array.from({length:32},(_,i)=>Array.from({length:32},(_,j)=>Number(i===j))),
  center:Array(32).fill(0),scale:Array(32).fill(1),pairs:[{a:1,b:2,head:head([8],3)}],control:head([-2,-2,2],3),parameters:111});
const candidate={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:4,toFrame:8};
const features=Array.from({length:4},()=>Array(32).fill(0));

describe("matched acoustic pronunciation heads",()=>{
  it("scores both complete candidates symmetrically and requires the pairwise/control disagreement",()=>{
    const c=featureComparison(model(),features,candidate)!;
    expect(featureReversal(c)).toBe(true);
    const reverse=featureComparison(model(),features,{...candidate,written:candidate.rival,rival:candidate.written})!;
    expect(reverse.pairwiseWritten).toBe(c.pairwiseRival);
    expect(reverse.pairwiseRival).toBe(c.pairwiseWritten);
    expect(reverse.controlWritten).toBe(c.controlRival);
    expect(featureReversal(reverse)).toBe(false);
    expect(featureReversal({...c,controlWritten:c.controlRival})).toBe(false);
    expect(featureReversal({...c,pairwiseWritten:NaN})).toBe(false);
  });
  it("reconciles inconsistent cycles, preserving total probability, blank and untouched class mass",()=>{
    const control=[.2,.1,.2,.3,.2].map(Math.log),pairs=[{a:1,b:2},{a:2,b:3},{a:1,b:3}];
    const coupled=reconcilePhones(control,[2,-1,0],pairs);
    expect(coupled.every(Number.isFinite)).toBe(true);
    expect(coupled.reduce((s,v)=>s+Math.exp(v),0)).toBeCloseTo(1,12);
    expect(coupled[0]).toBe(control[0]); expect(coupled[4]).toBe(control[4]);
    expect([1,2,3].reduce((s,i)=>s+Math.exp(coupled[i]),0)).toBeCloseTo(.6,12);
    const reordered=reconcilePhones(control,[0,2,-1],[pairs[2],pairs[0],pairs[1]]);
    coupled.forEach((v,i)=>expect(reordered[i]).toBeCloseTo(v,12));
    const neutral=reconcilePhones(control,[0,0,0],pairs);
    expect(neutral[1]).toBeCloseTo(neutral[2],12);expect(neutral[2]).toBeCloseTo(neutral[3],12);
  });
  it("marginalizes timing conditioned on the entire pronunciation being emitted",()=>{
    // With identity-neutral emissions, every feasible pronunciation has the same likelihood.
    const rows=Array.from({length:8},()=>[-1,-1,-1]);
    expect(featureBoundaryScore(rows,[1])).toBeCloseTo(-8,10);
    expect(featureBoundaryScore(rows,[1,1,2])).toBeCloseTo(-8,10);
    expect(featureBoundaryScore(rows.slice(0,1),[1,1])).toBe(-Infinity);
  });
  it("abstains on unknown phones, nonfinite features and impossible complete candidates",()=>{
    expect(featureComparison(model(),features,{...candidate,rival:"ث"})).toBeNull();
    expect(featureComparison(model(),[[NaN,...Array(31).fill(0)]],candidate)).toBeNull();
    expect(featureComparison(model(),features.slice(0,1),{...candidate,rival:"تت"})).toBeNull();
  });
  it("carries only metadata, retaining primary phones and posterior rows",()=>{
    const chunks=[{inventory:"test",frameMs:40,phonemes:[{p:"ت",start:4,end:7,conf:.99}],
      posteriors:{k:1,startFrame:4,nFrames:4,ids:[2,2,2,2],probs:[.99,.99,.99,.99]}}];
    const c=featureComparison(model(),features,candidate)!;
    const enriched=carryPairwiseEvidence(chunks,[c]);
    expect(enriched[0].posteriors).toBe(chunks[0].posteriors);
    expect(enriched[0].phonemes[0]).toEqual({...chunks[0].phonemes[0],pairwisePhoneVetoes:[c]});
    expect(chunks[0].phonemes[0]).not.toHaveProperty("pairwisePhoneVetoes");
  });
  it("rejects unmatched inventories, primary weights and head dimensions",()=>{
    // Exact integer solution: binary 34*17+1 = dense control 36*16+3 = 579.
    const m=model();
    m.pairs[0].head=head([8],17); m.control=head([-2,-2,2],16); m.parameters=579;
    const manifest={inventory:m.inventory,tokens:m.tokens,sha256:m.primarySha256} as ModelManifest;
    expect(validatePairwiseModel(m,manifest)).toBe(m);
    expect(()=>validatePairwiseModel({...m,primarySha256:"b".repeat(64)},manifest)).toThrow();
    expect(()=>validatePairwiseModel({...m,pairs:[{a:1,b:2,head:head([0,0],17)}]},manifest)).toThrow();
    expect(()=>validatePairwiseModel({...m,scale:Array(32).fill(0)},manifest)).toThrow();
    expect(()=>validatePairwiseModel({...m,projection:Array.from({length:32},()=>[])},manifest)).toThrow();
  });
  it("uses the primary streaming windows, cache clock and final-frame truncation",async()=>{
    const manifest={inventory:"test",tokens:"بت",sampleRate:16000,frameMs:40,window:45,shift:32,chunk:16,
      fbank:{mels:80,padValue:-23.0259},states:[{name:"count",shape:[1],dtype:"int64"}]} as ModelManifest;
    const captures:unknown[][]=[[],[]];
    const backend=(which:number):InferenceBackend=>({name:"test",async run(feeds){
      captures[which].push(Object.fromEntries(Object.entries(feeds).map(([k,v])=>[k,Array.from(v.data,Number)])));
      const count=Number(feeds.count.data[0]),frames=8;
      const raw=Float32Array.from({length:frames*2},(_,i)=>count*frames*2+i);
      const lp=Float32Array.from({length:frames*3},(_,i)=>[-3,-1,-2][i%3]);
      return {new_count:{type:"int64",dims:[1],data:new BigInt64Array([BigInt(count+1)])},
        encoder:{type:"float32",dims:[1,frames,2],data:raw},log_probs:{type:"float32",dims:[1,frames,3],data:lp}} satisfies Record<string,Tensor>;
    }});
    const samples=Float32Array.from({length:14000},(_,i)=>.2*Math.sin(i*.08));
    const listener=createListener({manifest,backend:backend(0),supported:[]},{posteriors:true,topK:3});
    let nativeFrames=0;
    for(let s=0;s<samples.length;s+=1000)nativeFrames+=(await listener.push(samples.slice(s,s+1000),16000))?.posteriors?.nFrames??0;
    nativeFrames+=(await listener.end())?.posteriors?.nFrames??0;
    const scorer={...model(),projection:Array.from({length:32},()=>[1,0])};
    const features=await featureObservations(backend(1),manifest,scorer,samples);
    expect(captures[1]).toEqual(captures[0]);
    expect(features.observations).toHaveLength(nativeFrames);
    expect(features.observations.map(r=>r[0])).toEqual(Array.from({length:nativeFrames},(_,i)=>2*i));
  });
});
