import { describe, expect, it } from "vitest";
import { carryInvariantEvidence, featureBoundaryScore, featureComparison, featureObservations, featureReversal, recordingDescriptor, validateInvariantModel } from "../src/invariant-phone.js";
import type { InvariantModel, FeatureHead } from "../src/invariant-phone.js";
import type { InferenceBackend, ModelManifest, Tensor } from "../src/model.js";
import { createListener } from "../src/listener.js";

const head=(bias:number[]):FeatureHead=>({hidden:Array.from({length:48},()=>Array(32).fill(0)),hiddenBias:Array(48).fill(0),
  output:bias.map(()=>Array(48).fill(0)),outputBias:bias});
const model=():InvariantModel=>({schema:1,inventory:"test",tokens:"بت",dimensions:32,primarySha256:"a".repeat(64),
  featureOutput:"encoder",projection:Array.from({length:32},(_,i)=>Array.from({length:32},(_,j)=>Number(i===j))),
  center:Array(32).fill(0),scale:Array(32).fill(1),invariant:head([-2,2,-2]),control:head([-2,-2,2])});
const candidate={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:4,toFrame:8};
const features=Array.from({length:4},()=>Array(32).fill(0));

describe("matched acoustic pronunciation heads",()=>{
  it("extracts finite recording descriptors from acoustics without candidate text",()=>{
    const audio=Float32Array.from({length:16000},(_,i)=>.2*Math.sin(i*.08));
    const descriptors=recordingDescriptor(audio);
    expect(descriptors).toHaveLength(8);
    expect(descriptors.every(Number.isFinite)).toBe(true);
    expect(recordingDescriptor(audio)).toEqual(descriptors);
    expect(()=>recordingDescriptor(new Float32Array(0))).toThrow();
  });
  it("scores both complete candidates symmetrically and requires the invariant/control disagreement",()=>{
    const c=featureComparison(model(),features,candidate)!;
    expect(featureReversal(c)).toBe(true);
    const reverse=featureComparison(model(),features,{...candidate,written:candidate.rival,rival:candidate.written})!;
    expect(reverse.invariantWritten).toBe(c.invariantRival);
    expect(reverse.invariantRival).toBe(c.invariantWritten);
    expect(reverse.controlWritten).toBe(c.controlRival);
    expect(featureReversal(reverse)).toBe(false);
    expect(featureReversal({...c,controlWritten:c.controlRival})).toBe(false);
    expect(featureReversal({...c,invariantWritten:NaN})).toBe(false);
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
    expect(featureComparison(model(),features,{...candidate,toFrame:9})).toBeNull();
  });
  it("carries only metadata, retaining primary phones and posterior rows",()=>{
    const chunks=[{inventory:"test",frameMs:40,phonemes:[{p:"ت",start:4,end:7,conf:.99}],
      posteriors:{k:1,startFrame:4,nFrames:4,ids:[2,2,2,2],probs:[.99,.99,.99,.99]}}];
    const c=featureComparison(model(),features,candidate)!;
    const enriched=carryInvariantEvidence(chunks,[c]);
    expect(enriched[0].posteriors).toBe(chunks[0].posteriors);
    expect(enriched[0].phonemes[0]).toEqual({...chunks[0].phonemes[0],invariantPhoneVetoes:[c]});
    expect(chunks[0].phonemes[0]).not.toHaveProperty("invariantPhoneVetoes");
  });
  it("rejects unmatched inventories, primary weights and head dimensions",()=>{
    const m=model(),manifest={inventory:m.inventory,tokens:m.tokens,sha256:m.primarySha256} as ModelManifest;
    expect(validateInvariantModel(m,manifest)).toBe(m);
    expect(()=>validateInvariantModel({...m,primarySha256:"b".repeat(64)},manifest)).toThrow();
    expect(()=>validateInvariantModel({...m,invariant:head([0,0])},manifest)).toThrow();
    expect(()=>validateInvariantModel({...m,scale:Array(32).fill(0)},manifest)).toThrow();
    expect(()=>validateInvariantModel({...m,projection:Array.from({length:32},()=>[])},manifest)).toThrow();
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
