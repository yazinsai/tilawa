import { describe, expect, it } from "vitest";
import { carryFixedEffectEvidence, createFixedEffectStream, featureBoundaryScore, featureComparison, featureObservations, featureReversal, validateFixedEffectModel } from "../src/fixed-effect-phone.js";
import type { FixedEffectModel, FeatureHead } from "../src/fixed-effect-phone.js";
import type { InferenceBackend, ModelManifest, Tensor } from "../src/model.js";
import { createListener } from "../src/listener.js";

const head=(bias:number[]):FeatureHead=>({hidden:Array.from({length:48},()=>Array(32).fill(0)),hiddenBias:Array(48).fill(0),
  output:bias.map(()=>Array(48).fill(0)),outputBias:bias});
const model=():FixedEffectModel=>({schema:1,inventory:"test",tokens:"بت",dimensions:32,primarySha256:"a".repeat(64),
  featureOutput:"encoder",projection:Array.from({length:32},(_,i)=>Array.from({length:32},(_,j)=>Number(i===j))),
  center:Array(32).fill(0),scale:Array(32).fill(1),fixedEffect:head([-2,2,-2]),control:head([-2,-2,2])});
const candidate={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:4,toFrame:8};
const features=Array.from({length:4},()=>Array(32).fill(0));

describe("matched acoustic pronunciation heads",()=>{
  it("scores both complete candidates symmetrically and requires the fixedEffect/control disagreement",()=>{
    const c=featureComparison(model(),features,candidate)!;
    expect(featureReversal(c)).toBe(true);
    const reverse=featureComparison(model(),features,{...candidate,written:candidate.rival,rival:candidate.written})!;
    expect(reverse.fixedEffectWritten).toBe(c.fixedEffectRival);
    expect(reverse.fixedEffectRival).toBe(c.fixedEffectWritten);
    expect(reverse.controlWritten).toBe(c.controlRival);
    expect(featureReversal(reverse)).toBe(false);
    expect(featureReversal({...c,controlWritten:c.controlRival})).toBe(false);
    expect(featureReversal({...c,fixedEffectWritten:NaN})).toBe(false);
  });
  it("sums only paths spelling the complete pronunciation, with blanks between repeats",()=>{
    // Three one-phone paths: blank/phone, phone/blank and phone/phone.
    const rows=Array.from({length:2},()=>[Math.log(.5),Math.log(.3),Math.log(.2)]);
    expect(Math.exp(featureBoundaryScore(rows,[1]))).toBeCloseTo(.15+.15+.09,10);
    expect(Math.exp(featureBoundaryScore(rows,[1,2]))).toBeCloseTo(.3*.2,10);
    expect(featureBoundaryScore(rows,[1,1])).toBe(-Infinity);
    expect(Math.exp(featureBoundaryScore([...rows,rows[0]],[1,1]))).toBeCloseTo(.3*.5*.3,10);
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
    const enriched=carryFixedEffectEvidence(chunks,[c]);
    expect(enriched[0].posteriors).toBe(chunks[0].posteriors);
    expect(enriched[0].phonemes[0]).toEqual({...chunks[0].phonemes[0],fixedEffectPhoneVetoes:[c]});
    expect(chunks[0].phonemes[0]).not.toHaveProperty("fixedEffectPhoneVetoes");
  });
  it("rejects unmatched inventories, primary weights and head dimensions",()=>{
    const m=model(),manifest={inventory:m.inventory,tokens:m.tokens,sha256:m.primarySha256} as ModelManifest;
    expect(validateFixedEffectModel(m,manifest)).toBe(m);
    expect(()=>validateFixedEffectModel({...m,primarySha256:"b".repeat(64)},manifest)).toThrow();
    expect(()=>validateFixedEffectModel({...m,fixedEffect:head([0,0])},manifest)).toThrow();
    expect(()=>validateFixedEffectModel({...m,scale:Array(32).fill(0)},manifest)).toThrow();
    expect(()=>validateFixedEffectModel({...m,projection:Array.from({length:32},()=>[])},manifest)).toThrow();
  });
  it("uses the primary streaming windows, cache clock and final-frame truncation",async()=>{
    const manifest={inventory:"test",tokens:"بت",sampleRate:16000,frameMs:40,window:45,shift:32,chunk:16,
      fbank:{mels:80,padValue:-23.0259},states:[{name:"count",shape:[1],dtype:"int64"}]} as ModelManifest;
    const captures:unknown[][]=[[],[]];
    const backend=(which:number):InferenceBackend=>({name:"test",async release(){},async run(feeds){
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
  it("makes a live comparison before evidence delivery, preserving primary tensors and independent caches",async()=>{
    const manifest={tokens:"بت",states:[{name:"count",shape:[1],dtype:"int64"}]} as ModelManifest;
    const original={log_probs:{type:"float32",dims:[1,4,3],data:new Float32Array(12)}} satisfies Record<string,Tensor>;
    const primary:InferenceBackend={name:"primary",async release(){},async run(){return original;}};
    const counts:number[]=[];
    const side:InferenceBackend={name:"side",async release(){},async run(feeds){
      const count=Number(feeds.count.data[0]);counts.push(count);
      return {...original,encoder:{type:"float32",dims:[1,4,32],data:new Float32Array(128)},
        new_count:{type:"int64",dims:[1],data:new BigInt64Array([BigInt(count+1)])}};
    }};
    const stream=createFixedEffectStream(primary,{backend:side,model:model()},manifest);
    expect(stream.compare(candidate)).toBe(true); // No acoustics yet.
    const feeds={x:{type:"float32",dims:[1],data:new Float32Array(1)},
      count:{type:"int64",dims:[1],data:new BigInt64Array([99n])}} satisfies Record<string,Tensor>;
    expect(await stream.backend.run(feeds)).toBe(original);
    expect(await stream.backend.run(feeds)).toBe(original);
    expect(counts).toEqual([0,1]);
    expect(stream.compare(candidate)).toBe(false);
    expect(stream.certificates[0]).toMatchObject(candidate);
    expect(stream.compare({...candidate,toFrame:9})).toBe(true);
    const next=createFixedEffectStream(primary,{backend:side,model:model()},manifest);
    await next.backend.run(feeds);
    expect(counts).toEqual([0,1,0]);
    expect(next.certificates).toEqual([]);
  });
});
