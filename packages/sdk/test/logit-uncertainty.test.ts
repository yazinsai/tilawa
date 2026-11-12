import { describe, expect, it } from "vitest";
import { carryLogitUncertaintyEvidence, featureBoundaryScore, featureComparison, featureObservations,
  featureReversal, integratedBoundaryScore, normalQuadrature, validateLogitUncertaintyModel } from "../src/logit-uncertainty.js";
import type { LogitUncertaintyModel, FeatureHead } from "../src/logit-uncertainty.js";
import type { InferenceBackend, ModelManifest, Tensor } from "../src/model.js";
import { createListener } from "../src/listener.js";

const head=(bias:number[]):FeatureHead=>({hidden:Array.from({length:48},()=>Array(32).fill(0)),hiddenBias:Array(48).fill(0),
  output:bias.map(()=>Array(48).fill(0)),outputBias:bias});
const model=():LogitUncertaintyModel=>({schema:1,inventory:"test",tokens:"بت",dimensions:32,primarySha256:"a".repeat(64),
  featureOutput:"encoder",projection:Array.from({length:32},(_,i)=>Array.from({length:32},(_,j)=>Number(i===j))),
  center:Array(32).fill(0),scale:Array(32).fill(1),rank:1,extent:4,quadrature:5,
  uncertainty:head([1,0,.1,0,4,0]),control:head([1,0,.1,0,0,0])});
const candidate={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:4,toFrame:8};
const features=Array.from({length:4},()=>Array(32).fill(0));
const normalize=(row:number[])=>{const z=Math.log(row.reduce((s,v)=>s+Math.exp(v),0));return row.map(v=>v-z);};
/** Exhaustively enumerate emitted symbol paths, independent of the forward recursion. */
const pathMass=(rows:number[][],ids:number[])=> {
  let mass=0;
  const visit=(path:number[],probability:number)=> {
    if(path.length===rows.length){
      const collapsed=path.filter((p,t)=>p!==0&&(t===0||p!==path[t-1]));
      if(JSON.stringify(collapsed)===JSON.stringify(ids))mass+=probability;
      return;
    }
    rows[path.length].forEach((p,i)=>visit([...path,i],probability*p));
  };
  visit([],1);return mass;
};

describe("shared logistic-normal pronunciation evidence",()=>{
  it("integrates likelihoods of full paths, reversing the same readout's mean preference",()=>{
    const c=featureComparison(model(),features,candidate)!;
    expect(featureReversal(c)).toBe(true);
    expect(Math.exp(c.meanWritten)).toBeCloseTo(.20126983785202998,12);
    expect(Math.exp(c.integratedWritten)).toBeCloseTo(.3317346709873556,12);
    const reverse=featureComparison(model(),features,{...candidate,written:candidate.rival,rival:candidate.written})!;
    expect(reverse.integratedWritten).toBe(c.integratedRival);
    expect(reverse.meanWritten).toBe(c.meanRival);
    expect(featureReversal(reverse)).toBe(false);
    expect(featureReversal({...c,meanWritten:c.meanRival})).toBe(false);
    expect(featureReversal({...c,integratedWritten:NaN})).toBe(false);
    // The matched deterministic training control never supplies the runtime mean comparison.
    expect(featureComparison({...model(),control:head([100,-100,100,0,0,0])},features,candidate)).toEqual(c);
  });
  it("matches explicit CTC path and latent enumeration across partially overlapping blocks",()=>{
    const mean=Array.from({length:4},()=>[.2,-.3,.1]),loading=mean.map(()=>[[0,.7,-.4]]);
    const points=normalQuadrature(1,3);
    let exact=0;
    // Absolute start=3 and extent=4 means one frame in the first block, three in the second.
    for(const a of points)for(const b of points){
      const rows=mean.map((row,t)=>normalize(row.map((v,p)=>v+(t===0?a:b).z[0]*loading[t][0][p])).map(Math.exp));
      exact+=a.weight*b.weight*pathMass(rows,[1,2]);
    }
    expect(Math.exp(integratedBoundaryScore(mean,loading,[1,2],4,3,3))).toBeCloseTo(exact,12);
  });
  it("keeps nearby latent perturbations shared rather than averaging emissions per frame",()=>{
    const mean=Array.from({length:4},()=>[1,0,.1]),loading=mean.map(()=>[[0,4,0]]);
    const points=normalQuadrature(1,5);
    const average=mean.map(row=>row.map((_,p)=>points.reduce((s,q)=>
      s+q.weight*Math.exp(normalize(row.map((v,i)=>v+q.z[0]*loading[0][0][i]))[p]),0)));
    const shared=Math.exp(integratedBoundaryScore(mean,loading,[1],4,0,5));
    expect(Math.abs(shared-pathMass(average,[1]))).toBeGreaterThan(.02);
  });
  it("reduces zero covariance to ordinary complete CTC and requires blanks between repeats",()=>{
    const logits=[[.1,.2,-.1],[.3,-.4,.2],[.1,.2,-.1]],mean=logits.map(normalize);
    const zero=logits.map(()=>[[0,0,0],[0,0,0]]);
    for(const ids of [[1],[1,2],[1,1]]){
      expect(featureBoundaryScore(mean,ids)).toBeCloseTo(Math.log(pathMass(mean.map(r=>r.map(Math.exp)),ids)),12);
      expect(integratedBoundaryScore(logits,zero,ids,4,1,5)).toBeCloseTo(featureBoundaryScore(mean,ids),12);
    }
    expect(featureBoundaryScore(mean.slice(0,2),[1,1])).toBe(-Infinity);
    expect(integratedBoundaryScore(logits.slice(0,2),zero.slice(0,2),[1,1],4,0,5)).toBe(-Infinity);
  });
  it("uses Gaussian product quadrature with normalized mass and unit variance",()=>{
    for(const rank of [1,2])for(const order of [3,5] as const){
      const points=normalQuadrature(rank,order);
      expect(points.reduce((s,p)=>s+p.weight,0)).toBeCloseTo(1,12);
      for(let d=0;d<rank;d++){
        expect(points.reduce((s,p)=>s+p.weight*p.z[d],0)).toBeCloseTo(0,12);
        expect(points.reduce((s,p)=>s+p.weight*p.z[d]**2,0)).toBeCloseTo(1,12);
      }
    }
  });
  it("conditions covariance on input features",()=>{
    const m=model();m.uncertainty.hidden[0][0]=1;m.uncertainty.output[4][0]=2;
    const changed=features.map(r=>[1,...r.slice(1)]);
    const a=featureComparison(m,features,candidate)!,b=featureComparison(m,changed,candidate)!;
    expect(a.meanWritten).toBe(b.meanWritten);
    expect(a.integratedWritten).not.toBe(b.integratedWritten);
  });
  it("abstains on unknown phones, nonfinite features, incomplete or mismatched windows",()=>{
    expect(featureComparison(model(),features,{...candidate,rival:"ث"})).toBeNull();
    expect(featureComparison(model(),[[NaN,...Array(31).fill(0)]],{...candidate,toFrame:5})).toBeNull();
    expect(featureComparison(model(),features.slice(0,1),{...candidate,toFrame:5,rival:"تت"})).toBeNull();
    expect(featureComparison(model(),features,{...candidate,toFrame:9})).toBeNull();
  });
  it("carries metadata while retaining primary phones and posterior rows",()=>{
    const chunks=[{inventory:"test",frameMs:40,phonemes:[{p:"ت",start:4,end:7,conf:.99}],
      posteriors:{k:1,startFrame:4,nFrames:4,ids:[2,2,2,2],probs:[.99,.99,.99,.99]}}];
    const c=featureComparison(model(),features,candidate)!;
    const enriched=carryLogitUncertaintyEvidence(chunks,[c]);
    expect(enriched[0].posteriors).toBe(chunks[0].posteriors);
    expect(enriched[0].phonemes[0]).toEqual({...chunks[0].phonemes[0],logitUncertaintyVetoes:[c]});
    expect(chunks[0].phonemes[0]).not.toHaveProperty("logitUncertaintyVetoes");
  });
  it("rejects mismatched primary weights, head geometry and unsupported latent geometry",()=>{
    const m=model(),manifest={inventory:m.inventory,tokens:m.tokens,sha256:m.primarySha256} as ModelManifest;
    expect(validateLogitUncertaintyModel(m,manifest)).toBe(m);
    expect(()=>validateLogitUncertaintyModel({...m,primarySha256:"b".repeat(64)},manifest)).toThrow();
    expect(()=>validateLogitUncertaintyModel({...m,uncertainty:head([0,0])},manifest)).toThrow();
    expect(()=>validateLogitUncertaintyModel({...m,scale:Array(32).fill(0)},manifest)).toThrow();
    expect(()=>validateLogitUncertaintyModel({...m,projection:Array.from({length:32},()=>[])},manifest)).toThrow();
    expect(()=>validateLogitUncertaintyModel({...m,rank:3},manifest)).toThrow();
    expect(()=>validateLogitUncertaintyModel({...m,extent:0},manifest)).toThrow();
  });
});

describe("frozen feature stream parity",()=>{
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
