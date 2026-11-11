import { describe, expect, it } from "vitest";
import { adaptReadout, featureBoundaryScore, featureComparison, featureObservations, featureReversal, softOccupancies,
  validateSessionReadoutModel, type SessionReadoutModel, type SupportObservation } from "../src/session-readout.js";
import type { InferenceBackend, LoadedModel, ModelManifest } from "../src/model.js";
import { createListener } from "../src/listener.js";

const log=(rows:number[][])=>rows.map(r=>r.map(Math.log));
const m:SessionReadoutModel={schema:1,inventory:"test",tokens:"بت",dimensions:1,primarySha256:"hash",
  featureOutput:"features",projection:[[1]],center:[0],scale:[1],initialization:[[0,0,0],[0,0,0]],regularizer:[1,1]};
const support:SupportObservation={pronunciation:"ب",fromFrame:0,toFrame:2,availableFrame:2,
  features:[[1],[1]],logProbs:log([[.1,.8,.1],[.2,.7,.1]])};

describe("causal episodically learned readout",()=> {
  it("matches the closed-form ridge solution and keeps its initialization unchanged",()=> {
    const before=JSON.stringify(m.initialization), w=adaptReadout(m,[support]);
    // With a constant x=[1,1], unit regularization and unit total sample mass,
    // both coordinates receive one third of the mean soft target log probability.
    const y=softOccupancies(support.logProbs,[1])!;
    const target=y.map(r=>r.map(v=>Math.log(Math.max(v,1e-6))));
    for(let p=0;p<3;p++) {
      const expected=(target[0][p]+target[1][p])/6;
      expect(w[0][p]).toBeCloseTo(expected,10);expect(w[1][p]).toBeCloseTo(expected,10);
    }
    expect(JSON.stringify(m.initialization)).toBe(before);
    expect(adaptReadout(m,[])).toEqual(m.initialization);
  });
  it("weights each accepted word equally despite different durations",()=> {
    const repeated={...support,toFrame:4,availableFrame:4,features:[[1],[1],[1],[1]],
      logProbs:log([[0.01,.98,.01],[.01,.98,.01],[.01,.98,.01],[.01,.98,.01]])};
    const a=adaptReadout(m,[support,repeated]);
    const b=adaptReadout(m,[repeated,support]);
    a.forEach((r,i)=>r.forEach((v,p)=>expect(v).toBeCloseTo(b[i][p],12)));
    const targets=[support,repeated].map(s=>softOccupancies(s.logProbs,[1])!);
    for(let p=0;p<3;p++) {
      const wordMeans=targets.map(y=>y.reduce((sum,r)=>sum+Math.log(Math.max(r[p],1e-6)),0)/y.length);
      expect(a[0][p]).toBeCloseTo((wordMeans[0]+wordMeans[1])/6,10);
    }
  });
  it("marginalizes repeated-phone boundaries and rejects incomplete pronunciation paths",()=> {
    const rows=log([[.1,.8,.1],[.8,.1,.1],[.1,.8,.1]]);
    const occupancy=softOccupancies(rows,[1,1])!;
    expect(occupancy[0][1]).toBeCloseTo(1);expect(occupancy[1][0]).toBeCloseTo(1);
    expect(occupancy[2][1]).toBeCloseTo(1);
    expect(softOccupancies(rows.slice(0,2),[1,1])).toBeNull();
    expect(featureBoundaryScore(rows.slice(0,2),[1,1])).toBe(-Infinity);
  });
  it("abstains without earlier support and rejects query or later speech in adaptation",()=> {
    const candidate={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:20,toFrame:22};
    expect(featureComparison(m,[[1],[1]],candidate,[])).toBeNull();
    expect(()=>featureComparison(m,[[1],[1]],candidate,[{...support,toFrame:14}])).toThrow("Non-causal");
    expect(()=>featureComparison(m,[[1],[1]],candidate,[{...support,availableFrame:21}])).toThrow("Non-causal");
    const result=featureComparison(m,[[1],[1]],candidate,[support])!;
    expect(result.supportFrames).toEqual([{fromFrame:0,toFrame:2,availableFrame:2}]);
    const swap=featureComparison(m,[[1],[1]],{...candidate,written:"ت",rival:"ب"},[support])!;
    expect(result.adaptedWritten).toBe(swap.adaptedRival);
    expect(result.adaptedRival).toBe(swap.adaptedWritten);
  });
  it("requires a finite strict reversal by the adapted counterpart",()=> {
    const c={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:20,toFrame:22,
      supportFrames:[{fromFrame:0,toFrame:2,availableFrame:2}],
      adaptedWritten:1,adaptedRival:0,unadaptedWritten:0,unadaptedRival:1};
    expect(featureReversal(c)).toBe(true);
    expect(featureReversal({...c,unadaptedWritten:1})).toBe(false);
    expect(featureReversal({...c,adaptedWritten:NaN})).toBe(false);
  });
  it("rejects a non-positive adaptation regularizer or mismatched encoder",()=> {
    const large={...m,dimensions:32,projection:Array.from({length:32},()=>[1]),center:Array(32).fill(0),
      scale:Array(32).fill(1),regularizer:Array(33).fill(1),initialization:Array.from({length:33},()=>[0,0,0])};
    const manifest={inventory:"test",tokens:"بت",sha256:"hash"} as ModelManifest;
    expect(validateSessionReadoutModel(large,manifest)).toEqual(large);
    expect(()=>validateSessionReadoutModel({...large,regularizer:Array(33).fill(0)},manifest)).toThrow();
    expect(()=>validateSessionReadoutModel(large,{...manifest,sha256:"other"})).toThrow();
  });
  it("extracts features on the same frame clock and cache sequence as streaming primary decoding",async()=> {
    const manifest={inventory:"test",tokens:"بت",frameMs:40,sampleRate:16000,window:31,shift:16,
      fbank:{mels:80,padValue:-23.02585},states:[{name:"count",dtype:"float32",shape:[1]}]} as ModelManifest;
    const feeds: number[][][]=[[],[]];
    const backend=(ear:number):InferenceBackend=>({name:"test",release:async()=>{},run:async input=>{
      feeds[ear].push(Array.from(input.x.data,Number));
      const count=Number(input.count.data[0]);
      return {log_probs:{type:"float32",dims:[1,4,3],data:Float32Array.from(Array.from({length:4},()=>[.1,.8,.1]).flat(),Math.log)},
        features:{type:"float32",dims:[1,4,1],data:Float32Array.from({length:4},(_,t)=>count*4+t)},
        new_count:{type:"float32",dims:[1],data:Float32Array.of(count+1)}};
    }});
    const audio=Float32Array.from({length:16000},(_,i)=>Math.sin(i*.07)*.2);
    const listener=createListener({manifest,backend:backend(0)} as LoadedModel,{posteriors:true});
    let frames=0;
    for(let at=0;at<audio.length;at+=1280)frames+=(await listener.push(audio.slice(at,at+1280),16000))?.posteriors?.nFrames??0;
    frames+=(await listener.end())?.posteriors?.nFrames??0;
    const features=await featureObservations(backend(1),manifest,m,audio);
    expect(features.observations).toHaveLength(frames);
    expect(features.logProbs).toHaveLength(frames);
    expect(features.availableFrames).toHaveLength(frames);
    expect(features.availableFrames.slice(0,4)).toEqual([9,9,9,9]);
    expect(features.availableFrames[4]).toBe(13);
    expect(features.observations.map(r=>r[0])).toEqual(Array.from({length:frames},(_,i)=>i));
    expect(feeds[1]).toEqual(feeds[0]);
  });
});
