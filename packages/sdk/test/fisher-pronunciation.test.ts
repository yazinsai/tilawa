import { describe, expect, it } from "vitest";
import { fisherComparison, eligibleFisherTake, fisherObservations, fisherBoundaryScore, fisherFrame, fisherSegment, fisherReversal, validateFisherModel, fisherVerdictRemovals, carryFisherEvidence, type FisherModel } from "../src/fisher-pronunciation.js";
import type { ModelManifest } from "../src/model.js";
import type { CheckerEvent, FisherPronunciationEvidence, PhonemeEvidence } from "@tilawa/core";

const model: FisherModel = {
  schema:1,inventory:"test",tokens:"بت",width:2,featureOutput:"h",logitOutput:"z",primarySha256:"a".repeat(64),parityWindows:8,
  center:[0,0],projection:[[1,0],[0,1]],dimensions:2,maxDuration:3,
  mixture:[{weight:1,mean:[0,0],variance:[1,1]}],
  fisherProjection:[[1,0,0,0],[0,0,1,0]],controlProjection:[[1,0,0,0,0],[0,0,1,0,0]],
  fisher:{phones:Array.from({length:3},()=>({mean:[0,0],variance:[1,1],count:1}))},
  control:{phones:Array.from({length:3},()=>({mean:[0,0],variance:[1,1],count:1}))},
};
describe("Fisher segment pronunciation evidence",()=>{
  it("retains signed mean and variance departures, rather than a phone posterior",()=>{
    expect(fisherFrame(model,[1,0])).toEqual([1,0,0,-1/Math.sqrt(2)]);
    expect(fisherFrame(model,[-1,0])).toEqual([-1,0,0,-1/Math.sqrt(2)]);
    expect(fisherFrame(model,[0,0])).toEqual([0,0,-1/Math.sqrt(2),-1/Math.sqrt(2)]);
  });
  it("distinguishes multimodal occupancy even when mean and covariance are equal",()=>{
    const m={...model,mixture:[{weight:.5,mean:[-1,0],variance:[.25,1]},{weight:.5,mean:[1,0],variance:[.25,1]}],
      fisherProjection:[[1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0]]};
    const a=fisherSegment(m,[[-1,0],[1,0],[-1,0],[1,0]]);
    const b=fisherSegment(m,[[-Math.sqrt(2),0],[0,0],[0,0],[Math.sqrt(2),0]]);
    a.control.forEach((v,i)=>expect(v).toBeCloseTo(b.control[i],10));
    expect(a.fisher).not.toEqual(b.fisher);
  });
  it("conditions duration paths on completion, including repeated phones and empty gaps",()=>{
    const values=Array.from({length:4},(_,t)=>Array.from({length:t+1},()=>[0,0,0]));
    expect(fisherBoundaryScore(values,[1,1],3)).toBeCloseTo(0,12);
    expect(fisherBoundaryScore(values,[1,2,1,2,1],3)).toBe(-Infinity);
    const changed=values.map(row=>row.map(v=>[v[0],-10,v[2]]));
    expect(fisherBoundaryScore(changed,[1],3)).toBeLessThan(fisherBoundaryScore(changed,[2],3));
  });
  it("accepts only finite strict reversals and binds to the primary bundle",()=>{
    const c={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:0,toFrame:1,
      fisherWritten:1,fisherRival:0,controlWritten:0,controlRival:1} satisfies FisherPronunciationEvidence;
    expect(fisherReversal(c)).toBe(true);
    expect(fisherReversal({...c,fisherWritten:0})).toBe(false);
    expect(fisherReversal({...c,controlWritten:1})).toBe(false);
    expect(fisherReversal({...c,fisherWritten:Infinity})).toBe(false);
    const manifest={inventory:model.inventory,tokens:model.tokens,sha256:model.primarySha256} as ModelManifest;
    expect(validateFisherModel(model,manifest)).toBe(model);
    expect(()=>validateFisherModel({...model,primarySha256:"b".repeat(64)},manifest)).toThrow();
    expect(()=>validateFisherModel({...model,mixture:[{weight:1,mean:[0,0],variance:[0,1]}]},manifest)).toThrow();
  });
  it("carries certificates without changing tokens and removes only exact live occurrences",()=>{
    const c={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:0,toFrame:1,
      fisherWritten:1,fisherRival:0,controlWritten:0,controlRival:1} satisfies FisherPronunciationEvidence;
    const chunks=[{phonemes:[{p:"ب",start:0,end:1,conf:.9}]}] as PhonemeEvidence[];
    const enriched=carryFisherEvidence(chunks,[c]);
    expect(chunks[0].phonemes[0].fisherPronunciationVetoes).toBeUndefined();
    expect(enriched[0].phonemes[0]).toEqual({...chunks[0].phonemes[0],fisherPronunciationVetoes:[c]});
    const e={id:"same",errorClass:"wrong_word",timeMs:40,location:c.word,expected:c.written,observed:c.rival} as CheckerEvent;
    const later={...e,timeMs:80};
    expect(fisherVerdictRemovals([e,later],[e,later],[later])).toEqual([e]);
  });
  it("excludes held-out speakers, exclusions, planted edits and repeats from fitting",()=>{
    const row={kind:"none",split:"train",speaker:"training",events:[],non_errors:[]};
    expect(eligibleFisherTake(row,new Set())).toBe(true);
    expect(eligibleFisherTake(row,new Set(["training"]))).toBe(false);
    expect(eligibleFisherTake({...row,split:"eval"},new Set())).toBe(false);
    expect(eligibleFisherTake({...row,events:[{}]},new Set())).toBe(false);
    expect(eligibleFisherTake({...row,non_errors:[{}]},new Set())).toBe(false);
    expect(eligibleFisherTake({...row,excluded:"rejected"},new Set())).toBe(false);
  });
  it("extracts the native acoustic clock with independent recurrent state",async()=>{
    let calls=0;
    const manifest={tokens:"بت",fbank:{mels:80,padValue:-23},window:45,shift:32,
      states:[{name:"cache",shape:[1],dtype:"float32"}]} as ModelManifest;
    const backend={name:"extractor",async release(){},async run(feeds:Record<string,import("../src/model.js").Tensor>){
      expect(Number(feeds.cache.data[0])).toBe(calls);
      const offset=100*calls++;
      return {
        log_probs:{type:"float32" as const,dims:[1,8,3],data:new Float32Array(24)},
        new_cache:{type:"float32" as const,dims:[1],data:new Float32Array([calls])},
        h:{type:"float32" as const,dims:[1,8,2],data:Float32Array.from({length:16},(_,i)=>offset+i)},
      };
    }};
    const rows=await fisherObservations(backend,manifest,model,new Float32Array(16000));
    expect(rows.raw).toHaveLength(23);expect(rows.logProbs).toHaveLength(23);
    expect(rows.raw[0]).toEqual([0,1]);expect(rows.raw[8]).toEqual([100,101]);
    expect(calls).toBe(3);
  });

  it("scores both complete pronunciations and abstains for missing phones or frames",()=>{
    const m={...model,maxDuration:1,
      fisher:{phones:[{mean:[0,0],variance:[1,1],count:1},{mean:[1,0],variance:[.01,.01],count:1},{mean:[-1,0],variance:[.01,.01],count:1}]},
      control:{phones:[{mean:[0,0],variance:[1,1],count:1},{mean:[-1,0],variance:[.01,.01],count:1},{mean:[1,0],variance:[.01,.01],count:1}]}};
    const c={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:0,toFrame:1};
    const result=fisherComparison(m,[[1,0]],c)!;
    expect(fisherReversal(result)).toBe(true);
    expect(fisherComparison(m,[[1,0]],{...c,rival:"x"})).toBe(null);
    expect(fisherComparison(m,[[1,0]],{...c,written:"بت"})).toBe(null);
    expect(fisherComparison(m,[[1,0]],{...c,toFrame:2})).toBe(null);
  });

});
