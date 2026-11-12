import { describe, expect, it } from "vitest";
import { Reservoir, ctcScore, eligibleFadingMemoryTake, fadingMemoryObservations, memoryComparison, readouts, shuffledOrder, validateFadingMemoryModel } from "../src/fading-memory.js";
import type { FadingMemoryModel } from "../src/fading-memory.js";
import type { InferenceBackend, ModelManifest } from "../src/model.js";

const model: FadingMemoryModel = {
  schema:1, inventory:"test", tokens:"ab", primarySha256:"test", width:1,
  featureOutput:"h", logitOutput:"z", center:[0], scale:[1], input:[[1]], rho:.8,
  memory:{weight:[[0,0],[0,1],[0,-1]],bias:[0,0,0]},
  instant:{weight:[[0,0],[0,1],[0,-1]],bias:[0,0,0]},
  shuffled:{weight:[[0,0],[0,1],[0,-1]],bias:[0,0,0]},
};

describe("fixed fading acoustic memory",()=> {
  it("retains history across chunk boundaries and contracts when new inputs agree",()=> {
    const continuous=new Reservoir([[1],[.5]],.8), split=new Reservoir([[1],[.5]],.8);
    const input=[[1],[0],[-.5],[.2],[0]], expected=input.map(x=>continuous.push(x));
    const actual=[...input.slice(0,2).map(x=>split.push(x)),...input.slice(2).map(x=>split.push(x))];
    expect(actual).toEqual(expected);
    const a=new Reservoir([[1],[.5]],.8), b=new Reservoir([[1],[.5]],.8);
    const first=a.push([1]), second=b.push([0]);
    let distance=Math.hypot(...first.map((v,i)=>v-second[i]));
    for(let t=0;t<8;t++) {
      const x=a.push([.2]), y=b.push([.2]);
      const next=Math.hypot(...x.map((v,i)=>v-y[i]));
      expect(next).toBeLessThanOrEqual(.8*distance+1e-12);distance=next;
    }
  });
  it("makes ordered history available while the instant arm depends only on the current frame",()=> {
    const withHistory=readouts(model,[[1],[0]]), withoutHistory=readouts(model,[[0],[0]]);
    expect(withHistory[0][1]).not.toEqual(withoutHistory[0][1]);
    expect(withHistory[1][1]).toEqual(withoutHistory[1][1]);
  });
  it("shuffles each frame exactly once and restores its current-frame alignment",()=> {
    const order=shuffledOrder(13);
    expect([...order].sort((a,b)=>a-b)).toEqual(Array.from({length:13},(_,i)=>i));
    expect(order).toEqual(shuffledOrder(13));
    const noMemory={...model,rho:0};
    expect(readouts(noMemory,[[1],[.2],[-1]])[2]).toEqual(readouts(noMemory,[[1],[.2],[-1]])[1]);
  });
  it("rejects noncontractive and mismatched readouts",()=> {
    const manifest={inventory:"test",tokens:"ab",sha256:"test"} as ModelManifest;
    expect(validateFadingMemoryModel(model,manifest)).toEqual(model);
    expect(()=>validateFadingMemoryModel({...model,rho:1},manifest)).toThrow();
    expect(()=>validateFadingMemoryModel({...model,shuffled:{weight:[],bias:[]}},manifest)).toThrow();
  });
  it("excludes repeats, edits, held-out speakers and non-training takes",()=> {
    const r={kind:"none",split:"train",speaker:"training"};
    expect(eligibleFadingMemoryTake(r,new Set())).toBe(true);
    for(const edit of [{non_errors:[{}]},{events:[{}]},{kind:"wrong_word"},{split:"validation"},{exclude:true}])
      expect(eligibleFadingMemoryTake({...r,...edit},new Set())).toBe(false);
    expect(eligibleFadingMemoryTake(r,new Set(["training"]))).toBe(false);
  });
  it("carries independent encoder caches through every full-recording window",async()=> {
    const starts:number[]=[];
    const backend:InferenceBackend={name:"fixture",release:async()=>{},run:async feeds=> {
      const state=Number(feeds.cache.data[0]);starts.push(state);
      return {
        log_probs:{type:"float32",dims:[1,2,3],data:new Float32Array(6)},
        h:{type:"float32",dims:[1,2,1],data:new Float32Array([2*state,2*state+1])},
        z:{type:"float32",dims:[1,2,3],data:new Float32Array(6)},
        new_cache:{type:"float32",dims:[1],data:new Float32Array([state+1])},
      };
    }};
    const manifest={sampleRate:16000,frameMs:40,tokens:"ab",window:15,shift:8,
      fbank:{mels:80,padValue:-23},states:[{name:"cache",shape:[1],dtype:"float32"}]} as ModelManifest;
    const audio=Float32Array.from({length:17600},(_,i)=>Math.sin(i*.1)*.1);
    const a=await fadingMemoryObservations(backend,manifest,model,audio), n=starts.length;
    const b=await fadingMemoryObservations(backend,manifest,model,audio);
    expect(n).toBeGreaterThan(1);
    expect(starts.slice(0,n)).toEqual(Array.from({length:n},(_,i)=>i));
    expect(starts.slice(n)).toEqual(starts.slice(0,n));
    expect(a.encoder.map(r=>r[0])).toEqual(Array.from({length:a.encoder.length},(_,i)=>i));
    expect(b).toEqual(a);
    expect(a.encoder.length).toBe(a.logProbs.length);
  });
});

describe("complete-pronunciation comparisons",()=> {
  it("sums whole CTC paths and requires a blank between repeated labels",()=> {
    const rows=[[.2,.8],[.3,.7]].map(r=>r.map(Math.log));
    expect(Math.exp(ctcScore(rows,[1]))).toBeCloseTo(.8*.7+.8*.3+.2*.7,12);
    expect(ctcScore(rows,[1,1])).toBe(-Infinity);
    expect(Math.exp(ctcScore([...rows,rows[0]],[1,1]))).toBeCloseTo(.8*.3*.8,12);
  });
  it("requires finite written preference from memory and rival preference from both controls",()=> {
    const written=[[.01,.89,.1]].map(r=>r.map(Math.log));
    const rival=[[.01,.1,.89]].map(r=>r.map(Math.log));
    const c={word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:0,toFrame:1};
    expect(memoryComparison(model,[written,rival,rival],c)?.veto).toBe(true);
    expect(memoryComparison(model,[written,rival,written],c)?.veto).toBe(false);
    expect(memoryComparison(model,[written,written,rival],c)?.veto).toBe(false);
    expect(memoryComparison(model,[written,rival,rival],{...c,written:"c"})?.veto).toBe(false);
    expect(memoryComparison(model,[written,rival,rival],{...c,toFrame:2})).toBeNull();
  });
});
