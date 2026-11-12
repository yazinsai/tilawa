import { describe, expect, it } from "vitest";
import type { ModelManifest } from "../src/model.js";
import { carrySparseFactorEvidence, eligibleSparseFactorTake, sparseFactorBoundaryScore, sparseFactorComparison,
  sparseFactorObservations, sparseFactorReversal, sparseFactorVerdictRemovals, validateSparseFactorModel,
  type SparseFactorModel } from "../src/sparse-factor.js";

const manifest = { inventory:"test", tokens:"ab", sha256:"parent" } as ModelManifest;
function model(): SparseFactorModel {
  const arm = (sign:number) => ({encoder:[[1],[-1]],encoderBias:[0,0],
    output:[[0,0],[sign,0],[-sign,0]],outputBias:[-1,0,0]});
  return {schema:1,inventory:"test",tokens:"ab",primarySha256:"parent",width:1,hidden:2,
    featureOutput:"features",center:[0],scale:[1],sparseFactor:arm(1),dense:arm(-1),control:arm(-1)};
}
const candidate = {word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:3,toFrame:5};

describe("matched sparse-factor acoustic evidence",()=> {
  it("requires equal overcomplete inference capacity, finite weights and exact primary identity",()=> {
    expect(validateSparseFactorModel(model(),manifest)).toEqual(model());
    expect(()=>validateSparseFactorModel({...model(),hidden:1},manifest)).toThrow();
    expect(()=>validateSparseFactorModel({...model(),primarySha256:"other"},manifest)).toThrow();
    expect(()=>validateSparseFactorModel({...model(),scale:[0]},manifest)).toThrow();
    const bad=model();bad.dense.encoder[0][0]=NaN;
    expect(()=>validateSparseFactorModel(bad,manifest)).toThrow();
  });
  it("requires positive written evidence under the sparse arm and rival preference in both controls",()=> {
    const c=sparseFactorComparison(model(),[[1],[1]],candidate)!;
    expect(sparseFactorReversal(c)).toBe(true);
    expect(sparseFactorReversal({...c,controlWritten:c.controlRival})).toBe(false);
    expect(sparseFactorReversal({...c,denseWritten:c.denseRival})).toBe(false);
    expect(sparseFactorReversal({...c,sparseFactorWritten:NaN})).toBe(false);
    expect(sparseFactorComparison(model(),[[1]],candidate)).toBeNull();
    expect(sparseFactorComparison(model(),[[1],[1]],{...candidate,rival:"x"})).toBeNull();
    expect(sparseFactorComparison(model(),[[1],[NaN]],candidate)).toBeNull();
  });
  it("keeps primary tokens and posteriors unchanged through JSON transport",()=> {
    const chunks=[{inventory:"test",frameMs:40,phonemes:[{p:"a",start:0,end:1,conf:.9}],
      posteriors:{k:1,startFrame:0,nFrames:2,ids:[1,1],probs:[.9,.9]}}];
    const c=sparseFactorComparison(model(),[[1],[1]],candidate)!;
    const enriched=JSON.parse(JSON.stringify(carrySparseFactorEvidence(chunks,[c])));
    expect(enriched[0].posteriors).toEqual(chunks[0].posteriors);
    const {sparseFactorVetoes,...phone}=enriched[0].phonemes[0];
    expect(phone).toEqual(chunks[0].phonemes[0]);
    expect(sparseFactorVetoes).toEqual([c]);
  });
  it("excludes held-out speakers, edited takes and repeats from canonical CTC supervision",()=> {
    const r={kind:"none",split:"train",speaker:"train",events:[],non_errors:[]};
    expect(eligibleSparseFactorTake(r,new Set())).toBe(true);
    expect(eligibleSparseFactorTake(r,new Set(["train"]))).toBe(false);
    expect(eligibleSparseFactorTake({...r,split:"eval"},new Set())).toBe(false);
    expect(eligibleSparseFactorTake({...r,events:[{}]},new Set())).toBe(false);
    expect(eligibleSparseFactorTake({...r,non_errors:[{}]},new Set())).toBe(false);
    expect(eligibleSparseFactorTake({...r,excluded:"audit rejected"},new Set())).toBe(false);
  });
  it("requires completion and blank separation between repeated phones",()=> {
    expect(sparseFactorBoundaryScore([[0,0,0]],[1,2])).toBe(-Infinity);
    expect(sparseFactorBoundaryScore([[0,0,0],[0,0,0]],[1,1])).toBe(-Infinity);
    expect(sparseFactorBoundaryScore([[0,0,0],[0,0,0],[0,0,0]],[1,1])).toBeCloseTo(0);
  });
  it("matches exhaustive complete-path marginalization with the shared timing prior",()=> {
    const values=[[.1,.3,.2],[.5,.4,.1],[.2,.8,.4],[.4,.2,.7]];
    for(const ids of [[1],[2],[1,2],[1,1]]) {
      const labels=[0,...ids.flatMap(p=>[p,0])];
      const edges=labels.map((p,s)=>[s,...(s+1<labels.length?[s+1]:[]),
        ...(s+2<labels.length&&labels[s+2]&&labels[s+2]!==p?[s+2]:[])]);
      let likelihood=0,timing=0;
      function visit(t:number,s:number,weight:number,score:number) {
        score+=values[t][labels[s]];
        if(t===values.length-1) {if(s>=labels.length-2){timing+=weight;likelihood+=weight*Math.exp(score);}return;}
        for(const next of edges[s])visit(t+1,next,weight/edges[s].length,score);
      }
      visit(0,0,1,0);visit(0,1,1,0);
      expect(sparseFactorBoundaryScore(values,ids)).toBeCloseTo(Math.log(likelihood/timing),12);
    }
  });
  it("retracts only the live occurrence removed by an otherwise identical replay",()=> {
    const warning={type:"checker",v:1,id:"one",timeMs:4000,state:"confirmed",errorClass:"wrong_word",
      location:candidate.word,expected:"a",observed:"b",confidence:1} as import("@tilawa/core").CheckerEvent;
    const repeat={...warning,id:"two",timeMs:8000}, graph={...warning,timeMs:9000};
    expect(sparseFactorVerdictRemovals([warning,repeat,graph],[warning,repeat,graph],[repeat,graph])).toEqual([warning]);
    expect(sparseFactorVerdictRemovals([warning],[{...warning,timeMs:4001}],[])).toEqual([]);
  });
  it("replays native features with separate recurrent state and the unchanged clock",async()=> {
    let calls=0;
    const geometry={tokens:"ab",fbank:{mels:80,padValue:-23},window:45,shift:32,
      states:[{name:"cache",shape:[1],dtype:"float32"}]} as ModelManifest;
    const backend={name:"extractor",async release(){},async run(feeds:Record<string,import("../src/model.js").Tensor>) {
      expect(Number(feeds.cache.data[0])).toBe(calls);
      const offset=100*calls++;
      return {log_probs:{type:"float32" as const,dims:[1,8,3],data:new Float32Array(24)},
        new_cache:{type:"float32" as const,dims:[1],data:new Float32Array([calls])},
        features:{type:"float32" as const,dims:[1,8,1],data:Float32Array.from({length:8},(_,i)=>offset+i)}};
    }};
    const rows=await sparseFactorObservations(backend,geometry,model(),new Float32Array(16000));
    expect(rows.features).toHaveLength(23);
    expect(rows.features[0]).toEqual([0]);
    expect(rows.features[8]).toEqual([100]);
    expect(calls).toBe(3);
  });
});
