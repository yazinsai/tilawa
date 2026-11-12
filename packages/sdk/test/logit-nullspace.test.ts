import { describe, expect, it } from "vitest";
import type { ModelManifest } from "../src/model.js";
import { carryNullspaceEvidence, eligibleNullspaceTake, nullspaceBoundaryScore, nullspaceComparison, nullspaceObservations, nullspaceReversal, nullspaceVerdictRemovals, validateNullspaceModel, type NullspaceModel } from "../src/logit-nullspace.js";

const manifest = { inventory:"test",tokens:"ab",sha256:"parent" } as ModelManifest;
function model(): NullspaceModel {
  const arm = () => ({center:[0,0,0],scale:[1,1,1],phones:[0,1,2].map(p=>({
    mean:[p,0,0],precision:[[1,0,0],[0,1,0],[0,0,1]],logDet:0,count:1,
  }))});
  return {schema:1,inventory:"test",tokens:"ab",primarySha256:"parent",dimensions:3,width:6,rank:3,nullity:3,
    featureOutput:"features",logitOutput:"logits",effectiveWeight:[[1,0,0],[0,1,0],[0,0,1],[0,0,0],[0,0,0],[0,0,0]],
    nullProjection:[[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]],nullspace:arm(),control:arm()};
}
const candidate = {word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:3,toFrame:5};

describe("final logit-nullspace acoustic evidence",()=> {
  it("requires an actual null projection tied to the deployed matrix",()=> {
    expect(validateNullspaceModel(model(),manifest)).toEqual(model());
    const leaking=model();leaking.nullProjection[0][0]=.1;
    expect(()=>validateNullspaceModel(leaking,manifest)).toThrow(/leaks/);
    const degenerate=model();degenerate.nullProjection[0]=Array(6).fill(0);
    expect(()=>validateNullspaceModel(degenerate,manifest)).toThrow(/orthonormal/);
    expect(()=>validateNullspaceModel({...model(),primarySha256:"other"},manifest)).toThrow();
  });
  it("requires the null arm to recover the written pronunciation and the matched control to favour the rival",()=> {
    const certificate=nullspaceComparison(model(),[[1,0,0],[1,0,0]],[[2,0,0],[2,0,0]],candidate)!;
    expect(nullspaceReversal(certificate)).toBe(true);
    expect(nullspaceReversal({...certificate,controlWritten:certificate.controlRival})).toBe(false);
    expect(nullspaceReversal({...certificate,nullspaceWritten:NaN})).toBe(false);
    expect(nullspaceComparison(model(),[[1,0,0]],[[2,0,0]],candidate)).toBeNull();
    expect(nullspaceComparison(model(),[[1,0,0],[1,0,0]],[[2,0,0],[2,0,0]],{...candidate,rival:"x"})).toBeNull();
  });
  it("keeps primary phones and posterior rows unchanged through JSON transport",()=> {
    const chunks=[{inventory:"test",frameMs:40,phonemes:[{p:"a",start:0,end:1,conf:.9}],posteriors:{k:1,startFrame:0,nFrames:2,ids:[1,1],probs:[.9,.9]}}];
    const certificate={...candidate,controlWritten:0,controlRival:1,nullspaceWritten:1,nullspaceRival:0};
    const enriched=JSON.parse(JSON.stringify(carryNullspaceEvidence(chunks,[certificate])));
    expect(enriched[0].posteriors).toEqual(chunks[0].posteriors);
    const {logitNullspaceVetoes,...phone}=enriched[0].phonemes[0];
    expect(phone).toEqual(chunks[0].phonemes[0]);
    expect(logitNullspaceVetoes).toEqual([certificate]);
  });
  it("excludes non-training speakers, performed edits and repeats from fitting",()=> {
    const r={kind:"none",split:"train",speaker:"train",events:[],non_errors:[]};
    expect(eligibleNullspaceTake(r,new Set())).toBe(true);
    expect(eligibleNullspaceTake(r,new Set(["train"]))).toBe(false);
    expect(eligibleNullspaceTake({...r,split:"eval"},new Set())).toBe(false);
    expect(eligibleNullspaceTake({...r,events:[{}]},new Set())).toBe(false);
    expect(eligibleNullspaceTake({...r,non_errors:[{}]},new Set())).toBe(false);
    expect(eligibleNullspaceTake({...r,excluded:"audit rejected"},new Set())).toBe(false);
  });
  it("rejects incomplete pronunciations and requires blank separation for genuine equal labels",()=> {
    expect(nullspaceBoundaryScore([[0,0,0]],[1,2])).toBe(-Infinity);
    expect(nullspaceBoundaryScore([[0,0,0],[0,0,0]],[1,1])).toBe(-Infinity);
    expect(nullspaceBoundaryScore([[0,0,0],[0,0,0],[0,0,0]],[1,1])).toBeCloseTo(0);
  });
  it("matches exhaustive complete-path marginalization and normalizes the shared timing prior",()=> {
    const values=[[.1,.3,.2],[.5,.4,.1],[.2,.8,.4],[.4,.2,.7]];
    for(const ids of [[1],[2],[1,2],[1,1]]) {
      const labels=[0,...ids.flatMap(p=>[p,0])];
      const edges=labels.map((p,s)=>[s,...(s+1<labels.length?[s+1]:[]),...(s+2<labels.length&&labels[s+2]&&labels[s+2]!==p?[s+2]:[])]);
      let likelihood=0,timing=0;
      function visit(t:number,s:number,weight:number,score:number) {
        score+=values[t][labels[s]];
        if(t===values.length-1) {if(s>=labels.length-2){timing+=weight;likelihood+=weight*Math.exp(score);}return;}
        for(const next of edges[s])visit(t+1,next,weight/edges[s].length,score);
      }
      visit(0,0,1,0);visit(0,1,1,0);
      expect(nullspaceBoundaryScore(values,ids)).toBeCloseTo(Math.log(likelihood/timing),12);
    }
  });
  it("retracts only the same live occurrence removed by certificate replay",()=> {
    const warning={type:"checker",v:1,id:"one",timeMs:4000,state:"confirmed",errorClass:"wrong_word",
      location:candidate.word,expected:candidate.written,observed:candidate.rival,confidence:1} as import("@tilawa/core").CheckerEvent;
    const repeat={...warning,id:"two",timeMs:8000}, laterGraph={...warning,timeMs:9000};
    const insertion={...warning,id:"three",errorClass:"added_word" as const};
    expect(nullspaceVerdictRemovals([warning,repeat,laterGraph,insertion],
      [warning,repeat,laterGraph,insertion],[repeat,laterGraph,insertion])).toEqual([warning]);
    expect(nullspaceVerdictRemovals([warning],[{...warning,timeMs:4001}],[])).toEqual([]);
  });
  it("retains the final projection's native clock and separate recurrent state",async()=> {
    let calls=0;
    const geometry={tokens:"ab",fbank:{mels:80,padValue:-23},window:45,shift:32,
      states:[{name:"cache",shape:[1],dtype:"float32"}]} as ModelManifest;
    const backend={name:"extractor",async release(){},async run(feeds:Record<string,import("../src/model.js").Tensor>) {
      expect(Number(feeds.cache.data[0])).toBe(calls);
      const offset=100*calls++;
      return {
        log_probs:{type:"float32" as const,dims:[1,8,3],data:new Float32Array(24)},
        new_cache:{type:"float32" as const,dims:[1],data:new Float32Array([calls])},
        features:{type:"float32" as const,dims:[1,8,6],data:Float32Array.from({length:48},(_,i)=>offset+i)},
        logits:{type:"float32" as const,dims:[1,8,3],data:Float32Array.from({length:24},(_,i)=>offset+i)},
      };
    }};
    const rows=await nullspaceObservations(backend,geometry,model(),new Float32Array(16000));
    expect(rows.nullspace).toHaveLength(23);
    expect(rows.control).toHaveLength(23);
    expect(rows.nullspace[0]).toEqual([3,4,5]);
    expect(rows.control[0]).toEqual([0,1,2]);
    expect(rows.nullspace[8]).toEqual([103,104,105]);
    expect(calls).toBe(3);
  });
});
