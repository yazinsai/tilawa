import { describe, expect, it } from "vitest";
import { carryResidualEvidence, correctedResidualRows, eligiblePreservationTake, eligibleResidualTake,
  residualComparison, residualCtcScore, residualObservations, residualPreference, treeValue, validateResidualModel } from "../src/boosted-residuals.js";
import type { ResidualModel, ResidualTree } from "../src/boosted-residuals.js";
import { createListener } from "../src/listener.js";
import type { InferenceBackend, LoadedModel, ModelManifest, Tensor } from "../src/model.js";
import type { PhonemeEvidence } from "@tilawa/core";

const tree:ResidualTree={feature:[0,-2,-2],threshold:[0,0,0],left:[1,-1,-1],right:[2,-1,-1],
  value:[[0,0,0],[0,3,-3],[0,-3,3]]};
const model:ResidualModel={schema:1,inventory:"test",tokens:"ab",dimensions:3,width:1,featureOutput:"h",logitOutput:"z",
  primarySha256:"test",rate:1,trees:[tree]};
const manifest={inventory:"test",tokens:"ab",sha256:"test"} as ModelManifest;
const request={word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:3,toFrame:6};
const features=[[-1],[-1],[-1]], logits=[[0,0,1],[0,0,1],[0,0,1]];

describe("boosted acoustic residuals",()=>{
  it("sums all complete CTC paths, with repeated labels requiring an intervening blank",()=>{
    const p=[[.2,.5,.3],[.4,.4,.2],[.3,.2,.5]];
    for(const ids of [[1],[1,2],[1,1]]) {
      let total=0;
      for(let a=0;a<3;a++)for(let b=0;b<3;b++)for(let c=0;c<3;c++) {
        const path=[a,b,c], text=path.filter((v,i)=>v&&(i===0||v!==path[i-1]));
        if(JSON.stringify(text)===JSON.stringify(ids))total+=p[0][a]*p[1][b]*p[2][c];
      }
      expect(residualCtcScore(p.map(row=>row.map(Math.log)),ids)).toBeCloseTo(Math.log(total),12);
    }
    expect(residualCtcScore([[0,0,0]],[1,1])).toBe(-Infinity);
  });
  it("uses shared vector leaves and sklearn's inclusive left threshold",()=>{
    expect(treeValue(tree,[0])).toEqual([0,3,-3]);
    expect(treeValue(tree,[.001])).toEqual([0,-3,3]);
    const corrected=correctedResidualRows(model,features,logits);
    expect(corrected.every(r=>Math.abs(r.reduce((s,v)=>s+Math.exp(v),0)-1)<1e-12)).toBe(true);
    const c=residualComparison(model,features,logits,request)!;
    expect(c.shippedWritten).toBeLessThan(c.shippedRival);
    expect(residualPreference(c)).toBe(true);
    const reversed=residualComparison(model,features,logits,{...request,written:"b",rival:"a"})!;
    expect(reversed.correctedRival).toBe(c.correctedWritten);
    expect(reversed.correctedWritten).toBe(c.correctedRival);
    expect(logits).toEqual([[0,0,1],[0,0,1],[0,0,1]]);
  });
  it("leaves a zero ensemble equal to frozen scoring and abstains on ties or incomplete inputs",()=>{
    const empty={...model,trees:[]};
    const c=residualComparison(empty,features,logits,request)!;
    expect(c.correctedWritten).toBe(c.shippedWritten);
    expect(c.correctedRival).toBe(c.shippedRival);
    expect(residualPreference({...c,correctedWritten:c.correctedRival})).toBe(false);
    expect(residualPreference({...c,correctedWritten:NaN})).toBe(false);
    expect(residualComparison(model,features,logits,{...request,written:"x"})).toBeNull();
    expect(residualComparison(model,features,logits,{...request,toFrame:7})).toBeNull();
    expect(residualComparison(model,[[-1]],[[0,0,1]],{...request,written:"aa",toFrame:4})).toBeNull();
  });
  it("rejects corrupt topology, feature dimensions and nonfinite leaves",()=>{
    expect(validateResidualModel(model,manifest)).toBe(model);
    const invalid=(t:ResidualTree)=>({...model,trees:[t]});
    expect(()=>validateResidualModel(invalid({...tree,left:[0,-1,-1]}),manifest)).toThrow("topology");
    expect(()=>validateResidualModel(invalid({...tree,feature:[1,-2,-2]}),manifest)).toThrow("feature");
    expect(()=>validateResidualModel(invalid({...tree,value:[[0,0,0],[NaN,0,0],[0,0,0]]}),manifest)).toThrow();
    expect(()=>validateResidualModel({...model,primarySha256:"other"},manifest)).toThrow();
  });
  it("keeps mistake recordings only as preservation controls and excludes nontraining speakers",()=>{
    const r={speaker:"learner",kind:"none",split:"train",events:[],non_errors:[]};
    expect(eligibleResidualTake(r,new Set())).toBe(true);
    for(const changed of [{kind:"wrong_word"},{non_errors:[{}]},{events:[{}]},{split:"dev"},{exclude:true},{scored:false}])
      expect(eligibleResidualTake({...r,...changed},new Set())).toBe(false);
    expect(eligibleResidualTake(r,new Set(["learner"]))).toBe(false);
    expect(eligiblePreservationTake({...r,kind:"wrong_word",events:[{}]},new Set())).toBe(true);
  });
  it("carries certificates without altering primary tokens or posteriors",()=>{
    const chunks:PhonemeEvidence[]=[{inventory:"test",frameMs:40,phonemes:[{p:"b",start:3,end:5,conf:.9}]}];
    const c=residualComparison(model,features,logits,request)!;
    expect(carryResidualEvidence(chunks,[])).toBe(chunks);
    const result=carryResidualEvidence(chunks,[c]);
    expect(result[0].phonemes[0].boostedResidualVetoes).toEqual([c]);
    expect(chunks[0].phonemes[0].boostedResidualVetoes).toBeUndefined();
    const {boostedResidualVetoes:ignored,...primary}=result[0].phonemes[0];
    expect(primary).toEqual(chunks[0].phonemes[0]);
  });
  it("makes corrected evidence available before checker confirmation without changing primary outputs",async()=>{
    const m={...manifest,inventory:"test",sampleRate:16000,frameMs:40,shift:8,window:15,
      fbank:{mels:80,padValue:-23},states:[{name:"state",dtype:"float32",shape:[1]}]} as ModelManifest;
    const lp=Float32Array.from([0,0,1,0,0,1],v=>v-Math.log(2+Math.E));
    const backend=(diagnostic=false):InferenceBackend=>({name:"fixture",release:async()=>{},run:async feeds=>{
      const t=(data:Float32Array,dims:number[]):Tensor=>({type:"float32",data,dims});
      const previous=Number(feeds.state.data[0]);
      if(diagnostic)feeds.state.data[0]=99; // An auxiliary backend cannot mutate primary caches.
      return {log_probs:t(lp.slice(),[1,2,3]),new_state:t(new Float32Array([previous+1]),[1]),
        h:t(new Float32Array([-1,-1]),[1,2,1]),z:t(new Float32Array([0,0,1,0,0,1]),[1,2,3])};
    }});
    const wave=new Float32Array(400+18*160);
    const run=async(residual:boolean)=>{
      const loaded:LoadedModel={manifest:m,backend:backend(),supported:[],
        ...(residual?{boostedResiduals:{model,backend:backend(true)}}:{})};
      const listener=createListener(loaded,{posteriors:true,topK:3}),evidence=[];
      const e=await listener.push(wave,16000);if(e)evidence.push(e);
      const early=listener.residualComparison?.({...request,fromFrame:0,toFrame:2});
      const end=await listener.end();if(end)evidence.push(end);
      return {listener,evidence,early};
    };
    const baseline=await run(false),reviewed=await run(true);
    expect(reviewed.evidence).toEqual(baseline.evidence);
    expect(residualPreference(reviewed.early!)).toBe(true);
    expect(reviewed.listener.residualComparison!({...request,fromFrame:0,toFrame:3})).toEqual(
      residualComparison(model,features,logits,{...request,fromFrame:0,toFrame:3}));
    expect(reviewed.listener.residualComparison!({...request,toFrame:100})).toBeNull();
  });
  it("uses independent streaming caches and native output geometry at the padded tail",async()=>{
    let calls=0;
    const backend:InferenceBackend={name:"diagnostic",release:async()=>{},run:async feeds=>{
      expect(feeds.state.data[0]).toBe(calls);
      calls++;
      const t=(data:Float32Array,dims:number[]):Tensor=>({type:"float32",data,dims});
      return {log_probs:t(new Float32Array(6),[1,2,3]),h:t(new Float32Array([calls,calls]),[1,2,1]),
        z:t(new Float32Array(6),[1,2,3]),new_state:t(new Float32Array([calls]),[1])};
    }};
    // 19 fbank frames yield 3 native CTC frames; the second window supplies one padded output.
    const m={...manifest,shift:8,window:15,fbank:{mels:80,padValue:-23},states:[{name:"state",dtype:"float32",shape:[1]}]} as ModelManifest;
    const out=await residualObservations(backend,m,model,new Float32Array(400+18*160));
    expect(out.features).toEqual([[1],[1],[2]]);
    expect(out.logits).toHaveLength(3);
    expect(calls).toBe(2);
  });
});
