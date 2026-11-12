import { describe, expect, it } from "vitest";
import type { ModelManifest } from "../src/model.js";
import { carryBottleneckEvidence, eligibleBottleneckTake, bottleneckBoundaryScore, bottleneckComparison,
  bottleneckObservations, bottleneckRows, bottleneckQuadrature, bottleneckReversal, bottleneckVerdictRemovals, validateBottleneckModel,
  type BottleneckModel } from "../src/bottleneck.js";

const manifest = { inventory:"test", tokens:"ab", sha256:"parent" } as ModelManifest;
function model(): BottleneckModel {
  const arm = (sign:number) => ({encoder:[[1],[0]],encoderBias:[0,-8],
    output:[[0],[sign],[-sign]],outputBias:[-1,0,0]});
  return {schema:1,inventory:"test",tokens:"ab",primarySha256:"parent",width:1,latent:1,fixedNoiseStd:.3,
    featureOutput:"features",center:[0],scale:[1],bottleneck:arm(1),deterministic:arm(-1),fixedNoise:arm(-1)};
}
const candidate = {word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:3,toFrame:5};

describe("matched bottleneck acoustic evidence",()=> {
  it("uses deterministic Gaussian nodes with exact means and covariance",()=> {
    const mean=[2,-3],std=[.5,2],nodes=bottleneckQuadrature(mean,std);
    expect(nodes).toHaveLength(4);
    for(let d=0;d<2;d++) {
      expect(nodes.reduce((sum,n)=>sum+n[d],0)/nodes.length).toBeCloseTo(mean[d],12);
      for(let j=0;j<2;j++) expect(nodes.reduce((sum,n)=>sum+(n[d]-mean[d])*(n[j]-mean[j]),0)/nodes.length)
        .toBeCloseTo(d===j ? std[d]**2 : 0,12);
    }
    expect(bottleneckQuadrature(mean,std)).toEqual(nodes);
  });
  it("integrates class probabilities rather than scoring only the latent mean",()=> {
    const m=model();
    m.bottleneck.encoderBias[1]=0;
    m.bottleneck.outputBias=[-2,1,-1];
    const row=bottleneckRows(m,m.bottleneck,[[0]],"bottleneck")[0];
    const softmax=(v:number[])=>v.map(x=>Math.exp(x)/v.reduce((s,x)=>s+Math.exp(x),0));
    const a=softmax([-2,0,0]),b=softmax([-2,2,-2]);
    expect(row.map(Math.exp)).toEqual(a.map((v,i)=>expect.closeTo((v+b[i])/2,12)));
    expect(Math.exp(row[1])).not.toBeCloseTo(softmax([-2,1,-1])[1],3);
  });
  it("requires equal learned parameter counts, finite weights and exact primary identity",()=> {
    expect(validateBottleneckModel(model(),manifest)).toEqual(model());
    expect(()=>validateBottleneckModel({...model(),latent:0},manifest)).toThrow();
    expect(()=>validateBottleneckModel({...model(),primarySha256:"other"},manifest)).toThrow();
    expect(()=>validateBottleneckModel({...model(),scale:[0]},manifest)).toThrow();
    const bad=model();bad.deterministic.encoder[0][0]=NaN;
    expect(()=>validateBottleneckModel(bad,manifest)).toThrow();
  });
  it("requires positive written evidence under the bottleneck arm and rival preference in both controls",()=> {
    const c=bottleneckComparison(model(),[[2],[2]],candidate)!;
    expect(bottleneckReversal(c)).toBe(true);
    expect(bottleneckReversal({...c,fixedNoiseWritten:c.fixedNoiseRival})).toBe(false);
    expect(bottleneckReversal({...c,deterministicWritten:c.deterministicRival})).toBe(false);
    expect(bottleneckReversal({...c,bottleneckWritten:NaN})).toBe(false);
    expect(bottleneckComparison(model(),[[1]],candidate)).toBeNull();
    expect(bottleneckComparison(model(),[[2],[2]],{...candidate,rival:"x"})).toBeNull();
    expect(bottleneckComparison(model(),[[1],[NaN]],candidate)).toBeNull();
  });
  it("keeps primary tokens and posteriors unchanged through JSON transport",()=> {
    const chunks=[{inventory:"test",frameMs:40,phonemes:[{p:"a",start:0,end:1,conf:.9}],
      posteriors:{k:1,startFrame:0,nFrames:2,ids:[1,1],probs:[.9,.9]}}];
    const c=bottleneckComparison(model(),[[2],[2]],candidate)!;
    const enriched=JSON.parse(JSON.stringify(carryBottleneckEvidence(chunks,[c])));
    expect(enriched[0].posteriors).toEqual(chunks[0].posteriors);
    const {bottleneckVetoes,...phone}=enriched[0].phonemes[0];
    expect(phone).toEqual(chunks[0].phonemes[0]);
    expect(bottleneckVetoes).toEqual([c]);
  });
  it("excludes held-out speakers, edited takes and repeats from canonical CTC supervision",()=> {
    const r={kind:"none",split:"train",speaker:"train",events:[],non_errors:[]};
    expect(eligibleBottleneckTake(r,new Set())).toBe(true);
    expect(eligibleBottleneckTake(r,new Set(["train"]))).toBe(false);
    expect(eligibleBottleneckTake({...r,split:"eval"},new Set())).toBe(false);
    expect(eligibleBottleneckTake({...r,events:[{}]},new Set())).toBe(false);
    expect(eligibleBottleneckTake({...r,non_errors:[{}]},new Set())).toBe(false);
    expect(eligibleBottleneckTake({...r,excluded:"audit rejected"},new Set())).toBe(false);
  });
  it("requires completion and blank separation between repeated phones",()=> {
    expect(bottleneckBoundaryScore([[0,0,0]],[1,2])).toBe(-Infinity);
    expect(bottleneckBoundaryScore([[0,0,0],[0,0,0]],[1,1])).toBe(-Infinity);
    expect(bottleneckBoundaryScore([[0,0,0],[0,0,0],[0,0,0]],[1,1])).toBeCloseTo(0);
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
      expect(bottleneckBoundaryScore(values,ids)).toBeCloseTo(Math.log(likelihood/timing),12);
    }
  });
  it("retracts only the live occurrence removed by an otherwise identical replay",()=> {
    const warning={type:"checker",v:1,id:"one",timeMs:4000,state:"confirmed",errorClass:"wrong_word",
      location:candidate.word,expected:"a",observed:"b",confidence:1} as import("@tilawa/core").CheckerEvent;
    const repeat={...warning,id:"two",timeMs:8000}, graph={...warning,timeMs:9000};
    expect(bottleneckVerdictRemovals([warning,repeat,graph],[warning,repeat,graph],[repeat,graph])).toEqual([warning]);
    expect(bottleneckVerdictRemovals([warning],[{...warning,timeMs:4001}],[])).toEqual([]);
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
    const rows=await bottleneckObservations(backend,geometry,model(),new Float32Array(16000));
    expect(rows.features).toHaveLength(23);
    expect(rows.features[0]).toEqual([0]);
    expect(rows.features[8]).toEqual([100]);
    expect(calls).toBe(3);
  });
});
