import { describe, expect, it } from "vitest";
import type { ModelManifest } from "../src/model.js";
import { carryDeconvolvedPhoneEvidence, eligibleDeconvolvedPhoneTake, deconvolvedBoundaryScore, deconvolvedComparison, deconvolvedObservations, deconvolvedReversal, deconvolvedVerdictRemovals, validateDeconvolvedPhoneModel, type DeconvolvedPhoneModel } from "../src/deconvolved-phone.js";

const manifest = { inventory:"test",tokens:"ab",sha256:"parent" } as ModelManifest;
function model(): DeconvolvedPhoneModel {
  const arm = () => ({center:[0,0,0],scale:[1,1,1],phones:[0,1,2].map(p=>({
    mean:[p,0,0],precision:[[1,0,0],[0,1,0],[0,0,1]],logDet:0,count:1,
  }))});
  return {schema:1,inventory:"test",tokens:"ab",primarySha256:"parent",dimensions:3,width:6,
    featureOutput:"features",logitOutput:"logits",featureCenter:[0,0,0,0,0,0],
    projection:[[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]],
    noiseCovariance:[[1,0,0],[0,1,0],[0,0,1]],trainingRecordings:3,latent:arm(),ordinary:arm()};

}
const candidate = {word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:3,toFrame:5};

describe("final deconvolved-phone acoustic evidence",()=> {
  it("validates one shared projection and positive observation covariance tied to the shipped graph",()=> {
    expect(validateDeconvolvedPhoneModel(model(),manifest)).toEqual(model());
    const malformed=model();malformed.projection[0]=[1];
    expect(()=>validateDeconvolvedPhoneModel(malformed,manifest)).toThrow();
    expect(()=>validateDeconvolvedPhoneModel({...model(),noiseCovariance:[[1,2,0],[2,1,0],[0,0,1]]},manifest)).toThrow();
    expect(()=>validateDeconvolvedPhoneModel({...model(),primarySha256:"other"},manifest)).toThrow();
    const density=model();density.latent.phones[1].logDet=4;
    expect(()=>validateDeconvolvedPhoneModel(density,manifest)).toThrow();
    const differentScale=model();differentScale.ordinary.scale[0]=2;
    expect(()=>validateDeconvolvedPhoneModel(differentScale,manifest)).toThrow();
  });
  it("requires a latent-written reversal against the ordinary arm on identical features",()=> {
    const fitted=model();
    // Removing inflated phone-a variance can change a normalized Gaussian ratio
    // without changing a single runtime feature or pronunciation.
    fitted.ordinary.phones[1].precision=[[.1,0,0],[0,.1,0],[0,0,.1]];
    fitted.ordinary.phones[1].logDet=6*Math.log(10);
    const certificate=deconvolvedComparison(fitted,[[1,0,0],[1,0,0]],candidate)!;
    expect(deconvolvedReversal(certificate)).toBe(true);
    expect(deconvolvedReversal({...certificate,ordinaryWritten:certificate.ordinaryRival})).toBe(false);
    expect(deconvolvedReversal({...certificate,latentWritten:NaN})).toBe(false);
    expect(deconvolvedComparison(model(),[[1,0,0]],candidate)).toBeNull();
    expect(deconvolvedComparison(model(),[[1,0,0],[1,0,0]],{...candidate,rival:"x"})).toBeNull();
  });
  it("keeps primary phones and posterior rows unchanged through JSON transport",()=> {
    const chunks=[{inventory:"test",frameMs:40,phonemes:[{p:"a",start:0,end:1,conf:.9}],posteriors:{k:1,startFrame:0,nFrames:2,ids:[1,1],probs:[.9,.9]}}];
    const certificate={...candidate,ordinaryWritten:0,ordinaryRival:1,latentWritten:1,latentRival:0};
    const enriched=JSON.parse(JSON.stringify(carryDeconvolvedPhoneEvidence(chunks,[certificate])));
    expect(enriched[0].posteriors).toEqual(chunks[0].posteriors);
    const {deconvolvedPhoneVetoes,...phone}=enriched[0].phonemes[0];
    expect(phone).toEqual(chunks[0].phonemes[0]);
    expect(deconvolvedPhoneVetoes).toEqual([certificate]);
  });
  it("excludes non-training speakers, performed edits and repeats from fitting",()=> {
    const r={kind:"none",split:"train",speaker:"train",events:[],non_errors:[]};
    expect(eligibleDeconvolvedPhoneTake(r,new Set())).toBe(true);
    expect(eligibleDeconvolvedPhoneTake(r,new Set(["train"]))).toBe(false);
    expect(eligibleDeconvolvedPhoneTake({...r,split:"eval"},new Set())).toBe(false);
    expect(eligibleDeconvolvedPhoneTake({...r,events:[{}]},new Set())).toBe(false);
    expect(eligibleDeconvolvedPhoneTake({...r,non_errors:[{}]},new Set())).toBe(false);
    expect(eligibleDeconvolvedPhoneTake({...r,excluded:"audit rejected"},new Set())).toBe(false);
  });
  it("rejects incomplete pronunciations and requires blank separation for genuine equal labels",()=> {
    expect(deconvolvedBoundaryScore([[0,0,0]],[1,2])).toBe(-Infinity);
    expect(deconvolvedBoundaryScore([[0,0,0],[0,0,0]],[1,1])).toBe(-Infinity);
    expect(deconvolvedBoundaryScore([[0,0,0],[0,0,0],[0,0,0]],[1,1])).toBeCloseTo(0);
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
      expect(deconvolvedBoundaryScore(values,ids)).toBeCloseTo(Math.log(likelihood/timing),12);
    }
  });
  it("retracts only the same live occurrence removed by certificate replay",()=> {
    const warning={type:"checker",v:1,id:"one",timeMs:4000,state:"confirmed",errorClass:"wrong_word",
      location:candidate.word,expected:candidate.written,observed:candidate.rival,confidence:1} as import("@tilawa/core").CheckerEvent;
    const repeat={...warning,id:"two",timeMs:8000}, laterGraph={...warning,timeMs:9000};
    const insertion={...warning,id:"three",errorClass:"added_word" as const};
    expect(deconvolvedVerdictRemovals([warning,repeat,laterGraph,insertion],
      [warning,repeat,laterGraph,insertion],[repeat,laterGraph,insertion])).toEqual([warning]);
    expect(deconvolvedVerdictRemovals([warning],[{...warning,timeMs:4001}],[])).toEqual([]);
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
    const rows=await deconvolvedObservations(backend,geometry,model(),new Float32Array(16000));
    expect(rows.observations).toHaveLength(23);
    expect(rows.observations[0]).toEqual([0,1,2]);
    expect(rows.observations[8]).toEqual([100,101,102]);
    expect(calls).toBe(3);
  });
});
