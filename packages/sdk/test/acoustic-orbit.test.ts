import { describe, expect, it } from "vitest";
import type { ModelManifest } from "../src/model.js";
import { carryAcousticOrbitEvidence, eligibleAcousticOrbitTake, orbitBoundaryScore, orbitComparison,
  orbitEmissions, orbitObservations, orbitReversal, validateAcousticOrbitModel, type AcousticOrbitModel } from "../src/acoustic-orbit.js";

const manifest={inventory:"test",tokens:"ab",sha256:"parent"} as ModelManifest;
function model():AcousticOrbitModel {
  return {schema:1,inventory:"test",tokens:"ab",primarySha256:"parent",width:4,dimensions:2,
    featureOutput:"features",logitOutput:"logits",featureCenter:[0,0,0,0],projection:[[1,0,0,0],[0,1,0,0]],
    trainingRecordings:3,means:[[9,9],[0,0],[1,2]],counts:[1,1,1],globalPrecision:[[1,0],[0,1]],
    regions:[{center:[0,2],precision:[[1,0],[0,.1]]},{center:[20,2],precision:[[.1,0],[0,1]]}],
    neighborhoodScale:1,temperature:1};
}
const candidate={word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:3,toFrame:5};

describe("local acoustic orbit geometry",()=> {
  it("reverses toward the written sequence only when both identical-feature controls prefer the rival",()=> {
    const fitted=model(),x=[[0,2],[0,2]];
    const certificate=orbitComparison(fitted,x,candidate)!;
    expect(orbitReversal(certificate)).toBe(true);
    expect(orbitReversal({...certificate,euclideanWritten:certificate.euclideanRival})).toBe(false);
    expect(orbitReversal({...certificate,globalWritten:certificate.globalRival})).toBe(false);
    expect(orbitReversal({...certificate,localWritten:NaN})).toBe(false);
    expect(orbitComparison(fitted,[[0,2]],candidate)).toBeNull();
    expect(orbitComparison(fitted,x,{...candidate,rival:"x"})).toBeNull();
  });
  it("makes nuisance directions depend on the acoustic neighborhood",()=> {
    const fitted=model();fitted.means=[[9,9],[0,0],[1,2]];
    const local=orbitEmissions(fitted,[[0,2],[20,2]],"local");
    // Vertical movement is cheap only by the first anchor, horizontal only by the second.
    expect(local[0][1]).toBeGreaterThan(local[0][2]);
    expect(local[1][1]).toBeLessThan(local[1][2]);
    expect(orbitEmissions(fitted,[[0,2]],"euclidean")[0][1]).toBeLessThan(orbitEmissions(fitted,[[0,2]],"euclidean")[0][2]);
    expect(local.every(r=>Math.abs(r.reduce((s,v)=>s+Math.exp(v),0)-1)<1e-12)).toBe(true);
  });
  it("rejects malformed geometry, unsupported prototypes and a different primary",()=> {
    expect(validateAcousticOrbitModel(model(),manifest)).toEqual(model());
    const invalid=model();invalid.regions[0].precision[0][0]=0;
    expect(()=>validateAcousticOrbitModel(invalid,manifest)).toThrow();
    expect(()=>validateAcousticOrbitModel({...model(),primarySha256:"other"},manifest)).toThrow();
    expect(()=>validateAcousticOrbitModel({...model(),neighborhoodScale:NaN},manifest)).toThrow();
    const missing=model();missing.counts[1]=0;
    expect(orbitComparison(missing,[[0,2],[0,2]],candidate)).toBeNull();
  });
  it("preserves primary tokens and posteriors through serialized certificates",()=> {
    const chunks=[{inventory:"test",frameMs:40,phonemes:[{p:"a",start:0,end:1,conf:.9}],posteriors:{k:1,startFrame:0,nFrames:2,ids:[1,1],probs:[.9,.9]}}];
    const certificate=orbitComparison(model(),[[0,2],[0,2]],candidate)!;
    const result=JSON.parse(JSON.stringify(carryAcousticOrbitEvidence(chunks,[certificate])));
    expect(result[0].posteriors).toEqual(chunks[0].posteriors);
    const {acousticOrbitVetoes,...phone}=result[0].phonemes[0];
    expect(phone).toEqual(chunks[0].phonemes[0]);expect(acousticOrbitVetoes).toEqual([certificate]);
  });
  it("excludes non-training speakers, scripted mistakes, excluded takes and repeats",()=> {
    const r={kind:"none",split:"train",speaker:"train",events:[],non_errors:[]};
    expect(eligibleAcousticOrbitTake(r,new Set())).toBe(true);
    expect(eligibleAcousticOrbitTake(r,new Set(["train"]))).toBe(false);
    expect(eligibleAcousticOrbitTake({...r,split:"eval"},new Set())).toBe(false);
    expect(eligibleAcousticOrbitTake({...r,events:[{}]},new Set())).toBe(false);
    expect(eligibleAcousticOrbitTake({...r,non_errors:[{}]},new Set())).toBe(false);
    expect(eligibleAcousticOrbitTake({...r,excluded:true},new Set())).toBe(false);
  });
  it("requires complete pronunciations and blank separation for repeated phones",()=> {
    expect(orbitBoundaryScore([[0,0,0]],[1,2])).toBe(-Infinity);
    expect(orbitBoundaryScore([[0,0,0],[0,0,0]],[1,1])).toBe(-Infinity);
    expect(orbitBoundaryScore([[0,0,0],[0,0,0],[0,0,0]],[1,1])).toBeCloseTo(0);
  });
  it("matches exhaustive pronunciation and boundary marginalization",()=> {
    const values=[[.1,.3,.2],[.5,.4,.1],[.2,.8,.4],[.4,.2,.7]];
    for(const ids of [[1],[2],[1,2],[1,1]]) {
      const labels=[0,...ids.flatMap(p=>[p,0])];
      const edges=labels.map((p,s)=>[s,...(s+1<labels.length?[s+1]:[]),...(s+2<labels.length&&labels[s+2]&&labels[s+2]!==p?[s+2]:[])]);
      let likelihood=0,timing=0;
      function visit(t:number,s:number,weight:number,score:number) {
        score+=values[t][labels[s]];
        if(t===values.length-1){if(s>=labels.length-2){timing+=weight;likelihood+=weight*Math.exp(score);}return;}
        for(const next of edges[s])visit(t+1,next,weight/edges[s].length,score);
      }
      visit(0,0,1,0);visit(0,1,1,0);
      expect(orbitBoundaryScore(values,ids)).toBeCloseTo(Math.log(likelihood/timing),12);
    }
  });
  it("uses independent recurrent state and the deployed final-frame clock",async()=> {
    let calls=0;
    const geometry={tokens:"ab",fbank:{mels:80,padValue:-23},window:45,shift:32,
      states:[{name:"cache",shape:[1],dtype:"float32"}]} as ModelManifest;
    const backend={name:"extractor",async release(){},async run(feeds:Record<string,import("../src/model.js").Tensor>) {
      expect(Number(feeds.cache.data[0])).toBe(calls);
      const offset=100*calls++;
      return {log_probs:{type:"float32" as const,dims:[1,8,3],data:new Float32Array(24)},
        new_cache:{type:"float32" as const,dims:[1],data:new Float32Array([calls])},
        features:{type:"float32" as const,dims:[1,8,4],data:Float32Array.from({length:32},(_,i)=>offset+i)},
        logits:{type:"float32" as const,dims:[1,8,3],data:new Float32Array(24)}};
    }};
    const rows=await orbitObservations(backend,geometry,model(),new Float32Array(16000));
    expect(rows.observations).toHaveLength(23);expect(rows.observations[0]).toEqual([0,1]);
    expect(rows.observations[8]).toEqual([100,101]);expect(calls).toBe(3);
  });
});
