import { describe, expect, it } from "vitest";
import type { ModelManifest, Tensor } from "../src/model.js";
import { carryGlottalSourceEvidence, glottalComparison, glottalObservations, glottalReversal, glottalVerdictRemovals,
  pronunciationLogP, readoutRows, validateGlottalSourceModel, type GlottalSourceModel } from "../src/glottal-source.js";

const manifest={inventory:"test",tokens:"ab",sha256:"parent"} as ModelManifest;
function model():GlottalSourceModel {
  return {schema:1,inventory:"test",tokens:"ab",width:1,primarySha256:"parent",featureOutput:"features",
    invariant:{weight:[[0],[2],[-2]],bias:[0,0,0]},control:{weight:[[0],[-2],[2]],bias:[0,0,0]}};
}
const candidate={word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:3,toFrame:5};

describe("paired source-invariant phone evidence",()=> {
  it("requires strict opposing complete-pronunciation preferences on the same rows",()=> {
    const c=glottalComparison(model(),[[1],[1]],candidate)!;
    expect(glottalReversal(c)).toBe(true);
    expect(glottalReversal({...c,controlWritten:c.controlRival})).toBe(false);
    expect(glottalReversal({...c,invariantWritten:NaN})).toBe(false);
    expect(glottalComparison(model(),[[1]],candidate)).toBeNull();
    expect(glottalComparison(model(),[[1],[1]],{...candidate,rival:"x"})).toBeNull();
  });
  it("validates both readout geometries and ties them to the exact primary model",()=> {
    expect(validateGlottalSourceModel(model(),manifest)).toEqual(model());
    expect(()=>validateGlottalSourceModel({...model(),primarySha256:"other"},manifest)).toThrow();
    expect(()=>validateGlottalSourceModel({...model(),control:{weight:[[1]],bias:[0]}},manifest)).toThrow();
    const bad=model();bad.invariant.weight[0][0]=NaN;
    expect(()=>validateGlottalSourceModel(bad,manifest)).toThrow();
  });
  it("matches exhaustive CTC path summation, including repeated phone occurrences",()=> {
    const rows=[[.4,.5,.1],[.4,.4,.2],[.2,.7,.1],[.6,.3,.1]];
    for (const ids of [[1],[2],[1,2],[1,1]]) {
      let sum=0;
      const visit=(path:number[])=> {
        if (path.length===rows.length) {
          const collapsed=path.filter((p,i)=>p!==0&&(i===0||path[i-1]!==p));
          if (JSON.stringify(collapsed)===JSON.stringify(ids)) sum+=path.reduce((v,p,i)=>v*rows[i][p],1);
          return;
        }
        for (let p=0;p<3;p++) visit([...path,p]);
      };
      visit([]);
      expect(pronunciationLogP(rows.map(r=>r.map(Math.log)),ids)).toBeCloseTo(Math.log(sum),12);
    }
    expect(pronunciationLogP([[0,0,0],[0,0,0]],[1,1])).toBe(-Infinity);
    expect(pronunciationLogP([[0,0,0]],[1,2])).toBe(-Infinity);
    expect(readoutRows(model().invariant,[[1]])[0].reduce((s,v)=>s+Math.exp(v),0)).toBeCloseTo(1,12);
  });
  it("transports only metadata and retains primary tokens and posteriors",()=> {
    const chunks=[{inventory:"test",frameMs:40,phonemes:[{p:"a",start:0,end:1,conf:.9}],
      posteriors:{k:1,startFrame:0,nFrames:2,ids:[1,1],probs:[.9,.9]}}];
    const c=glottalComparison(model(),[[1],[1]],candidate)!;
    const enriched=JSON.parse(JSON.stringify(carryGlottalSourceEvidence(chunks,[c])));
    expect(enriched[0].posteriors).toEqual(chunks[0].posteriors);
    const {glottalSourceVetoes,...phone}=enriched[0].phonemes[0];
    expect(phone).toEqual(chunks[0].phonemes[0]);
    expect(glottalSourceVetoes).toEqual([c]);
  });
  it("retracts only the same live occurrence, retaining repeats and later graph verdicts",()=> {
    const warning={type:"checker",v:1,id:"one",timeMs:4000,state:"confirmed",errorClass:"wrong_word",
      location:candidate.word,expected:candidate.written,observed:candidate.rival,confidence:1} as import("@tilawa/core").CheckerEvent;
    const repeat={...warning,id:"two",timeMs:8000}, graph={...warning,timeMs:9000};
    const insertion={...warning,id:"three",errorClass:"added_word" as const};
    expect(glottalVerdictRemovals([warning,repeat,graph,insertion],[warning,repeat,graph,insertion],[repeat,graph,insertion])).toEqual([warning]);
    expect(glottalVerdictRemovals([warning],[{...warning,timeMs:4001}],[])).toEqual([]);
  });
  it("keeps the native acoustic clock with independent recurrent state",async()=> {
    let calls=0;
    const geometry={tokens:"ab",fbank:{mels:80,padValue:-23},window:45,shift:32,
      states:[{name:"cache",shape:[1],dtype:"float32"}]} as ModelManifest;
    const backend={name:"extractor",async release(){},async run(feeds:Record<string,Tensor>) {
      expect(Number(feeds.cache.data[0])).toBe(calls);
      const offset=100*calls++;
      return {log_probs:{type:"float32" as const,dims:[1,8,3],data:new Float32Array(24)},
        new_cache:{type:"float32" as const,dims:[1],data:new Float32Array([calls])},
        features:{type:"float32" as const,dims:[1,8,1],data:Float32Array.from({length:8},(_,i)=>offset+i)}};
    }};
    const rows=await glottalObservations(backend,geometry,model(),new Float32Array(16000));
    expect(rows).toHaveLength(23);
    expect(rows[0]).toEqual([0]);expect(rows[8]).toEqual([100]);expect(calls).toBe(3);
  });
});
