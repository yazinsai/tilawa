import { describe, expect, it } from "vitest";
import type { ModelManifest } from "../src/model.js";
import { carryFingerprintEvidence, eligibleFingerprintTake, fingerprintBoundaryScore, fingerprintComparison, fingerprintObservations, fingerprintReversal, fingerprintVerdictRemovals, validateFingerprintModel, type FingerprintModel, fingerprintMetadata } from "../src/response-fingerprint.js";

const manifest = { inventory:"test",tokens:"ab",sha256:"parent",sampleRate:16000,frameMs:40,
  fbank:{mels:80,padValue:-23},window:45,shift:32,states:[{name:"cache",shape:[1],dtype:"float32"}],
} as ModelManifest;
function model(): FingerprintModel {
  const arm = () => ({center:Array(6).fill(0),scale:Array(6).fill(1),phones:[0,1,2].map(p=>({
    mean:[p,0,0,0,0,0],precision:Array.from({length:6},(_,i)=>Array.from({length:6},(_,j)=>Number(i===j))),logDet:0,count:1,
  }))});
  return {...fingerprintMetadata(manifest),response:arm(),control:arm()};
}
const candidate = {word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:3,toFrame:5};

describe("final response-fingerprint acoustic evidence",()=> {
  it("binds geometry, perturbations and both equal-capacity arms to the primary",()=> {
    expect(validateFingerprintModel(model(),manifest)).toEqual(model());
    expect(()=>validateFingerprintModel({...model(),epsilon:.1},manifest)).toThrow();
    expect(()=>validateFingerprintModel({...model(),primarySha256:"other"},manifest)).toThrow();
    const bad=model();bad.control.scale.pop();
    expect(()=>validateFingerprintModel(bad,manifest)).toThrow();
    const indefinite=model();indefinite.response.phones[0].precision[0][0]=0;
    expect(()=>validateFingerprintModel(indefinite,manifest)).toThrow();
  });
  it("requires the response arm to recover the written pronunciation and the matched control to favour the rival",()=> {
    const certificate=fingerprintComparison(model(),[[1,0,0,0,0,0],[1,0,0,0,0,0]],[[2,0,0,0,0,0],[2,0,0,0,0,0]],candidate)!;
    expect(fingerprintReversal(certificate)).toBe(true);
    expect(fingerprintReversal({...certificate,controlWritten:certificate.controlRival})).toBe(false);
    expect(fingerprintReversal({...certificate,responseWritten:NaN})).toBe(false);
    expect(fingerprintComparison(model(),[[1,0,0,0,0,0]],[[2,0,0]],candidate)).toBeNull();
    expect(fingerprintComparison(model(),[[1,0,0,0,0,0],[1,0,0,0,0,0]],[[2,0,0,0,0,0],[2,0,0,0,0,0]],{...candidate,rival:"x"})).toBeNull();
  });
  it("keeps primary phones and posterior rows unchanged through JSON transport",()=> {
    const chunks=[{inventory:"test",frameMs:40,phonemes:[{p:"a",start:0,end:1,conf:.9}],posteriors:{k:1,startFrame:0,nFrames:2,ids:[1,1],probs:[.9,.9]}}];
    const certificate={...candidate,controlWritten:0,controlRival:1,responseWritten:1,responseRival:0};
    const enriched=JSON.parse(JSON.stringify(carryFingerprintEvidence(chunks,[certificate])));
    expect(enriched[0].posteriors).toEqual(chunks[0].posteriors);
    const {responseFingerprintVetoes,...phone}=enriched[0].phonemes[0];
    expect(phone).toEqual(chunks[0].phonemes[0]);
    expect(responseFingerprintVetoes).toEqual([certificate]);
  });
  it("excludes non-training speakers, performed edits and repeats from fitting",()=> {
    const r={kind:"none",split:"train",speaker:"train",events:[],non_errors:[]};
    expect(eligibleFingerprintTake(r,new Set())).toBe(true);
    expect(eligibleFingerprintTake(r,new Set(["train"]))).toBe(false);
    expect(eligibleFingerprintTake({...r,split:"eval"},new Set())).toBe(false);
    expect(eligibleFingerprintTake({...r,events:[{}]},new Set())).toBe(false);
    expect(eligibleFingerprintTake({...r,non_errors:[{}]},new Set())).toBe(false);
    expect(eligibleFingerprintTake({...r,excluded:"audit rejected"},new Set())).toBe(false);
  });
  it("rejects incomplete pronunciations and requires blank separation for genuine equal labels",()=> {
    expect(fingerprintBoundaryScore([[0,0,0]],[1,2])).toBe(-Infinity);
    expect(fingerprintBoundaryScore([[0,0,0],[0,0,0]],[1,1])).toBe(-Infinity);
    expect(fingerprintBoundaryScore([[0,0,0],[0,0,0],[0,0,0]],[1,1])).toBeCloseTo(0);
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
      expect(fingerprintBoundaryScore(values,ids)).toBeCloseTo(Math.log(likelihood/timing),12);
    }
  });
  it("retracts only the same live occurrence removed by certificate replay",()=> {
    const warning={type:"checker",v:1,id:"one",timeMs:4000,state:"confirmed",errorClass:"wrong_word",
      location:candidate.word,expected:candidate.written,observed:candidate.rival,confidence:1} as import("@tilawa/core").CheckerEvent;
    const repeat={...warning,id:"two",timeMs:8000}, laterGraph={...warning,timeMs:9000};
    const insertion={...warning,id:"three",errorClass:"added_word" as const};
    expect(fingerprintVerdictRemovals([warning,repeat,laterGraph,insertion],
      [warning,repeat,laterGraph,insertion],[repeat,laterGraph,insertion])).toEqual([warning]);
    expect(fingerprintVerdictRemovals([warning],[{...warning,timeMs:4001}],[])).toEqual([]);
    // A live graph verdict is never eligible, even if stored replay treats that exact occurrence as a string warning.
    expect(fingerprintVerdictRemovals([warning],[warning],[],new Set())).toEqual([]);
  });
  it("extracts signed symmetric local derivatives on the native clock from identical clean caches",async()=> {
    const feedsSeen:{x:Float32Array;cache:number}[]=[];
    const sharedState=new Float32Array(1);
    const backend={name:"linear-logit-model",async release(){},async run(feeds:Record<string,import("../src/model.js").Tensor>) {
      const cache=Number(feeds.cache.data[0]),x=feeds.x.data as Float32Array;
      feedsSeen.push({x:x.slice(),cache});
      const log_probs=new Float32Array(24);
      for(let t=0;t<8;t++) {
        let u=cache+t*.1;
        for(let b=0;b<80;b++) u+=x[(t*4+3)*80+b]*(b<40?1:-1)/80;
        const logits=[u,-u,.5*u], peak=Math.max(...logits);
        const normalizer=peak+Math.log(logits.reduce((sum,v)=>sum+Math.exp(v-peak),0));
        log_probs.set(logits.map(v=>v-normalizer),t*3);
      }
      // Mutating input and reusing output state must not contaminate the next arm/window.
      feeds.cache.data[0]=99;
      sharedState[0]=cache+1;
      return {log_probs:{type:"float32" as const,dims:[1,8,3],data:log_probs},
        new_cache:{type:"float32" as const,dims:[1],data:sharedState}};
    }};
    const meta=fingerprintMetadata(manifest);
    const audio=Float32Array.from({length:16000},(_,i)=>.1*Math.sin(i*.09));
    const rows=await fingerprintObservations(backend,manifest,meta,audio,f=>f===0||f===8);
    expect(rows.response).toHaveLength(23);
    expect(rows.control).toHaveLength(23);
    expect(feedsSeen.map(f=>f.cache)).toEqual([0,0,0,0,0,1,1,1,1,1,2]);
    expect(rows.response[1]).toEqual([]);
    for(const f of [0,8]) {
      const expected=meta.basis.flatMap(basis=>{
        const a=basis.reduce((sum,v,b)=>sum+v*(b<40?1:-1)/80,0);
        return [5*a/6,-7*a/6,a/3];
      });
      rows.response[f].forEach((v,d)=>expect(v).toBeCloseTo(expected[d],4));
    }
    expect(rows.response[0][0]).toBeGreaterThan(0);
    expect(rows.response[0][3]).toBeLessThan(0);
    expect(rows.control[0].slice(0,3)).toEqual(rows.control[0].slice(3));
    expect(rows.control[8].slice(3)).toEqual(rows.control[7].slice(0,3));
    for(const offset of [0,5]) for(let arm=1;arm<=4;arm++) {
      const baseline=feedsSeen[offset].x,perturbed=feedsSeen[offset+arm].x;
      const basis=meta.basis[Math.floor((arm-1)/2)],sign=arm%2?-1:1;
      for(let i=0;i<45;i++) for(let b=0;b<80;b++) {
        const envelope=Math.abs(i-3)<=4 ? (1+Math.cos(Math.PI*(i-3)/5))/2 : 0;
        expect(perturbed[i*80+b]-baseline[i*80+b]).toBeCloseTo(sign*meta.epsilon*envelope*basis[b],5);
      }
    }
  });
  it("enforces extraction deadlines and rejects an incompatible output clock",async()=> {
    const backend={name:"bad",async release(){},async run(){return {
      log_probs:{type:"float32" as const,dims:[1,1,3],data:new Float32Array(3)},
    };}};
    await expect(fingerprintObservations(backend,manifest,fingerprintMetadata(manifest),new Float32Array(16000),undefined,0))
      .rejects.toThrow(/budget/);
    await expect(fingerprintObservations(backend,manifest,fingerprintMetadata(manifest),new Float32Array(16000)))
      .rejects.toThrow(/clock/);
  });
});
