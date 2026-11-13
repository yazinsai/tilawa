import { describe, expect, it } from "vitest";
import type { ModelManifest } from "../src/model.js";
import { carryOccurrenceEvidence, eligibleOccurrenceTake, occurrenceBoundaryScore, occurrenceComparison,
  occurrenceObservations, occurrenceReversal, occurrenceVerdictRemovals, validateOccurrenceModel,
  type OccurrenceModel } from "../src/occurrence-mixture.js";

const manifest={inventory:"test",tokens:"ab",sha256:"parent"} as ModelManifest;
const model=():OccurrenceModel=>({schema:1,inventory:"test",tokens:"ab",primarySha256:"parent",width:1,
  dimensions:1,featureOutput:"features",center:[0],projection:[[1]],phones:[
    {weights:[.5,.5],means:[[10],[10]],variances:[[1],[1]],count:1},
    {weights:[.5,.5],means:[[0],[0]],variances:[[1],[1]],count:1},
    {weights:[.5,.5],means:[[-1],[1]],variances:[[.05],[.05]],count:1},
  ]});
const candidate={word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:3,toFrame:5};

describe("occurrence-tied phone mixtures",()=> {
  it("matches exhaustive timing and latent-component marginalization",()=> {
    const v=[[[.1,.4],[.3,.8],[.2,.5]],[[.5,.2],[.4,.3],[.1,.7]],[[.2,.9],[.8,.1],[.4,.2]],[[.4,.2],[.2,.6],[.7,.1]]];
    const weights=[[.4,.6],[.3,.7],[.8,.2]];
    for(const ids of [[1],[2],[1,2],[1,1]])for(const tied of [true,false]) {
      const labels=[0,...ids.flatMap(p=>[p,0])];
      let likelihood=0,timing=0;
      function visit(t:number,s:number,path:number[],weight:number) {
        path=[...path,s];
        if(t===v.length-1) {
          if(s<labels.length-2)return;
          timing+=weight;
          let mass=1;
          for(let i=0;i<path.length;) {
            const state=path[i],p=labels[state];let end=i+1;
            while(end<path.length&&path[end]===state)end++;
            if(tied&&p!==0)mass*=weights[p].reduce((sum,w,k)=>sum+w*Math.exp(path.slice(i,end).reduce((z,_,j)=>z+v[i+j][p][k],0)),0);
            else for(let j=i;j<end;j++)mass*=weights[p].reduce((sum,w,k)=>sum+w*Math.exp(v[j][p][k]),0);
            i=end;
          }
          likelihood+=weight*mass;return;
        }
        const next=[s,...(s+1<labels.length?[s+1]:[]),...(s+2<labels.length&&labels[s+2]!==0&&labels[s+2]!==labels[s]?[s+2]:[])];
        for(const n of next)visit(t+1,n,path,weight/next.length);
      }
      visit(0,0,[],1);visit(0,1,[],1);
      expect(occurrenceBoundaryScore(v,weights,ids,tied)).toBeCloseTo(Math.log(likelihood/timing),12);
    }
  });
  it("agrees for one component and one-frame occurrences, and normalizes timing",()=> {
    const v=[[[0],[.2],[.3]],[[0],[.4],[.1]],[[0],[.5],[.6]]],weights=[[1],[1],[1]];
    for(const ids of [[1],[1,2],[1,1]])expect(occurrenceBoundaryScore(v,weights,ids,true)).toBeCloseTo(occurrenceBoundaryScore(v,weights,ids,false),12);
    const zero=Array.from({length:5},()=>Array.from({length:3},()=>[0,0]));
    for(const tied of [true,false])expect(occurrenceBoundaryScore(zero,[[.1,.9],[.3,.7],[.8,.2]],[1,1],tied)).toBeCloseTo(0,12);
    expect(occurrenceBoundaryScore([[[0,0],[1,2],[2,1]]],[[.5,.5],[.5,.5],[.5,.5]],[1],true))
      .toBeCloseTo(occurrenceBoundaryScore([[[0,0],[1,2],[2,1]]],[[.5,.5],[.5,.5],[.5,.5]],[1],false),12);
  });
  it("rejects incoherent rival components but retains a coherently spoken rival",()=> {
    const incoherent=occurrenceComparison(model(),[[-1],[1]],candidate)!;
    expect(occurrenceReversal(incoherent)).toBe(true);
    const coherent=occurrenceComparison(model(),[[1],[1]],candidate)!;
    expect(coherent.tiedRival).toBeGreaterThan(coherent.tiedWritten);
    expect(occurrenceReversal(coherent)).toBe(false);
    expect(occurrenceReversal({...incoherent,tiedWritten:NaN})).toBe(false);
    expect(occurrenceReversal({...incoherent,switchingWritten:incoherent.switchingRival})).toBe(false);
  });
  it("distinguishes repeated occurrences and requires complete pronunciations",()=> {
    const v=Array.from({length:3},()=>[[0],[0],[0]]),w=[[1],[1],[1]];
    expect(occurrenceBoundaryScore(v.slice(0,1),w,[1,2],true)).toBe(-Infinity);
    expect(occurrenceBoundaryScore(v.slice(0,2),w,[1,1],true)).toBe(-Infinity);
    expect(occurrenceBoundaryScore(v,w,[1,1],true)).toBeCloseTo(0);
  });
  it("rejects malformed assets and abstains on missing acoustic coverage",()=> {
    expect(validateOccurrenceModel(model(),manifest)).toEqual(model());
    expect(()=>validateOccurrenceModel({...model(),primarySha256:"other"},manifest)).toThrow();
    const invalid=model();invalid.phones[1].weights=[.5,.7];
    expect(()=>validateOccurrenceModel(invalid,manifest)).toThrow();
    expect(occurrenceComparison(model(),[[-1]],candidate)).toBeNull();
    expect(occurrenceComparison(model(),[[-1],[1]],{...candidate,rival:"x"})).toBeNull();
    const absent=model();absent.phones[2].count=0;
    expect(occurrenceComparison(absent,[[-1],[1]],candidate)).toBeNull();
  });
  it("retains primary tokens and posteriors through certificate transport",()=> {
    const chunks=[{inventory:"test",frameMs:40,phonemes:[{p:"a",start:0,end:1,conf:.9}],posteriors:{k:1,startFrame:0,nFrames:2,ids:[1,1],probs:[.9,.9]}}];
    const cert={...candidate,tiedWritten:1,tiedRival:0,switchingWritten:0,switchingRival:1};
    const enriched=JSON.parse(JSON.stringify(carryOccurrenceEvidence(chunks,[cert])));
    expect(enriched[0].posteriors).toEqual(chunks[0].posteriors);
    const {occurrenceMixtureVetoes,...phone}=enriched[0].phonemes[0];
    expect(phone).toEqual(chunks[0].phonemes[0]);expect(occurrenceMixtureVetoes).toEqual([cert]);
  });
  it("excludes release speakers, edits, repeats and rejected recordings",()=> {
    const r={kind:"none",split:"train",speaker:"train",events:[],non_errors:[]};
    expect(eligibleOccurrenceTake(r,new Set())).toBe(true);
    for(const changes of [{split:"eval"},{events:[{}]},{non_errors:[{}]},{excluded:"rejected"},{kind:"wrong_word"}])
      expect(eligibleOccurrenceTake({...r,...changes},new Set())).toBe(false);
    expect(eligibleOccurrenceTake(r,new Set(["train"]))).toBe(false);
  });
  it("only retracts the same live occurrence removed by certificate replay",()=> {
    const warning={type:"checker",v:1,id:"one",timeMs:4000,state:"confirmed",errorClass:"wrong_word",
      location:candidate.word,expected:"a",observed:"b",confidence:1} as import("@tilawa/core").CheckerEvent;
    const repeat={...warning,id:"two",timeMs:8000},graph={...warning,timeMs:9000},insertion={...warning,errorClass:"added_word" as const};
    expect(occurrenceVerdictRemovals([warning,repeat,graph,insertion],[warning,repeat,graph,insertion],[repeat,graph,insertion])).toEqual([warning]);
  });
  it("replays the deployed clock with a separate recurrent cache",async()=> {
    let calls=0;
    const geometry={tokens:"ab",fbank:{mels:80,padValue:-23},window:45,shift:32,states:[{name:"cache",shape:[1],dtype:"float32"}]} as ModelManifest;
    const backend={name:"extractor",async release(){},async run(feeds:Record<string,import("../src/model.js").Tensor>) {
      expect(Number(feeds.cache.data[0])).toBe(calls);
      const offset=100*calls++;
      return {log_probs:{type:"float32" as const,dims:[1,8,3],data:new Float32Array(24)},
        new_cache:{type:"float32" as const,dims:[1],data:new Float32Array([calls])},
        features:{type:"float32" as const,dims:[1,8,1],data:Float32Array.from({length:8},(_,i)=>offset+i)}};
    }};
    const rows=await occurrenceObservations(backend,geometry,model(),new Float32Array(16000));
    expect(rows.features).toHaveLength(23);expect(rows.logProbs).toHaveLength(23);
    expect(rows.features[8]).toEqual([100]);expect(calls).toBe(3);
  });
});
