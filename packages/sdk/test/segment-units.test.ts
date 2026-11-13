import { describe, expect, it } from "vitest";
import { canonicalSegmentTake, carrySegmentEvidence, eligibleSegmentTake, segmentComparison, segmentDescriptor, segmentPronunciationScore, segmentReversal, validateSegmentModel } from "../src/segment-units.js";
import type { SegmentModel } from "../src/segment-units.js";

function model():SegmentModel {
  const arm=()=>({centers:[[0,0,0],[2,2,2]],variance:.1,phones:[[.5,.5],[.9,.1],[.1,.9]],counts:[1,1,1]});
  const forced=arm();forced.phones=[[.5,.5],[.1,.9],[.9,.1]];
  return {schema:1,inventory:"test",tokens:"ab",width:1,featureOutput:"features",primarySha256:"parent",
    center:[0],scale:[1],projection:[[1]],maxDuration:2,penalty:4,independent:arm(),forced};
}
const candidate={word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:3,toFrame:5};
describe("independent acoustic segment units",()=> {
  it("rejects unequal control capacity and invalid pronunciation distributions",()=> {
    const manifest={inventory:"test",tokens:"ab",sha256:"parent"} as import("../src/model.js").ModelManifest;
    expect(validateSegmentModel(model(),manifest)).toEqual(model());
    const mismatch=model();mismatch.forced.centers.pop();
    expect(()=>validateSegmentModel(mismatch,manifest)).toThrow();
    const unnormalized=model();unnormalized.independent.phones[1]=[.9,.9];
    expect(()=>validateSegmentModel(unnormalized,manifest)).toThrow();
    expect(()=>validateSegmentModel({...model(),primarySha256:"other"},manifest)).toThrow();
  });
  it("preserves primary tokens and posterior rows through certificate serialization",()=> {
    const chunks=[{inventory:"test",frameMs:40,phonemes:[{p:"a",start:0,end:1,conf:.9}],
      posteriors:{k:1,startFrame:0,nFrames:2,ids:[1,1],probs:[.9,.9]}}];
    const c={...candidate,independentWritten:1,independentRival:0,forcedWritten:0,forcedRival:1};
    const enriched=JSON.parse(JSON.stringify(carrySegmentEvidence(chunks,[c])));
    expect(enriched[0].posteriors).toEqual(chunks[0].posteriors);
    const {segmentUnitVetoes,...phone}=enriched[0].phonemes[0];
    expect(phone).toEqual(chunks[0].phonemes[0]);expect(segmentUnitVetoes).toEqual([c]);
  });
  it("uses all eligible training audio, but only none takes for pronunciation supervision",()=> {
    for(const kind of ["none","wrong_word","repeat"]) {
      expect(eligibleSegmentTake({split:"train",speaker:"s"},new Set())).toBe(true);
      expect(canonicalSegmentTake({kind,events:[],non_errors:[]})).toBe(kind==="none");
    }
    expect(eligibleSegmentTake({split:"eval",speaker:"s"},new Set())).toBe(false);
    expect(eligibleSegmentTake({split:"train",speaker:"s"},new Set(["s"]))).toBe(false);
    expect(eligibleSegmentTake({split:"train",speaker:"s",excluded:"audit"},new Set())).toBe(false);
    expect(canonicalSegmentTake({kind:"none",non_errors:[{}]})).toBe(false);
  });
  it("normalizes trajectories in time without merging the neighboring interval",()=> {
    expect(segmentDescriptor([[0],[6],[12],[99]],0,3)).toEqual([2,6,10]);
    expect(segmentDescriptor([[7]],0,1)).toEqual([7,7,7]);
  });
  it("requires positive written evidence and opposite control preference over the full same window",()=> {
    const c=segmentComparison(model(),[[0],[0]],candidate)!;
    expect(segmentReversal(c)).toBe(true);
    expect(segmentReversal({...c,forcedWritten:c.forcedRival})).toBe(false);
    expect(segmentReversal({...c,independentWritten:NaN})).toBe(false);
    expect(segmentComparison(model(),[[0]],candidate)).toBeNull();
    expect(segmentComparison(model(),[[0],[0]],{...candidate,rival:"x"})).toBeNull();
    const uncovered=model();uncovered.independent.counts[1]=0;
    expect(segmentComparison(uncovered,[[0],[0]],candidate)).toBeNull();
  });
  it("conditions the segment timing prior on completion, including repeated-phone separation",()=> {
    const m=model(),uniform={...m.independent,phones:[[.5,.5],[.5,.5],[.5,.5]]};
    expect(segmentPronunciationScore(m,uniform,[[0],[0],[0]],[1,1])).toBeCloseTo(0,12);
    expect(segmentPronunciationScore(m,uniform,[[0],[0]],[1,1])).toBe(-Infinity);
    expect(segmentPronunciationScore(m,uniform,[[0],[0]],[1,2])).toBeCloseTo(0,12);
  });
  it("matches exhaustive joint marginalization over boundaries and phone correspondences",()=> {
    const m=model(),arm=m.independent,x=[[0],[.7],[1.4]],ids=[1,2],labels=[0,1,0,2,0];
    const edges=labels.map((p,s)=>[s,...(s+1<labels.length?[s+1]:[]),...(s+2<labels.length&&labels[s+2]&&labels[s+2]!==p?[s+2]:[])]);
    let likelihood=0,timing=0;
    function visit(start:number,state:number,weight:number,score:number) {
      for(let length=1;length<=m.maxDuration&&start+length<=x.length;length++) {
        const d=segmentDescriptor(x,start,start+length),q=arm.centers.map(c=>Math.exp(-c.reduce((s,v,j)=>s+(v-d[j])**2,0)/(2*arm.variance)));
        const z=q.reduce((a,b)=>a+b,0),emission=length*Math.log(2*q.reduce((s,v,k)=>s+v/z*arm.phones[labels[state]][k],0));
        const w=weight*Math.exp(-m.penalty+length*Math.log(z/2))/m.maxDuration;
        if(start+length===x.length){if(state>=labels.length-2){timing+=w;likelihood+=w*Math.exp(score+emission);}continue;}
        for(const next of edges[state])visit(start+length,next,w/edges[state].length,score+emission);
      }
    }
    visit(0,0,1,0);visit(0,1,1,0);
    expect(segmentPronunciationScore(m,arm,x,ids)).toBeCloseTo(Math.log(likelihood/timing),12);
  });
  it("agrees with the Python training scorer at the shipped codebook capacity",()=> {
    const m=model();m.maxDuration=12;
    const arm={centers:[...Array.from({length:16},()=>[0,0,0]),...Array.from({length:16},()=>[2,2,2])],variance:.1,counts:[1,1,1],
      phones:[Array(32).fill(1/32),[...Array(16).fill(.9/16),...Array(16).fill(.1/16)],
        [...Array(16).fill(.1/16),...Array(16).fill(.9/16)]]};
    const expected=[.3094622456415017,-.4505145456703623,1.7632039187245532,-1.0216512472299684];
    [[1],[2],[1,2],[1,1]].forEach((ids,i)=>expect(segmentPronunciationScore(m,arm,[[0],[.7],[1.4]],ids)).toBeCloseTo(expected[i],11));
  });
});
