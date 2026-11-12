import { describe, expect, it } from "vitest";
import { boundedComparison, eligibleBoundedTake, sequenceBrier, sequenceCtcScore, validateBoundedModel } from "../src/bounded-sequence.js";
import type { BoundedModel } from "../src/bounded-sequence.js";
import type { ModelManifest } from "../src/model.js";

describe("bounded complete CTC sequence readouts",()=> {
  it("matches exhaustive collapsed-path mass, including empty and repeated sequences",()=> {
    const probabilities=[[.5,.3,.2],[.2,.6,.2],[.3,.2,.5],[.4,.4,.2]];
    for(const labels of [[],[1],[1,2],[1,1]]) {
      let mass=0;
      for(let code=0;code<3**probabilities.length;code++) {
        let n=code,previous=-1,p=1; const collapsed:number[]=[];
        for(const row of probabilities) {
          const id=n%3;n=Math.floor(n/3);p*=row[id];
          if(id!==previous&&id!==0)collapsed.push(id);previous=id;
        }
        if(JSON.stringify(collapsed)===JSON.stringify(labels))mass+=p;
      }
      expect(sequenceCtcScore(probabilities.map(r=>r.map(Math.log)),labels)).toBeCloseTo(Math.log(mass),12);
    }
    expect(sequenceCtcScore([[Math.log(.5),Math.log(.5)]],[1,1])).toBe(-Infinity);
    expect(sequenceCtcScore([[0,-Infinity]],[0])).toBe(-Infinity);
  });
  it("keeps unnamed mass in a bounded proper bucket without conditioning on candidates",()=> {
    expect(sequenceBrier([.2,.3])).toBeCloseTo(.64+.09+.25,12);
    expect(sequenceBrier([1,0])).toBe(0);
    expect(sequenceBrier([0,1])).toBe(2);
    expect(sequenceBrier([0,0])).toBe(2);
    expect(()=>sequenceBrier([.8,.8])).toThrow();
  });
  it("requires a full-pronunciation reversal in both independently trained candidate-set pairs",()=> {
    const head=(bias:number[])=>({weight:bias.map(()=>[0]),bias});
    const written=head([0,3,0,-5]),rival=head([0,0,3,-5]);
    const m:BoundedModel={schema:1,inventory:"test",tokens:"بنَ",width:1,featureOutput:"h",logitOutput:"z",primarySha256:"test",
      center:[0],scale:[1],candidateLimits:[8,32],bounded:written,control:rival,enlargedBounded:written,enlargedControl:rival};
    const c={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ن",fromFrame:0,toFrame:2,writtenParts:["ب"],rivalParts:["ن"]};
    expect(boundedComparison(m,[[0],[0]],c)?.veto).toBe(true);
    expect(boundedComparison({...m,control:written},[[0],[0]],c)?.veto).toBe(false);
    expect(boundedComparison({...m,enlargedBounded:rival},[[0],[0]],c)?.veto).toBe(false);
    expect(boundedComparison({...m,enlargedControl:written},[[0],[0]],c)?.veto).toBe(false);
    expect(boundedComparison(m,[[0]],c)).toBeNull();
    expect(boundedComparison(m,[[0],[0]],{...c,rival:"?",rivalParts:["?"]})?.veto).toBe(false);
    expect(boundedComparison(m,[[0],[0]],{...c,rivalParts:["ب"]})).toBeNull();
    const manifest={inventory:m.inventory,tokens:m.tokens,sha256:m.primarySha256} as ModelManifest;
    expect(validateBoundedModel(m,manifest)).toBe(m);
    expect(()=>validateBoundedModel({...m,enlargedControl:{...rival,bias:[0]}},manifest)).toThrow();
    expect(()=>validateBoundedModel({...m,primarySha256:"other"},manifest)).toThrow();
    expect(()=>validateBoundedModel({...m,candidateLimits:[8,8]},manifest)).toThrow();
  });
  it("accepts only usable none-labelled training-speaker takes",()=> {
    const r={kind:"none",split:"train",speaker:"a"};
    expect(eligibleBoundedTake(r,new Set())).toBe(true);
    for(const changed of [{split:"eval"},{speaker:""},{exclude:true},{events:[{}]},{non_errors:[{}]},{scored:false}])
      expect(eligibleBoundedTake({...r,...changed},new Set())).toBe(false);
    expect(eligibleBoundedTake(r,new Set(["a"]))).toBe(false);
  });
});
