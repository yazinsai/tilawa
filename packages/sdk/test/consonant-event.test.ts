import { describe, expect, it } from "vitest";
import { consonantComparison, consonantCtcScore, consonantEvents, eligibleConsonantTake, validateConsonantModel } from "../src/consonant-event.js";
import type { ConsonantModel } from "../src/consonant-event.js";
import type { ModelManifest } from "../src/model.js";

describe("consonant articulation transcripts",()=> {
  it("omits vowels and duration while preserving repeated articulations",()=> {
    expect(consonantEvents(["ررَحمَاانِ", "ننننَ", "مِںںں"])).toBe("رحمننمن");
    expect(consonantEvents(["بَابِ", "بُب", "نن", "ن"])).toBe("ببببنن");
    expect(consonantEvents(["۾۾۾", "وَ", "يَ"])).toBe("موي");
  });
  it("uses only usable none-labelled training speakers",()=> {
    const r={kind:"none",split:"train",speaker:"a"};
    expect(eligibleConsonantTake(r,new Set())).toBe(true);
    for(const changed of [{split:"eval"},{speaker:""},{exclude:true},{events:[{}]},{non_errors:[{}]},{scored:false}])
      expect(eligibleConsonantTake({...r,...changed},new Set())).toBe(false);
    expect(eligibleConsonantTake(r,new Set(["a"]))).toBe(false);
  });
});

describe("complete-sequence CTC comparison",()=> {
  it("matches exhaustive CTC paths including the blank required between repeated identities",()=> {
    const probabilities=[[.5,.3,.2],[.2,.6,.2],[.3,.2,.5],[.4,.4,.2]];
    for(const labels of [[1],[1,2],[1,1]]) {
      let mass=0;
      for(let code=0;code<3**probabilities.length;code++) {
        let n=code,previous=-1,p=1; const collapsed:number[]=[];
        for(const row of probabilities) {
          const id=n%3;n=Math.floor(n/3);p*=row[id];
          if(id!==previous&&id!==0)collapsed.push(id);previous=id;
        }
        if(JSON.stringify(collapsed)===JSON.stringify(labels))mass+=p;
      }
      expect(consonantCtcScore(probabilities.map(r=>r.map(Math.log)),labels)).toBeCloseTo(Math.log(mass),12);
    }
    expect(consonantCtcScore([[Math.log(.5),Math.log(.5)]],[1,1])).toBe(-Infinity);
  });
  it("requires opposite preferences and abstains on identical consonants or inconsistent windows",()=> {
    const head=(bias:number[])=>({input:[[0]],inputBias:[0],output:bias.map(()=>[0]),outputBias:bias});
    const m:ConsonantModel={schema:1,inventory:"test",tokens:"بنَ",width:1,featureOutput:"h",logitOutput:"z",primarySha256:"test",
      center:[0],scale:[1],consonant:head([0,3,0,-5]),control:head([0,0,3,-5])};
    const c={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ن",fromFrame:0,toFrame:2,writtenParts:["ب"],rivalParts:["ن"]};
    expect(consonantComparison(m,[[0],[0]],c)?.veto).toBe(true);
    expect(consonantComparison({...m,control:m.consonant},[[0],[0]],c)?.veto).toBe(false);
    expect(consonantComparison(m,[[0],[0]],{...c,rival:"بَ",rivalParts:["بَ"]})).toBeNull();
    expect(consonantComparison(m,[[0]],c)).toBeNull();
    expect(consonantComparison(m,[[0],[0]],{...c,rivalParts:["ب"]})).toBeNull();
    const manifest={inventory:m.inventory,tokens:m.tokens,sha256:m.primarySha256} as ModelManifest;
    expect(validateConsonantModel(m,manifest)).toBe(m);
    expect(()=>validateConsonantModel({...m,control:{...m.control,inputBias:[0,0]}},manifest)).toThrow();
    expect(()=>validateConsonantModel({...m,primarySha256:"other"},manifest)).toThrow();
  });
});
