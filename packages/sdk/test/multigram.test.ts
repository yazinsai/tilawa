import { describe, expect, it } from "vitest";
import { eligibleMultigramTake, multigramCtcScore, multigramComparison, validateMultigramModel } from "../src/multigram.js";
import type { MultigramModel } from "../src/multigram.js";
import type { ModelManifest } from "../src/model.js";

function brute(rows:number[][], text:string, units:string[]):number {
  let total=0;
  const walk=(t:number,last:number,spoken:string,p:number)=> {
    if(t===rows.length) {if(spoken===text)total+=p; return;}
    rows[t].forEach((v,u)=>walk(t+1,u,spoken+(u && u!==last?units[u-1]:""),p*Math.exp(v)));
  };
  walk(0,0,"",1);
  return Math.log(total);
}
describe("segmentation-marginalized multigram CTC",()=> {
  it("equals exhaustive acoustic-path enumeration, including every unit segmentation",()=> {
    const units=["a","b","ab","aa"];
    for(let n=1;n<=4;n++) {
      const rows=Array.from({length:n},(_,t)=>[.1+t*.01,.2,.3-t*.01,.25,.15].map(Math.log));
      for(const text of ["a","b","ab","aa","aab","abab","aaaa","ba"]) {
        const expected=brute(rows,text,units),actual=multigramCtcScore(rows,text,units);
        if(expected===-Infinity)expect(actual).toBe(-Infinity);
        else expect(actual).toBeCloseTo(expected,10);
      }
    }
  });
  it("sums singleton and multiphone segmentations without picking one",()=> {
    const rows=[[.2,.25,.25,.3],[.2,.25,.25,.3]].map(r=>r.map(Math.log));
    const all=multigramCtcScore(rows,"ab",["a","b","ab"]);
    expect(all).toBeGreaterThan(Math.log(.25*.25));
    expect(all).toBeCloseTo(Math.log(.25*.25+.3*.3+2*.2*.3),10);
  });
  it("requires blank between repeated identical units and accepts only complete pronunciations",()=> {
    const row=[Math.log(.1),Math.log(.9)];
    expect(multigramCtcScore([row,row],"aa",["a"])).toBe(-Infinity);
    expect(multigramCtcScore([row,row,row],"aa",["a"])).toBeCloseTo(Math.log(.9*.1*.9),10);
    expect(multigramCtcScore([[.1,.8,.1].map(Math.log)],"ab",["a","b"])).toBe(-Infinity);
    expect(multigramCtcScore([],"a",["a"])).toBe(-Infinity);
    expect(multigramCtcScore([row],"",["a"])).toBe(-Infinity);
    expect(multigramCtcScore([row],"z",["a"])).toBe(-Infinity);
  });
  it("reduces to ordinary phone CTC for a singleton inventory",()=> {
    const units=["a","b"],rows=Array.from({length:4},()=>[.2,.5,.3].map(Math.log));
    for(const text of ["a","ab","aa","aba"])expect(multigramCtcScore(rows,text,units)).toBeCloseTo(brute(rows,text,units),10);
  });
  it("keeps unknown phones, incomplete windows and non-reversals from issuing certificates",()=> {
    const head={input:[[0]],inputBias:[0],output:[[0],[0],[0]],outputBias:[0,1,0]};
    const m={schema:1,inventory:"test",tokens:"ab",width:1,featureOutput:"features",logitOutput:"logits",
      primarySha256:"x",units:["a","b"],center:[0],scale:[1],multigram:head,control:head} as MultigramModel;
    const c={word:{surah:1,ayah:1,word:1},written:"a",rival:"b",writtenParts:["a"],rivalParts:["b"],fromFrame:0,toFrame:1};
    expect(multigramComparison(m,[[0]],c)?.veto).toBe(false);
    expect(multigramComparison(m,[[0]],{...c,written:"z",writtenParts:["z"]})).toBeNull();
    expect(multigramComparison(m,[[0]],{...c,toFrame:2})).toBeNull();
    expect(multigramComparison(m,[[0]],{...c,rivalParts:["a"]})).toBeNull();
    expect(()=>validateMultigramModel({...m,units:["a","a"]},{tokens:"ab",inventory:"test",sha256:"x"} as ModelManifest)).toThrow();
  });
  it("issues a certificate only when the complete multigram preference reverses the phone control",()=> {
    const head=(bias:number[])=>({input:[[0]],inputBias:[0],output:bias.map(()=>[0]),outputBias:bias});
    const m={schema:1,inventory:"test",tokens:"ab",width:1,featureOutput:"features",logitOutput:"logits",
      primarySha256:"x",units:["a","b","ab"],center:[0],scale:[1],
      multigram:head([0,0,0,4]),control:head([0,4,0,-4])} as MultigramModel;
    const c={word:{surah:1,ayah:1,word:1},written:"ab",rival:"a",writtenParts:["ab"],rivalParts:["a"],fromFrame:0,toFrame:2};
    const result=multigramComparison(m,[[0],[0]],c)!;
    expect(result.veto).toBe(true);
    expect(result.multigramWritten).toBeGreaterThan(result.multigramRival);
    expect(result.controlWritten).toBeLessThan(result.controlRival);
    expect(multigramComparison({...m,control:head([0,0,4,-4])},[[0],[0]],c)?.veto).toBe(false);
  });
  it("uses only usable none-labelled training speakers",()=> {
    const take={kind:"none",split:"train",speaker:"training",events:[],non_errors:[]};
    expect(eligibleMultigramTake(take,new Set())).toBe(true);
    for(const change of [{kind:"wrong_word"},{split:"test"},{exclude:true},{non_errors:[{}]},{events:[{}]},{scored:false}])
      expect(eligibleMultigramTake({...take,...change},new Set())).toBe(false);
    expect(eligibleMultigramTake(take,new Set(["training"]))).toBe(false);
  });
});
