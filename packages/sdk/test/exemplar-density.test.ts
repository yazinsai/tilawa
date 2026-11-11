import { describe, expect, it } from "vitest";
import { exemplarComparison, exemplarRows, exemplarSequenceScore, validateExemplarModel } from "../src/exemplar-density.js";
import type { ExemplarModel } from "../src/exemplar-density.js";
import type { ModelManifest } from "../src/model.js";

const bank = (): ExemplarModel => ({schema:1,inventory:"test",tokens:"ab",primarySha256:"abc",output:"features",
  center:[0],projection:[[1]],bank:[[-1],[1],[-1],[1],[-1],[1]],
  soft:[[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]],recording:[0,0,1,1,2,2],
  weights:[.5,.5,.5,.5,.5,.5],frequencies:[1.5,1.5],recordingCount:3,k:2,alpha:.01});
const manifest={inventory:"test",tokens:"ab",sha256:"abc"} as ModelManifest;
const c={word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:0,toFrame:1};

describe("soft-labelled exemplar density",()=> {
  it("compares acoustic label concentration and preserves blank evidence",()=> {
    const rows=exemplarRows(bank(),[[-1]],[[Math.log(.3),Math.log(.1),Math.log(.6)]]);
    expect(rows[0][0]).toBe(Math.log(.3));
    expect(rows[0].reduce((s,p)=>s+Math.exp(p),0)).toBeCloseTo(1,12);
    expect(rows[0][1]).toBeGreaterThan(rows[0][2]);
    const score=exemplarComparison(bank(),[[-1]],[[Math.log(.3),Math.log(.1),Math.log(.6)]],c)!;
    expect(score.headWritten).toBeLessThan(score.headRival);
    expect(score.exemplarWritten).toBeGreaterThan(score.exemplarRival);
  });
  it("is independent of candidate ordering",()=> {
    const lp=[[Math.log(.1),Math.log(.2),Math.log(.7)]];
    const a=exemplarComparison(bank(),[[-1]],lp,c)!;
    const b=exemplarComparison(bank(),[[-1]],lp,{...c,written:c.rival,rival:c.written})!;
    expect(a.exemplarWritten).toBe(b.exemplarRival);
    expect(a.exemplarRival).toBe(b.exemplarWritten);
  });
  it("corrects class frequencies rather than favouring the populous label",()=> {
    const m=bank();m.soft=m.soft.map(()=>[.9,.1]);m.frequencies=[2.7,.3];
    const rows=exemplarRows(m,[[0]],[[Math.log(.2),Math.log(.4),Math.log(.4)]]);
    expect(rows[0][1]).toBeCloseTo(rows[0][2],12);
  });
  it("abstains when a complete pronunciation needs unsupported phones",()=> {
    const m=bank();m.frequencies[0]=0;
    expect(exemplarComparison(m,[[0]],[[0,-2,-2]],c)).toBeNull();
    expect(exemplarComparison(bank(),[[0]],[[0,-2,-2]],{...c,written:"z"})).toBeNull();
  });
  it("checks recording budgets, soft-label frequencies and graph identity",()=> {
    expect(validateExemplarModel(bank(),manifest)).toEqual(bank());
    const m=bank();m.weights[0]=2;
    expect(()=>validateExemplarModel(m,manifest)).toThrow();
    expect(()=>validateExemplarModel({...bank(),primarySha256:"other"},manifest)).toThrow();
  });
});

describe("complete-pronunciation boundary marginal",()=> {
  it("equals exhaustive CTC path summation, including repeated labels",()=> {
    const probabilities=[[.2,.5,.3],[.4,.2,.4],[.3,.6,.1]];
    const rows=probabilities.map(r=>r.map(Math.log));
    for(const target of [[1],[1,1],[1,2]]) {
      let sum=0;
      for(let a=0;a<3;a++)for(let b=0;b<3;b++)for(let c=0;c<3;c++) {
        const path=[a,b,c],collapsed=path.filter((v,i)=>v!==0&&(i===0||v!==path[i-1]));
        if(JSON.stringify(collapsed)===JSON.stringify(target))sum+=probabilities[0][a]*probabilities[1][b]*probabilities[2][c];
      }
      expect(Math.exp(exemplarSequenceScore(rows,target))).toBeCloseTo(sum,12);
    }
    expect(exemplarSequenceScore(rows,[1,1,1])).toBe(-Infinity);
  });
});
