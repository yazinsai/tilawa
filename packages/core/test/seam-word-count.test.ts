import { describe, expect, it } from "vitest";
import { wordCountGraph, wordCountLogSum, wordCountSlot } from "../src/seam-word-count.js";
import { createWordCountDetector } from "../src/seam-word-count.js";
import { loadReferenceIndex } from "../src/reference.js";
import type { PhonemeEvidence } from "../src/evidence.js";
import type { PositionEvent } from "../src/events.js";
import { createChecker } from "../src/checker.js";

const frames = (rows: number[][]) => ({ startFrame: 0, endFrame: rows.length,
  row: (f: number) => Float64Array.from(rows[f], Math.log) });
const graph = (extra: boolean, repeats = false) => wordCountGraph([[[1]], [[2]], [[1]]], 3, [1,2], new Set([1,2]), extra, repeats);
function exhaustive(rows: number[][], extra: boolean): number {
  let sum = 0;
  const visit = (at: number, prev: number, text: string, p: number) => {
    if (at === rows.length) {
      if (extra ? /^13[12]{2,}3231$/.test(text) : text === "13231") sum += p;
      return;
    }
    rows[at].forEach((q, id) => visit(at + 1, id, text + (id && id !== prev ? id : ""), p * q));
  };
  visit(0, 0, "", 1);
  return Math.log(sum);
}
describe("separator-head closed word count", () => {
  it("matches exhaustive frame paths for both counts, without duplicated blank timings", () => {
    const rows = Array.from({length: 8}, (_, t) => t % 2 ? [.4,.1,.2,.3] : [.1,.4,.3,.2]);
    for (const extra of [false,true]) expect(wordCountLogSum(graph(extra), frames(rows), 0, rows.length))
      .toBeCloseTo(exhaustive(rows,extra),12);
  });
  it("cannot buy two consonants with one held run or skip either extra boundary", () => {
    const rows = (ids: number[]) => ids.map(id => Array.from({length:4}, (_, p) => Number(p === id)));
    for (const ids of [[1,3,2,2,2,3,2,3,1], [1,3,1,2,2,3,1]])
      expect(wordCountLogSum(graph(true), frames(rows(ids)), 0, ids.length)).toBeNull();
    const ids = [1,3,2,0,2,3,2,3,1];
    expect(wordCountLogSum(graph(true), frames(rows(ids)), 0, ids.length)).toBe(0);
  });
  it("explains repeated written words through separator-bearing repeats", () => {
    const ids = [1,3,1,3,2,3,1], rows = ids.map(id => Array.from({length:4}, (_, p) => Number(p === id)));
    expect(wordCountLogSum(graph(false,true), frames(rows), 0, rows.length)).toBe(0);
    expect(wordCountLogSum(graph(false), frames(rows), 0, rows.length)).toBeNull();
  });
  it("abstains on missing rows and impossible complete text", () => {
    const f = frames([[.25,.25,.25,.25]]);
    expect(wordCountLogSum(graph(false),f,-1,1)).toBeNull();
    expect(wordCountLogSum(graph(false),f,0,2)).toBeNull();
    expect(wordCountLogSum(graph(false),f,0,1)).toBeNull();
  });
  it("extracts the accepted slot, collapses holds, and retains blank-separated consonants", () => {
    const ids = [1,3,2,2,0,2,3,2,3,1];
    const f = frames(ids.map(id => Array.from({length:4}, (_, p) => Number(p === id))));
    expect(wordCountSlot(graph(true),f,0,ids.length)).toEqual({phones:[2,2],fromFrame:2,toFrame:6});
    expect(wordCountSlot(graph(false),f,0,ids.length)).toBeNull();
    expect(wordCountSlot(graph(true),f,0,ids.length+1)).toBeNull();
  });
});

describe("stored separator head on primary tracker steps", () => {
  const index = loadReferenceIndex({version:"test",text:"test",profile:{},inventory:{id:"test",tokens:"بتسق"},
    ayahs:[{s:1,a:1,w:["left"],p:"ب",o:[0],iso:["ب"]},
      {s:1,a:2,w:["first","second"],p:"تب",o:[0,1],iso:["ت","ب"]}]});
  const position = (word:number,frame:number): PositionEvent => ({type:"position",v:1,timeMs:frame*40,
    referenceVersion:"test",expected:index.words[word],state:"locked",observed:[{word:index.words[word],
      span:{start:index.words[0],end:index.words[word]},confidence:1,cost:0,per:0}]});
  const stream = (cut=0): PhonemeEvidence => {
    const labels=[1,5,3,0,4,5,2,5,1];
    return {inventory:"test",frameMs:40,phonemes:[{p:"ب",start:0,end:0,conf:1,
      wordCountPosteriors:[{tokens:"بتسق|",cleanMaxMargin:cut,startFrame:0,nFrames:labels.length,
        logProbs:labels.flatMap(id=>Array.from({length:6},(_,p)=>p===id ? -0.01 : -20))}]}]};
  };
  it("finds an unnamed two-consonant slot only after all three consecutive written steps", () => {
    const detector=createWordCountDetector(index),e=stream();
    expect(detector.push(e,[position(0,0),position(1,6)])).toEqual([]);
    const got=detector.push({...e,phonemes:[]},[position(2,9)]);
    expect(got).toHaveLength(1);
    expect(got[0]).toMatchObject({word:1,observed:"سق",fromFrame:2,toFrame:5});
    expect(detector.push({...e,phonemes:[]},[position(2,10)])).toEqual([]);
  });
  it("requires strict clean-envelope separation and continuous head coverage", () => {
    const steps=[position(0,0),position(1,6),position(2,9)];
    const first=createWordCountDetector(index).push(stream(),steps)[0];
    expect(createWordCountDetector(index).push(stream(first.margin),steps)).toEqual([]);
    const missing=stream();
    missing.phonemes[0].wordCountPosteriors![0].startFrame=1;
    expect(createWordCountDetector(index).push(missing,steps)).toEqual([]);
    expect(createWordCountDetector(index).push({...missing,phonemes:[]},steps)).toEqual([]);
  });
  it("does not join steps across a lock loss, and survives ordinary JSON evidence transport", () => {
    const detector=createWordCountDetector(index),e=stream();
    detector.push(e,[position(0,0)]);
    detector.push({...e,phonemes:[]},[{...position(1,5),state:"lost",expected:null}]);
    expect(detector.push({...e,phonemes:[]},[position(1,6),position(2,9)])).toEqual([]);
    const transported=JSON.parse(JSON.stringify(e));
    expect(createWordCountDetector(index).push(transported,[position(0,0),position(1,6),position(2,9)])).toHaveLength(1);
  });
  it("confirms through the checker with no primary surplus and retains its non-Quran guard", () => {
    const head=stream().phonemes[0].wordCountPosteriors;
    const evidence:PhonemeEvidence={inventory:"test",frameMs:40,phonemes:[
      {p:"ب",start:0,end:0,conf:1},{p:"ت",start:6,end:6,conf:1},{p:"ب",start:8,end:8,conf:1}],
      posteriors:{k:5,startFrame:0,nFrames:9,ids:Array.from({length:9},()=>[0,1,2,3,4]).flat(),
        probs:Array.from({length:9},(_,t)=>[0,1,2,3,4].map(p=>Number(p===(t===0||t===8?1:t===6?2:0)))).flat()}};
    const run=(withHead:boolean,phrase=false)=>{
      const checker=createChecker(index,{passage:{from:index.words[0],to:index.words[2]},
        phrases:{version:"test",phrases:phrase?[{name:"formula",text:"formula",p:"سق"}]:[]}});
      const e={...evidence,phonemes:evidence.phonemes.map((p,i)=>!i&&withHead?{...p,wordCountPosteriors:head}:p)};
      const events=[...checker.push(e,[position(0,0),position(1,6),position(2,9)]),...checker.end()];
      return events.filter(e=>e.state==="confirmed"&&e.errorClass==="added_word");
    };
    expect(run(false)).toEqual([]);
    expect(run(true)).toEqual([expect.objectContaining({location:expect.objectContaining({ayah:2,word:1}),observed:"سق"})]);
    expect(run(true,true)).toEqual([]);
  });
});
