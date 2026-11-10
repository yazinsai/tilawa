import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import type { AcousticRadiusCandidate } from "../src/checker.js";
import type { AcousticRadiusEvidence } from "../src/evidence.js";
import type { CheckerEvent } from "../src/events.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import { parallelPaths } from "../src/parallel-phrases.js";

const index=loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json",import.meta.url),"utf8")));
const from={surah:26,ayah:142,word:1},to={surah:26,ayah:143,word:3};
const target=index.words.find(w=>w.surah===26&&w.ayah===142&&w.word===5)!;
const start=index.words.find(w=>w.surah===from.surah&&w.ayah===from.ayah&&w.word===from.word)!.start;
const fixture=()=>syntheticFixture(index,{id:"radius-certificate-unit",from,to,chunkSize:4,edits:[
  {kind:"del",at:target.start-start,count:target.end-target.start},{kind:"ins",at:target.start-start,text:"نُۥۥح"},
]}).evidence.map(e=>withPosteriors(index,e));
function run(metadata:AcousticRadiusEvidence[]=[], discover?:(c:AcousticRadiusCandidate)=>boolean) {
  const checker=createChecker(index,{passage:{from,to},policy:{confirm:{after:"next-word"}},acousticRadius:discover});
  const chunks=fixture(), events:CheckerEvent[]=[];
  chunks[0].phonemes[0]={...chunks[0].phonemes[0],acousticRadii:metadata};
  for (const e of chunks) events.push(...checker.push(e,[]));
  events.push(...checker.end());
  return {events,mistakes:checker.mistakes()};
}
function certificates() {
  const candidates:AcousticRadiusCandidate[]=[];
  run([],c=>{candidates.push(c);return true;});
  expect(candidates.length).toBeGreaterThan(0);
  return candidates.map(c=>({...c,baselineMargin:c.margin,boundaryMargin:c.need-1,energy:.1,rejectionEnergy:.2}));
}
describe("listener radius veto certificates",()=> {
  it("discovery leaves every event unchanged and a veto changes only the final state",()=> {
    const baseline=run(), seen:AcousticRadiusCandidate[]=[];
    expect(baseline.mistakes.some(e=>e.errorClass==="wrong_word")).toBe(true);
    expect(run([],c=>{seen.push(c);return true;})).toEqual(baseline);
    const vetoed=run(certificates());
    expect(vetoed.events).toEqual(baseline.events.map(e=>e.state==="confirmed"&&e.errorClass==="wrong_word"?{...e,state:"uncertain"}:e));
    expect(vetoed.mistakes.some(e=>e.errorClass==="wrong_word")).toBe(false);
  });
  it("rejects unsupported, changed-window or changed-hypothesis certificates",()=> {
    const valid=certificates(), baseline=run();
    for(const change of [
      (c:AcousticRadiusEvidence)=>({...c,energy:c.rejectionEnergy}),
      (c:AcousticRadiusEvidence)=>({...c,boundaryMargin:c.need}),
      (c:AcousticRadiusEvidence)=>({...c,baselineMargin:c.need-1}),
      (c:AcousticRadiusEvidence)=>({...c,energy:NaN}),
      (c:AcousticRadiusEvidence)=>({...c,toFrame:c.toFrame+1}),
      (c:AcousticRadiusEvidence)=>({...c,accused:c.accused+"ب"}),
      (c:AcousticRadiusEvidence)=>({...c,need:c.need+1}),
    ]) expect(run(valid.map(change))).toEqual(baseline);
  });
  it("preserves an attested parallel-phrase graph confirmation",()=> {
    const position=index.words.findIndex(w=>w.surah===84&&w.ayah===25&&w.word===1);
    const path=parallelPaths(index).get(position)!.find(p=>p.altWord==="ءِننننَ")!;
    const wi=path.word,alt=path.altWord;
    const first=index.words[path.start],last=index.words[path.start+5],changed=index.words[wi];
    const chunks=syntheticFixture(index,{id:"radius-parallel-preservation",from:first,to:last,chunkSize:6,edits:[
      {kind:"del",at:changed.start-first.start,count:changed.end-changed.start},
      {kind:"ins",at:changed.start-first.start,text:alt},
    ]}).evidence.map(e=>withPosteriors(index,e,j=>e.phonemes[j].p==="ن" ? [["ن",.6],["ل",.05],["",.3]] : undefined));
    const calls:AcousticRadiusCandidate[]=[];
    const checker=createChecker(index,{passage:{from:first,to:last},policy:{confirm:{after:"delay",ms:60000}},
      acousticRadius(c){calls.push(c);return false;}});
    for(const e of chunks)checker.push(e,[]);checker.end();
    expect(checker.mistakes().some(e=>e.errorClass==="wrong_word")).toBe(true);
    expect(calls.length).toBe(0);
  });
  it("preserves an insertion when all string-path substitution decisions are vetoed",()=> {
    const first=index.words.find(w=>w.surah===2&&w.ayah===255&&w.word===1)!;
    const last=index.words.find(w=>w.surah===2&&w.ayah===256&&w.word===10)!;
    const where=index.words.find(w=>w.surah===2&&w.ayah===255&&w.word===23)!;
    const alien=index.words.find(w=>w.surah===108&&w.ayah===1&&w.word===3)!;
    const chunks=syntheticFixture(index,{id:"radius-insertion-preservation",from:first,to:last,chunkSize:8,edits:[
      {kind:"ins",at:where.start-first.start,text:index.phonemes.slice(alien.start,alien.end)},
    ]}).evidence.map(e=>withPosteriors(index,e));
    function check(reject:boolean) {
      const checker=createChecker(index,{passage:{from:first,to:last},...(reject&&{acousticRadius:()=>false})});
      for(const e of chunks)checker.push(e,[]);checker.end();return checker.mistakes();
    }
    const baseline=check(false);
    expect(baseline.some(e=>e.errorClass==="added_word")).toBe(true);
    expect(check(true)).toEqual(baseline);
  });
});
