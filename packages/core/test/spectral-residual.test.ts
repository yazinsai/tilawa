import { readFileSync } from "node:fs";
import { describe,expect,it } from "vitest";
import { createChecker } from "../src/checker.js";
import type { SpectralResidualRequest } from "../src/evidence.js";
import type { SpectralResidualEvidence } from "../src/evidence.js";
import { syntheticFixture,withPosteriors } from "../src/fixtures/index.js";
import { loadReferenceIndex } from "../src/reference.js";

const index=loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json",import.meta.url),"utf8")));
const from={surah:2,ayah:255,word:18},to={surah:2,ayah:255,word:30},target={surah:2,ayah:255,word:23};
const wi=index.words.findIndex(w=>w.surah===target.surah && w.ayah===target.ayah && w.word===target.word);
const word=index.words[wi],first=index.words.find(w=>w.surah===from.surah && w.ayah===from.ayah && w.word===from.word)!;
const rival=index.words.find(w=>w.surah===108 && w.ayah===1 && w.word===3)!;
const fixture=syntheticFixture(index,{id:"within-band residual-certificate",from,to,chunkSize:4,edits:[
 {kind:"del",at:word.start-first.start,count:word.end-word.start},
 {kind:"ins",at:word.start-first.start,text:index.phonemes.slice(rival.start,rival.end)},
]});
function run(metadata:SpectralResidualEvidence[]=[],candidate?:(c:SpectralResidualRequest)=>boolean) {
 const checker=createChecker(index,{passage:{from,to},policy:{confirm:{after:"next-word"}},spectralResidual:candidate});
 for(const [i,e] of fixture.evidence.entries()) {
  const enriched=withPosteriors(index,e);
  if(!i && enriched.phonemes.length) enriched.phonemes[0]={...enriched.phonemes[0],spectralResiduals:metadata};
  checker.push(JSON.parse(JSON.stringify(enriched)),[]);
 }
 checker.end();return checker.mistakes();
}
function certificate() {
 const candidates:SpectralResidualRequest[]=[];
 expect(run([],c=>{candidates.push(c);return true;}).some(e=>e.errorClass==="wrong_word")).toBe(true);
 const c=candidates.find(c=>c.word===wi)!;
 expect(c).toBeDefined();
 return {...c,word:target,writtenLogLikelihood:-1,accusedLogLikelihood:-2};
}
describe("within-band residual acoustic listener certificate",()=>{
 it("vetoes the matching string accusation through ordinary JSON evidence",()=>{
  const c=certificate();
  expect(run([c])).toEqual([]);
 });
 it("requires strict finite preference and the exact slot and hypotheses",()=>{
  const c=certificate();
  for(const mismatch of [{...c,fromFrame:c.fromFrame+1},{...c,slotFromFrame:c.slotFromFrame+1},{...c,toFrame:c.toFrame-1},{...c,accused:"ب"},{...c,written:"ت"},
    {...c,writtenLogLikelihood:-2},{...c,writtenLogLikelihood:Infinity},{...c,accusedLogLikelihood:-1},{...c,accusedLogLikelihood:NaN}])
    expect(run([mismatch]).some(e=>e.errorClass==="wrong_word")).toBe(true);
 });
 it("abstains when evidence is absent",()=>{
  expect(run().some(e=>e.errorClass==="wrong_word")).toBe(true);
 });
});

it("keeps a parallel-graph confirmation when every string-path request is vetoed",()=>{
 const graphFrom={surah:37,ayah:108,word:1},graphTarget={surah:37,ayah:108,word:2};
 const graphWord=index.words.find(w=>w.surah===37&&w.ayah===108&&w.word===2)!;
 const graphFirst=index.words.find(w=>w.surah===37&&w.ayah===108&&w.word===1)!;
 const graphRival=index.words.find(w=>w.surah===37&&w.ayah===119&&w.word===2)!;
 const end=index.words[index.words.indexOf(graphFirst)+5];
 const f=syntheticFixture(index,{id:"within-band residual-graph-preservation",from:graphFrom,to:end,chunkSize:6,edits:[
  {kind:"del",at:graphWord.start-graphFirst.start,count:graphWord.end-graphWord.start},
  {kind:"ins",at:graphWord.start-graphFirst.start,text:index.phonemes.slice(graphRival.start,graphRival.end)},
 ]});
 const checker=createChecker(index,{passage:{from:graphFrom,to:end},policy:{confirm:{after:"next-word"}},spectralResidual:()=>false});
 for(const e of f.evidence)checker.push(withPosteriors(index,e),[]);
 checker.end();
 expect(checker.mistakes().some(e=>e.errorClass==="wrong_word"&&e.location.surah===graphTarget.surah&&
  e.location.ayah===graphTarget.ayah&&e.location.word===graphTarget.word)).toBe(true);
});
