import { readFileSync } from "node:fs";
import { describe,expect,it } from "vitest";
import { createChecker,loadReferenceIndex } from "../src/index.js";
import { syntheticFixture,withPosteriors } from "../src/fixtures/index.js";
import type { PauseMaskEvidence,PhonemeEvidence,CheckerEvent,WordRef } from "../src/index.js";
const index=loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json",import.meta.url),"utf8")));
const from={surah:2,ayah:255,word:18},to={surah:2,ayah:255,word:30};
const find=(r:WordRef)=>index.words.find(w=>w.surah===r.surah&&w.ayah===r.ayah&&w.word===r.word)!;
it("keeps a primary wrong-word accusation even when the side review blanks its evidence",()=>{
  const target={surah:2,ayah:255,word:23},word=find(target),alien=find({surah:108,ayah:1,word:3});
  const fixture=syntheticFixture(index,{id:"isolation",from,to,chunkSize:8,edits:[
    {kind:"del",at:word.start-find(from).start,count:word.end-word.start},
    {kind:"ins",at:word.start-find(from).start,text:index.phonemes.slice(alien.start,alien.end)},
  ]});
  const chunks=fixture.evidence.map(e=>withPosteriors(index,e));
  const phones=chunks.flatMap(e=>e.phonemes),first=phones[0].start,last=phones.at(-1)!.end+1,k=chunks[0].posteriors!.k;
  const scope:PauseMaskEvidence={inventory:index.inventoryId,frameMs:chunks[0].frameMs,fromFrame:first,toFrame:last,phonemes:[],
    posteriors:{startFrame:first,nFrames:last-first,k,ids:Array.from({length:last-first},()=>Array.from({length:k},(_,i)=>i)).flat(),
      probs:Array.from({length:last-first},()=>[1,...Array(k-1).fill(0)]).flat()}};
  const enriched=chunks.map((e,i)=>i?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,pauseMaskEvidence:[scope]})});
  const run=(input:PhonemeEvidence[])=>{
    const checker=createChecker(index,{passage:{from,to}}),events:CheckerEvent[]=[];
    for(const e of input)events.push(...checker.push(e,[]));
    events.push(...checker.end());return {events,mistakes:checker.mistakes()};
  };
  const original=structuredClone(enriched),baseline=run(chunks),candidate=run(enriched);
  expect(baseline.mistakes.some(e=>e.errorClass==="wrong_word")).toBe(true);
  expect(candidate).toEqual(baseline);expect(enriched).toEqual(original);
});
