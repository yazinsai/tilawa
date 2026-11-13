import { describe,expect,it } from "vitest";
import { trainingPronunciations,transcriptLogProbability } from "../src/acoustic-span-training.js";
import type { ReferenceIndex } from "@tilawa/core";
const index={phonemes:"بتت",words:[{surah:1,ayah:1,word:1,start:0,end:1},{surah:1,ayah:1,word:2,start:1,end:2},
  {surah:2,ayah:1,word:1,start:2,end:3}],wordText:["first","second","rival"],isolatedWordForms:["ب","ت","ت"]} as ReferenceIndex;
describe("recorded training pronunciations",()=>{
  it("accepts none-labelled complete readings and keeps substitutions as their spoken candidates",()=>{
    expect(trainingPronunciations(index,{kind:"none",ayahs:["1:1"]})?.recorded).toEqual(["ب","ت"]);
    const transcript=trainingPronunciations(index,{kind:"substitution",ayahs:["1:1"],events:[{
      errorClass:"wrong_word",state:"confirmed",location:{surah:1,ayah:1,word:1},observed:"rival"}]});
    expect(transcript?.written).toEqual(["ب","ت"]);
    expect(transcript?.recorded).toEqual(["ت","ت"]);
    expect(transcript?.kind).toBe("supported-substitution");
  });
  it("rejects omitted sounds, missing performed spellings and labels with unsupported deviations",()=>{
    for(const errorClass of ["omitted_word","wrong_word"])expect(trainingPronunciations(index,{kind:"substitution",ayahs:["1:1"],
      events:[{errorClass,state:"confirmed",location:{surah:1,ayah:1,word:1}}]})).toBeNull();
    expect(trainingPronunciations(index,{kind:"none",ayahs:["1:1"],non_errors:[{kind:"repeat"}]})).toBeNull();
  });
  it("uses complete transcript mass and cannot omit adjacent repeated phones",()=>{
    const lp=[[.1,.1,.8],[.9,.05,.05],[.1,.1,.8]].map(r=>r.map(Math.log));
    expect(transcriptLogProbability(lp,[2,2])).toBeCloseTo(Math.log(.8*.9*.8),12);
    expect(transcriptLogProbability(lp,[2,2])).toBeGreaterThan(transcriptLogProbability(lp,[1,2]));
    expect(transcriptLogProbability(lp.slice(0,2),[2,2])).toBe(-Infinity);
  });
});
