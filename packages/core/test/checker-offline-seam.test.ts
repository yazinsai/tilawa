import { expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { loadReferenceIndex } from "../src/reference.js";
import type { PhonemeEvidence, SeamSpellingEvidence } from "../src/evidence.js";

const index = loadReferenceIndex({version:"test",text:"test",profile:{},inventory:{id:"test",tokens:"بتثجدحخزسم"},
  ayahs:Array.from({length:8},(_,i) => ({s:1,a:i+1,w:i===7?["سم","خز"]:i===6?["ثج","خز"]:["بت","دح"],
    p:i===7?"سمخز":i===6?"ثجخز":"بتدح",o:[0,2]}))});
const word = {surah:1,ayah:2,word:1};
const spelling = (said:string,conf=1): SeamSpellingEvidence => ({word,toFrame:60,
  phonemes:[...said].map((p,i) => ({p,start:20+i,end:20+i,conf}))});
function run(extra:SeamSpellingEvidence[]) {
  const checker = createChecker(index,{passage:{from:{surah:1,ayah:1,word:1},to:{surah:1,ayah:2,word:2}},
    phrases:{version:"test",phrases:[{name:"formula",text:"formula",p:"ثج"}]}});
  const e:PhonemeEvidence = {inventory:"test",frameMs:40,
    phonemes:[..."بتدحبتدح"].map((p,i) => ({p,start:i*4,end:i*4,conf:1,...(i===0?{seamSpellings:extra}:{})}))};
  checker.push(e,[]);
  const events=checker.end();
  return {events,mistakes:checker.mistakes()};
}

it("uses secondary clarity and the folded spell rule without changing primary decisions",() => {
  const baseline=run([]), judged=run([spelling("سم")]);
  expect(judged.mistakes.filter(e => e.errorClass==="added_word")).toMatchObject([
    {location:word,observed:"سم",state:"confirmed",confidence:1,timeMs:2400}]);
  expect(judged.events.filter(e => e.errorClass!=="added_word")).toEqual(baseline.events);
});

it("rejects neighbours, prefixes, back-runs, non-words and weak secondary phones",() => {
  for (const s of ["بت","ب","بتدح","ثج","خد"]) expect(run([spelling(s)]).mistakes.filter(e => e.errorClass==="added_word")).toEqual([]);
  expect(run([spelling("سم",.5)]).mistakes.filter(e => e.errorClass==="added_word")).toEqual([]);
});

it("emits only one addition for duplicate crops and rejects invalid token clocks",() => {
  expect(run([spelling("سم"),spelling("سم")]).mistakes.filter(e => e.errorClass==="added_word")).toHaveLength(1);
  const invalid=spelling("سم"); invalid.phonemes[0].end=60;
  expect(run([invalid]).mistakes.filter(e => e.errorClass==="added_word")).toEqual([]);
});
