import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { createChecker, loadReferenceIndex } from "../src/index.js";
import type { PhonemeEvidence, SeamInsertionEvidence } from "../src/index.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json",import.meta.url),"utf8")));
const wi = index.words.findIndex(w => w.surah === 1 && w.ayah === 2 && w.word === 1);
const alien = index.words.findIndex(w => index.wordText![index.words.indexOf(w)].replace(/[\p{M}ـ]/gu,"") === "كريم");
const observed = index.phonemes.slice(index.words[alien].start,index.words[alien].end);
const base: SeamInsertionEvidence = {word:{surah:1,ayah:2,word:1},observed,fromFrame:10,toFrame:40,marginal:.9,source:"word-ear"};
const run = (certificates: SeamInsertionEvidence[]) => {
  const checker = createChecker(index);
  const evidence: PhonemeEvidence = {inventory:index.inventoryId,frameMs:40,
    phonemes:[{p:"ب",start:0,end:0,conf:.1,seamInsertions:certificates}]};
  checker.push(JSON.parse(JSON.stringify(evidence)),[]);
  checker.end();
  return checker.mistakes();
};

it("uses the teacher's certificate without student spellings or confidence, and deduplicates replays", () => {
  const result = run([base,base]);
  expect(result).toHaveLength(1);
  expect(result[0]).toMatchObject({errorClass:"added_word",state:"confirmed",observed,confidence:.9});
});

it("requires a strict majority, word-ear provenance and a whole non-neighbour word", () => {
  expect(run([{...base,marginal:.5}])).toEqual([]);
  expect(run([{...base,source:undefined}])).toEqual([]);
  expect(run([{...base,observed:index.phonemes.slice(index.words[wi].start,index.words[wi].end)}])).toEqual([]);
  expect(run([{...base,marginal:NaN}])).toEqual([]);
});
