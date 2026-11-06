import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { loadReferenceIndex } from "@tilawa/core";
import { extraWords, wordEarCandidates } from "../src/word-ear.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json",import.meta.url),"utf8")));
const phrases = JSON.parse(readFileSync(new URL("../../core/assets/phrases-v1.json",import.meta.url),"utf8"));
const word = index.words.findIndex(w => w.surah === 1 && w.ayah === 2 && w.word === 1);
const span = { word, fromFrame: 10, toFrame: 40, seamFromFrame: 0, seamToFrame: 32 };

it("diffs whole words in order without using their confidence", () => {
  expect(extraWords(["الحمد","لله"],[{text:"كريم",logp:-Infinity},{text:"ٱلْحَمْدُ",logp:0},{text:"لله",logp:0}])).toEqual(["كريم"]);
  expect(extraWords(["الحمد","لله"],[{text:"الحمد",logp:0},{text:"الحمد",logp:0},{text:"لله",logp:0}])).toEqual(["الحمد"]);
});

it("proposes whole non-neighbour words and rejects repeats, prefixes and formula words", () => {
  const heard = (text: string) => [{text,logp:-Infinity}];
  const candidates = wordEarCandidates(index,span,heard("كريم"),phrases);
  expect(candidates.length).toBeGreaterThan(0);
  expect(candidates[0]).toMatchObject({word,fromFrame:10,toFrame:40});
  expect(wordEarCandidates(index,span,heard("الحمد"),phrases)).toEqual([]);
  expect(wordEarCandidates(index,span,heard("الحم"),phrases)).toEqual([]);
  expect(wordEarCandidates(index,span,heard("الله"),phrases)).toEqual([]);
  expect(wordEarCandidates(index,span,heard("مخترعxyz"),phrases)).toEqual([]);
});
