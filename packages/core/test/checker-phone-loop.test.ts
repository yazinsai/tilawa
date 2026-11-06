import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { createChecker } from '../src/checker.js';
import { loadReferenceIndex } from '../src/reference.js';
import { syntheticFixture, withPosteriors } from '../src/fixtures/index.js';
const index=loadReferenceIndex(JSON.parse(readFileSync(new URL('../assets/ref-v2.json',import.meta.url),'utf8')));
const from={surah:44,ayah:9,word:1},to={surah:44,ayah:11,word:5},seam={surah:44,ayah:10,word:1};
function check(marginal:number,observed='ظغ',kind:'phone-loop'|'exact'='phone-loop'){
 const start=index.words.find(w=>w.surah===from.surah&&w.ayah===from.ayah&&w.word===from.word)!.start;
 const right=index.words.find(w=>w.surah===seam.surah&&w.ayah===seam.ayah&&w.word===seam.word)!;
 const at=right.start-start;
 const fixture=syntheticFixture(index,{id:'blank-seam',from,to,edits:[{kind:'gap',at,frames:20}],chunkSize:10000});
 const evidence=withPosteriors(index,fixture.evidence[0]);
 const end=evidence.phonemes[at].start;
 evidence.phonemes[0].seamInsertions=[{kind:kind==='phone-loop'?kind:undefined,word:seam,observed,fromFrame:end-10,toFrame:end-2,marginal}];
 const checker=createChecker(index,{passage:{from,to},policy:{minConfidence:.99,localWindow:{nats:Infinity}}});
 checker.push(JSON.parse(JSON.stringify(evidence)),[]);checker.end();
 return checker.mistakes().filter(e=>e.errorClass==='added_word'&&JSON.stringify(e.location)===JSON.stringify(seam));
}
describe('teacher phone-loop certificates',()=>{
 it('confirms extra speech absent from student tokens with a marginal above one half',()=>{
  expect(check(.5001)).toMatchObject([{observed:'ظغ',state:'confirmed',confidence:.5001}]);
 });
 it('requires the loop certificate, two consonants and a strict majority',()=>{
  expect(check(.5)).toHaveLength(0);
  expect(check(.49)).toHaveLength(0);
  expect(check(.9,'ظ')).toHaveLength(0);
  expect(check(.9,'ظغ','exact')).toHaveLength(0);
 });
 it('rejects neighbour prefixes and held breath/vowel tails',()=>{
  const right=index.words.find(w=>w.surah===seam.surah&&w.ayah===seam.ayah&&w.word===seam.word)!;
  expect(check(.9,index.phonemes.slice(right.start,right.end))).toHaveLength(0);
  expect(check(.9,'ۦں')).toHaveLength(0);
 });
});
