import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { createChecker } from '../src/checker.js';
import { loadReferenceIndex } from '../src/reference.js';
import { syntheticFixture, withPosteriors } from '../src/fixtures/index.js';
import type { SeamInsertionWindow } from '../src/checker.js';
const index=loadReferenceIndex(JSON.parse(readFileSync(new URL('../assets/ref-v2.json',import.meta.url),'utf8')));
const from={surah:44,ayah:9,word:1},to={surah:44,ayah:11,word:5},seam={surah:44,ayah:10,word:1};
function seamFixture(gapFrames=20){
 const start=index.words.find(w=>w.surah===from.surah&&w.ayah===from.ayah&&w.word===from.word)!.start;
 const right=index.words.find(w=>w.surah===seam.surah&&w.ayah===seam.ayah&&w.word===seam.word)!;
 const at=right.start-start;
 const fixture=syntheticFixture(index,{id:'blank-seam',from,to,edits:[{kind:'gap',at,frames:gapFrames}],chunkSize:10000});
 const evidence=withPosteriors(index,fixture.evidence[0]);
 return {evidence,at};
}
function check(marginal:number,observed='ظغ',kind:'phone-loop'|'exact'='phone-loop'){
 const {evidence,at}=seamFixture();
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
 it('starts the cold crop after the previous phone held run, inside the labelled pause',()=>{
  const {evidence,at}=seamFixture();
  const left=evidence.phonemes[at-1],pauseStart=left.end+1,heldEnd=pauseStart+3;
  const posterior=evidence.posteriors!,id=[...index.tokens].indexOf(left.p)+1;
  const ids=Array.from(posterior.ids),probs=Array.from(posterior.probs);
  const filler=[...index.tokens].map((_,i)=>i+1).filter(i=>i!==id);
  for(let f=pauseStart;f<heldEnd;f++){
   const row=(f-posterior.startFrame)*posterior.k;
   for(let j=0;j<posterior.k;j++){
    ids[row+j]=j===0?id:j===1?0:filler[j-2];
    probs[row+j]=j===0?.9:j===1?.05:0;
   }
  }
  evidence.posteriors={...posterior,ids,probs};
  const seams:SeamInsertionWindow[]=[];
  const checker=createChecker(index,{passage:{from,to},seamPhoneLoop:s=>seams.push(s)});
  checker.push(evidence,[]);checker.end();
  const wi=index.words.findIndex(w=>w.surah===seam.surah&&w.ayah===seam.ayah&&w.word===seam.word);
  expect(seams.filter(s=>s.word===wi).length).toBeGreaterThan(0);
  expect(seams.filter(s=>s.word===wi).every(s=>s.pauseFromFrame===heldEnd)).toBe(true);
 });
 it('does not invent a pause onset on a connected seam',()=>{
  const {evidence}=seamFixture(0),seams:SeamInsertionWindow[]=[];
  const checker=createChecker(index,{passage:{from,to},seamPhoneLoop:s=>seams.push(s)});
  checker.push(evidence,[]);checker.end();
  const wi=index.words.findIndex(w=>w.surah===seam.surah&&w.ayah===seam.ayah&&w.word===seam.word);
  expect(seams.filter(s=>s.word===wi).length).toBeGreaterThan(0);
  expect(seams.filter(s=>s.word===wi).every(s=>s.pauseFromFrame===undefined)).toBe(true);
 });
});
