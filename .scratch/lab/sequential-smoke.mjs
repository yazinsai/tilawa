import assert from 'node:assert/strict';
import { createRestartEvidence, restartKernel, fitRestartChannel, restartRecordingLogP } from '../../packages/core/src/sequential-restart.ts';
import { createLocalSearch } from '../../packages/core/src/local.ts';
import { restartSpeakerFolds, restartTrainingLabels } from '../../packages/sdk/src/restart-training.ts';
const close = (a,b) => assert.ok(Math.abs(a-b)<1e-10, `${a} != ${b}`);
const phones=['ب','ت','ث','د'];
const channel={phones,rows:phones.map((_,i)=>[...phones.map((_,j)=>i===j?.98:.005),.005]),insertion:.01,inserted:phones.map(()=>.25)};
const ref='بببببتتتتتثثثثثددددد';
const g={reference:ref,lo:0,hi:ref.length,starts:[0,5,10,15],movement:()=>20,entry:()=>0};
const kernel=restartKernel(channel,g);
for(const at of [0,4,g.hi]) {
 const row=new Float64Array(g.hi+1).fill(-Infinity);row[at]=0;
 close(phones.reduce((s,p)=>s+kernel(row,p).reduce((s,x)=>s+Math.exp(x),0),0),1);
}
const mono={phones:['ب'],rows:[[.8,.2]],insertion:.1,inserted:[1]};
const e=createRestartEvidence({channel:mono,threshold:0},{reference:'بببب',lo:0,hi:4,starts:[0,1,2,3],movement:()=>20,entry:()=>0},0,20);
for(let t=1;t<=12;t++) {
 e.step('ب',0);close(e.comparisons().reduce((s,c)=>s+Math.exp(c.logOdds+6),0),t/(t+1));
 assert.ok(e.comparisons().every(c=>!c.admitted));
}
const jump=createRestartEvidence({channel,threshold:0},g,0,20);
for(let i=0;i<3;i++)jump.step('ب',i);
const before=jump.comparison(15);
for(let i=0;i<5;i++)jump.step('د',3);
assert.ok(jump.comparison(15).admitted);
assert.ok(jump.comparison(15).logOdds>before.logOdds);
assert.ok(jump.comparison(15).logOdds>jump.comparison(10).logOdds);
const transient=createRestartEvidence({channel,threshold:0},g,0,20);
for(let i=0;i<5;i++)transient.step('ب',i);
transient.step('د',5);const transientFirst=transient.comparison(15);
assert.ok(transientFirst.increment>0);assert.equal(transientFirst.admitted,false);
for(let i=0;i<4;i++)transient.step('ت',5+i);
assert.ok(transient.comparison(15).logOdds<transientFirst.logOdds);
const duplicated=createRestartEvidence({channel,threshold:0},{...g,starts:[...g.starts,...g.starts]},0,20);
const unique=createRestartEvidence({channel,threshold:0},g,0,20);
for(const p of 'ببتد'){unique.step(p,0);duplicated.step(p,0);}
assert.deepEqual(duplicated.comparisons(),unique.comparisons());
assert.equal(duplicated.comparison(g.hi+1).supported,false);
const reread=jump.comparisons();assert.deepEqual(jump.comparisons(),reread);
const saved=jump.save();
const tail=()=>{for(const p of 'بتد')jump.step(p,1);return jump.comparisons();};
const first=tail();jump.restore(saved);assert.deepEqual(tail(),first);
const records=[{written:'بتثد',heard:'بتثد'},{written:'بتثد',heard:'بتد'}];
const fitted=fitRestartChannel(records,phones,4);
for(const row of [...fitted.rows,fitted.inserted])close(row.reduce((a,b)=>a+b,0),1);
assert.ok(restartRecordingLogP(fitted,records[0])>restartRecordingLogP(fitted,{written:'دددد',heard:'بتثد'}));
assert.throws(()=>restartKernel({...fitted,inserted:[1,1,1,1]},g));
const index={version:'toy',inventoryId:'toy',tokens:new Set(ref),phonemes:ref,words:[0,5,10,15].map((start,i)=>({surah:1,ayah:1,word:i+1,start,end:start+5})),gramSize:4,grams:new Map(),wordAt:offset=>Math.min(3,Math.max(0,Math.floor(offset/5)))};
const localChannel=fitRestartChannel([{written:ref,heard:ref}],[...index.tokens],1);
const seed={offset:0,origin:0,edits:0};
const orig=createLocalSearch(index,seed),audit=createLocalSearch(index,seed,{calibration:{channel:localChannel,threshold:0},audit:true});
for(const p of 'بببتتثددددد'){orig.step(p);audit.step(p);assert.deepEqual(audit.position(),orig.position());}
const comparisons=[];
const guarded=createLocalSearch(index,seed,{calibration:{channel:localChannel,threshold:Number.MAX_VALUE},onComparison:c=>comparisons.push(c)}),original=createLocalSearch(index,seed);
for(const p of 'بببببددددد'){guarded.step(p);original.step(p);}
assert.ok(comparisons.length>0);assert.equal(original.position().origin,15);assert.equal(guarded.position().origin,0);
const snap=guarded.save();const localTail=()=>[...'بددددد'].map(p=>{guarded.step(p);return guarded.position();});
const localFirst=localTail();guarded.restore(snap);assert.deepEqual(localTail(),localFirst);
const label=(id,speaker=id)=>({id,speaker,split:'train',kind:'none',scored:true,ayahs:['1:1'],wav:`${id}.wav`,events:[],non_errors:[]});
assert.deepEqual(restartTrainingLabels([label('ok'),label('leak','held'),{...label('held','held'),split:'eval'}]),[label('ok')]);
const ls=[label('a'),label('b'),label('a2','a')];assert.deepEqual([...restartSpeakerFolds(ls)],[...restartSpeakerFolds([...ls].reverse())]);
console.log('Passed local smoke assertions: kernel normalization, proper time prior, once-only reads, destination evidence, rewind, fit, audit, veto and speaker isolation.');
