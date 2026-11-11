import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { loadNodeModel } from '../../packages/sdk/src/node.js';
import { createListener } from '../../packages/sdk/src/listener.js';
import { decodeWav } from '../../packages/sdk/src/wav.js';
import { fitPhoneBackground } from '../../packages/core/src/acceptance-volume.js';

const root = `${process.env.LAB_DATA}/help-train`;
const labels = readFileSync(`${root}/labels.jsonl`,'utf8').trim().split('\n').map(x=>JSON.parse(x));
if (labels.some(l=>l.split!=='train')) throw new Error('Training export contains other partitions');
const clean = labels.filter(l=>l.kind==='none' && !(l.events??[]).length && !(l.non_errors??[]).length).sort((a,b)=>a.id.localeCompare(b.id));
const model = await loadNodeModel(`${process.env.LAB_DATA}/models/spliced-clean-int8`);
const cache = '/scratch/tracker-reference-acceptance-volume/clean-decodes';
mkdirSync(cache,{recursive:true});
const recordings: any[] = [], hashes: any[] = [];
for (const l of clean) {
  const data = readFileSync(`${root}/${l.wav}`);
  const hash = createHash('sha256').update(data).digest('hex');
  const path = `${cache}/${hash}.json`;
  let phones: string;
  if (existsSync(path)) phones = JSON.parse(readFileSync(path,'utf8')).phones;
  else {
    const wav = decodeWav(data), listener = createListener(model);
    const evidence: any[] = [];
    const step = Math.round(wav.sampleRate*0.32);
    for(let i=0;i<wav.samples.length;i+=step) {const e=await listener.push(wav.samples.slice(i,i+step),wav.sampleRate); if(e)evidence.push(e);}
    const end=await listener.end(); if(end)evidence.push(end);
    phones = evidence.flatMap(e=>e.phonemes).map(p=>p.p).join('');
    writeFileSync(path,JSON.stringify({phones}));
  }
  recordings.push({...l,phones}); hashes.push({speaker:l.speaker,hash});
  console.log('decoded',recordings.length,'/',clean.length,phones.length);
}
await model.backend.release();
const background = fitPhoneBackground(model.manifest.tokens,recordings);
const speakers = [...new Set(recordings.map(r=>r.speaker))].sort();
// Grouped training-only checks; no practice data enters fitting or model selection.
const folds = speakers.slice(0,5).map((_,fold)=>{
  const train=recordings.filter(r=>speakers.indexOf(r.speaker)%5!==fold);
  const held=recordings.filter(r=>speakers.indexOf(r.speaker)%5===fold);
  const m=fitPhoneBackground(model.manifest.tokens,train);
  let cost=0,n=0;
  for(const r of held)for(let i=0;i<r.phones.length;i++) {
    const at=m.tokens.indexOf(r.phones[i]); const prev=i?m.tokens.indexOf(r.phones[i-1]):-1;
    cost-=Math.log(prev<0?m.initial[at]:m.next[prev][at]); n++;
  }
  return {fold,trainSpeakers:new Set(train.map(r=>r.speaker)).size,heldSpeakers:new Set(held.map(r=>r.speaker)).size,n,nll:cost/n};
});
const out={background,provenance:{recordings:clean.length,speakers:speakers.length,tokens:recordings.reduce((s,r)=>s+r.phones.length,0),
  audioHashes:hashes,listener:model.manifest.id,listenerHash:model.manifest.sha256,folds}};
writeFileSync(`${process.env.LAB_OUT}/background.json`,JSON.stringify(out,null,1));
writeFileSync('/scratch/tracker-reference-acceptance-volume/background.json',JSON.stringify(out));
console.log('FIT',JSON.stringify({...out.provenance,audioHashes:undefined}));
