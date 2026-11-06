/** Clean training-speaker envelope for the frozen separator head. Accepted
 * repeats are negatives too. Tracker events place every span; labels select
 * recordings only. This tool does not score a practice or release set. */
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { loadReferenceIndex, createTracker, trackerInput, collapseHeld } from '../packages/core/src/index.ts';
import { wordCountGraph, wordCountLogSum } from '../packages/core/src/seam-word-count.ts';
import { waqfMarked } from '../packages/core/src/mushaf-marks.ts';
import { insertionCells } from '../packages/sdk/src/seam-insertion.ts';
import { createListener } from '../packages/sdk/src/listener.ts';
import { loadNodeModel, readWav } from '../packages/sdk/src/node.ts';

const json = (p: string) => JSON.parse(readFileSync(p,'utf8'));
const data = process.env.LAB_DATA!, out = process.env.LAB_OUT!;
const loaded = await loadNodeModel(process.env.HEAD!, {threads:2});
const head = loaded.manifest.wordCount ? {...loaded,manifest:{...loaded.manifest,tokens:loaded.manifest.wordCount.tokens,wordCount:undefined},
  backend:{...loaded.backend,name:loaded.backend.name,release:()=>loaded.backend.release(),async run(feeds:any){
    const output=await loaded.backend.run(feeds);return {...output,log_probs:output.word_count_log_probs};
  }}} : loaded;
const primary = await loadNodeModel(join(data,'models/spliced-clean-int8'), {threads:2});
const index = loadReferenceIndex(json('/app/packages/core/assets/ref-v2.json'));
const phrases = json('/app/packages/core/assets/phrases-v1.json');
const key = (w:any) => `${w.surah}:${w.ayah}:${w.word}`;
const wi = new Map(index.words.map((w,i) => [key(w),i]));
const ids = new Map([...head.manifest.tokens].map((p,i) => [p,i+1]));
const sep = ids.get('|')!;
const phones = [...head.manifest.tokens].flatMap((p,i) => '|ڇؙۜ'.includes(p) ? [] : [i+1]);
const consonants = new Set(phones.filter(i => 'ءبتثجحخدذرزسشصضطظعغفقكلمنهوي'.includes(head.manifest.tokens[i-1])));
const variants = (at:number) => [...new Set([...(index.wordForms?.[at] ?? []),
  index.phonemes.slice(index.words[at].start,index.words[at].end),index.isolatedWordForms?.[at] ?? '']
  .filter(Boolean).map(s => collapseHeld([...s]).join('')))].map(s => [...s].map(p => ids.get(p)!));
const labels = readFileSync(join(data,'help-train/labels.jsonl'),'utf8').trim().split('\n').map(s => JSON.parse(s));
const kinds=(process.env.CALIBRATION_KINDS ?? 'none,repeat').split(',');
const clean = labels.filter(e => e.split==='train' && e.scored===true && !e.excluded &&
  !e.events?.length && !e.unsupported?.length && kinds.includes(e.kind));
console.log(JSON.stringify({labels:labels.length,clean:clean.length,kinds:[...new Set(clean.map(e=>e.kind))]}));
const result:any[]=[];
let recordings=0;
for (let at=0;at<clean.length;at++) {
  if (at % Number(process.env.LAB_SHARDS ?? 1)!==Number(process.env.LAB_SHARD ?? 0)) continue;
  const entry=clean[at], wav=await readWav(join(data,'help-train',entry.wav));
  const listener=createListener(primary,{posteriors:true}); const chunks:any[]=[];
  for (let s=0;s<wav.samples.length;s+=16000*20) {
    const e=await listener.push(wav.samples.subarray(s,s+16000*20),wav.sampleRate); if(e) chunks.push(e);
  }
  const end=await listener.end(); if(end) chunks.push(end);
  const tracker=createTracker(index,{mode:'free',phrases});
  const steps=new Map<number,{frame:number,run:number}>(); const seams:any[]=[];
  let last=-1,run=0;
  for (const cell of insertionCells(chunks)) for (const event of tracker.push(trackerInput(cell))) {
    if(event.state!=='locked'||!event.expected) {last=-1;run++;continue;}
    const w=wi.get(key(event.expected))!;
    if(w===last) continue;
    if(last>=0 && w!==last+1) run++;
    last=w; const frame=Math.round(event.timeMs/primary.manifest.frameMs); steps.set(w,{frame,run});
    const prev=w-2,w1=w-1;
    if(prev<0 || (index.words[w1].word!==1&&!waqfMarked(prev))) continue;
    const start=steps.get(prev),middle=steps.get(w1);
    if(!start||!middle||start.run!==run||middle.run!==run||frame<=start.frame)continue;
    seams.push({word:w1,from:start.frame,to:frame,entry:middle.frame});
  }
  recordings++;
  if(!seams.length) {console.log(JSON.stringify({id:entry.id,seams:0}));continue;}
  const rows:Float32Array[]=[];const backend=head.backend;
  const wrapped={...head,backend:{...backend,name:backend.name,release:()=>backend.release(),async run(feeds:any){
    const output=await backend.run(feeds);rows.push((output.log_probs.data as Float32Array).slice());return output;
  }}};
  const second=createListener(wrapped);
  for(let s=0;s<wav.samples.length;s+=16000*20)await second.push(wav.samples.subarray(s,s+16000*20),wav.sampleRate);
  await second.end();
  const vocab=head.manifest.tokens.length+1,flat=new Float32Array(rows.reduce((n,r)=>n+r.length,0));
  let offset=0;for(const row of rows){flat.set(row,offset);offset+=row.length;}
  const frames={startFrame:0,endFrame:second.decoded,row:(f:number)=>flat.subarray(f*vocab,(f+1)*vocab)};
  for(const seam of seams){
    const words=[variants(seam.word-1),variants(seam.word),variants(seam.word+1)];
    const no=wordCountLogSum(wordCountGraph(words,sep,phones,consonants,false),frames,seam.from,seam.to);
    const yes=wordCountLogSum(wordCountGraph(words,sep,phones,consonants,true),frames,seam.from,seam.to);
    result.push({id:entry.id,device:entry.device,word:key(index.words[seam.word]),fromS:seam.from*.04,
      toS:seam.to*.04,entryS:seam.entry*.04,no,yes,margin:no!==null&&yes!==null?yes-no:null});
  }
  console.log(JSON.stringify({id:entry.id,seams:seams.length,best:Math.max(...result.filter(r=>r.id===entry.id).map(r=>r.margin??-Infinity))}));
  writeFileSync(join(out,'clean-margins.json'),JSON.stringify({recordings,eligible:clean.length,rows:result}));
}
writeFileSync(join(out,'clean-margins.json'),JSON.stringify({recordings,eligible:clean.length,rows:result}));
await primary.backend.release();await head.backend.release();
