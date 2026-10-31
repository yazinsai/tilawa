/** Read-only control replay. Retain all states: an alarm later resolved was still shown. */
import {readFileSync,writeFileSync} from 'node:fs';
import {parseArgs} from 'node:util';
import {createEngine} from '../session.js';
import {loadNodeModel} from '../node.js';
import {decodeWav} from '../wav.js';
import {measureLatency} from '../latency.js';
import {loadReferenceIndex} from '@tilawa/core';
const {values}=parseArgs({options:{model:{type:'string'},out:{type:'string'}}});
if(!values.model||!values.out)throw Error('model and out required');
const asset=(p:string)=>JSON.parse(readFileSync(new URL(`../../../core/assets/${p}`,import.meta.url),'utf8'));
const engine=createEngine({reference:loadReferenceIndex(asset('ref-v2.json')),phrases:asset('phrases-v1.json'),model:await loadNodeModel(values.model),recordEvidence:true});
const controls=JSON.parse(readFileSync('/lab/data/sparse-real-controls/controls.json','utf8'));
const result=[];
for(const c of controls){
 const wav=decodeWav(readFileSync(c.path));
 for(const mode of ['free','declared']){
  const session=mode==='free'?engine.startFreeRecitation():engine.startMemorization({surah:c.surah,fromAyah:c.fromAyah,toAyah:c.toAyah});
  const r=await measureLatency(session,wav.samples,wav.sampleRate,20);
  const positions=r.events.filter(e=>e.type==='position');
  const events=r.events.filter(e=>e.type==='checker');
  const target=events.filter(e=>e.location.surah===c.location.surah&&e.location.ayah===c.location.ayah&&e.location.word===c.location.word);
  const evidence=session.evidence();
  result.push({control:c,mode,positions,events,target,everConfirmed:target.some(e=>e.state==='confirmed'),resolved:target.some(e=>e.state==='resolved'),phonemes:evidence.flatMap(e=>e.phonemes)});
 }
}
writeFileSync(values.out,JSON.stringify({model:values.model,scope:'Two independently transcribed self-corrections; all states retained; unrelated warnings unresolved',result},null,1));
