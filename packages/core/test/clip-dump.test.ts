import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { loadReferenceIndex } from '../src/index.js';
import type { CheckerEvent, ReferenceAsset } from '../src/index.js';
import type { AcceptanceClip, ClipScore } from '../src/bench/acceptance.js';
import { CLIP_DUMP_VERSION, dumpArm } from '../src/bench/clip-dump.js';
const asset: ReferenceAsset = JSON.parse(readFileSync(new URL('../assets/ref-v2.json', import.meta.url), 'utf8'));
const index = loadReferenceIndex(asset);
const clip: AcceptanceClip = { id:'example', ayahs:['94:5'], events:[{errorClass:'omitted_word',state:'confirmed',location:{surah:94,ayah:5,word:2}}],nonErrors:[],evidence:{inventory:index.inventoryId,frameMs:40,nFrames:150,phonemes:[]} };
const event = (state: CheckerEvent['state'], ayah=5): CheckerEvent => ({type:'checker',v:1,id:`94:${ayah}:2#1`,timeMs:9000,state,errorClass:'wrong_word',location:{surah:94,ayah,word:2},expected:'مَعَ',observed:'هءَ',confidence:.8});
const score = (final: CheckerEvent[]): ClipScore => ({final,confirmed:final.filter(e=>e.state==='confirmed'),hits:0,misses:clip.events,falsePositives:final.filter(e=>e.state==='confirmed'),repeatFalsePositives:0,pending:0,pendingHits:0,gate:{onsetMs:1200,firstCorrectLockMs:5000,correctWithin5s:true,wrongLockFirst:false,outsideLock:false,jumps:0,falseJumps:0}});
describe('review dump evidence',()=>{
 it('retains original final checker states and emission times without turning labels into reports',()=>{
  const s=score([event('confirmed'),{...event('uncertain'),id:'uncertain'},{...event('resolved'),id:'resolved'},{...event('suspected'),id:'suspected'}]);
  const before=JSON.stringify(s),d=dumpArm(index,clip,[],'help/example.wav',{score:s});
  expect(CLIP_DUMP_VERSION).toBe(2);
  expect(d.engineEvents.map(e=>e.state)).toEqual(['confirmed','uncertain','resolved','suspected']);
  expect(d.engineEvents.map(e=>e.checkerEvent)).toEqual(s.final);
  expect(d.misses[0].checkerEvent).toBeUndefined();
  expect(d.misses[0].state).toBe('confirmed');
  expect(JSON.stringify(s)).toBe(before);
 });
 it('uses unrestricted final reports for context arms and keeps source audio separate from session time',()=>{
  const report=event('confirmed',4),raw=score([report]);
  const d=dumpArm(index,clip,[],'help/example.wav',{score:score([]),raw,segments:[{source:'before',ayahs:['94:4'],startMs:0,onsetMs:0,endMs:4000,planted:false},{source:'example',ayahs:['94:5'],startMs:4600,onsetMs:5800,endMs:10600,planted:false}],segmentAudio:['leadin/before.wav','help/example.wav'],leadAyahs:['94:4']});
  expect(d.engineEvents[0].audio).toBe('leadin/before.wav');
  expect(d.engineEvents[0].checkerEvent?.timeMs).toBe(9000);
  expect(d.leadInWarnings?.[0].checkerEvent).toEqual(report);
 });
 it('keeps an omission miss and different-class report as separate original scorer outcomes',()=>{
  const s=score([event('confirmed')]),d=dumpArm(index,clip,[{at:'94:5:2',text:'مَعَ',ref:'مَعَ',said:'هءَ',fromS:2.6,toS:3.28}],'help/example.wav',{score:s});
  expect(d.misses[0].errorClass).toBe('omitted_word');expect(d.falsePositives[0].errorClass).toBe('wrong_word');
  expect(d.misses[0].seekS).toBe(1.6);expect(d.falsePositives[0].seekS).toBe(1.6);
  expect(d.hits).toEqual([]);
 });
});
