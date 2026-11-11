import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { acousticRows, createAcousticRunScorer } from '../../../packages/core/src/acoustic-run.ts';
import { PosteriorFrames } from '../../../packages/core/src/ctc.ts';
import { loadReferenceIndex } from '../../../packages/core/src/reference.ts';
import { createLocalSearch } from '../../../packages/core/src/local.ts';
import { trackerInput } from '../../../packages/core/src/tracker.ts';
import { acousticTrainingLabels } from '../../../packages/sdk/src/acoustic-run-training.ts';
const model = { version: 1, inventory: 'toy', tokens: 'ab', frames: 64,
 acoustic: [[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,0,1],[0,0,0]]], acousticBias:[0,0],
 text: [[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,0,1],[0,0,0]]], textBias:[0,0],
 joint: [[0,0,0,0,1,0],[0,0,0,0,0,1]], jointBias:[0,0], output:[1,1], positionWeight:20, temperature:1 };
const score = createAcousticRunScorer(model, 'toy');
assert(score([[.6,.3,.1]], 'a') < score([[.6,.3,.1]], 'b'));
const rows = [[.6,.35,.05],[.6,.05,.35]];
assert(score(rows, 'ab') < score(rows, 'ba'));
assert(score([...rows].reverse(), 'ba') < score([...rows].reverse(), 'ab'));
const frames = new PosteriorFrames(2);
frames.append({k:3,startFrame:0,nFrames:2,ids:[0,1,2,0,1,2],probs:[.6,.3,.1,.8,.1,.1]});
assert(Math.abs(acousticRows(frames, 0, 2, 1)[0][0] - .7) < 1e-9);
const e = {inventory:'toy',frameMs:40,phonemes:[],posteriors:{k:3,startFrame:0,nFrames:1,ids:[0,1,2],probs:[.6,.3,.1]}};
assert.equal(trackerInput(e).localPosteriors, e.posteriors);
assert.equal(trackerInput(e).posteriors, undefined);
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL('../../../packages/core/assets/ref-v2.json',import.meta.url),'utf8')));
const start = index.words[100].start;
const initial = {offset:start,origin:start,edits:0};
const baseline = createLocalSearch(index,initial);
let calls = 0;
const learned = createLocalSearch(index,initial,{score:()=>{calls++;return 0;}});
for (const p of index.phonemes.slice(start,start+48)) {
 baseline.step(p); learned.step(p); assert.deepEqual(learned.position(),baseline.position());
}
assert.equal(calls,0);
let candidates = [];
const training = createLocalSearch(index,initial,{training:true,inspect:c=>{candidates=c;}});
for(const p of index.phonemes.slice(start,start+24)) training.step(p);
assert.equal(new Set(candidates.map(c=>c.observations)).size,1);
assert.equal(candidates[0].observations,16);
const clean = {id:'a',speaker:'one',split:'train',kind:'none',scored:true,events:[],non_errors:[],ayahs:[],wav:'a.wav'};
assert.equal(acousticTrainingLabels([clean,{...clean,id:'b',split:'eval'}]).length,0);
console.log('Local smoke checks passed: sub-blank evidence, temporal order, row retention, unchanged progress, common interval, speaker exclusion.');
