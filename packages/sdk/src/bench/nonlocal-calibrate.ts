/** Text-free self-supervised selection on none-labelled, training-speaker recordings only.
 * Run on CPU remotely: tsx src/bench/nonlocal-calibrate.ts --data .../help-train --out .../calibration.json
 * Independent injected noise supplies paired observations; Quran text and recognition outputs are never read. */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { decodeWav } from "../wav.js";
import { Resampler } from "../audio.js";
import { NonlocalAcousticDenoiser } from "../nonlocal-acoustic.js";
import type { NonlocalGeometry } from "../nonlocal-acoustic.js";

const { values } = parseArgs({ options: { data:{type:"string"}, out:{type:"string"} } });
if (!values.data || !values.out) throw new Error("--data and --out are required");
type Label = { speaker:string; split:string; kind:string; wav:string; id:string };
const labels = readFileSync(resolve(values.data,"labels.jsonl"),"utf8").trim().split("\n").map(s => JSON.parse(s) as Label)
  .filter(l => (l.split === "train" || l.split === "training") && l.kind === "none");
if (!labels.length) throw new Error("No explicitly training-split none-labelled recordings; do not fall back to other splits");
const hash = (s:string) => [...s].reduce((a,c) => Math.imul(a^c.charCodeAt(0),16777619)>>>0,2166136261);
const speakers = [...new Set(labels.map(l => l.speaker))].sort();
if (speakers.length < 3) throw new Error("At least three training speakers are required");
const folds = new Map(speakers.map((s,i) => [s,i%3]));
const grid: NonlocalGeometry[] = [];
for (const radius of [5,8]) for (const contextLimit of [0.25,0.5,1]) for (const consistencyLimit of [0.25,0.5,1])
  grid.push({radius,contextLimit,consistencyLimit,donors:3});
const results: {speaker:string; fold:number; scores:number[]; repaired:number[]}[] = [];
for (const label of labels) {
  const wav = decodeWav(readFileSync(resolve(values.data,label.wav)));
  let audio = wav.samples;
  if (wav.sampleRate !== 16000) {
    const rs = new Resampler(wav.sampleRate,16000), a = rs.push(audio), b = rs.flush();
    audio = new Float32Array(a.length+b.length); audio.set(a); audio.set(b,a.length);
  }
  if (audio.length < 16000*2) continue;
  // Fixed per-recording seed and independent samples; no transcription or planted labels enter the loss.
  let seed = hash(label.id);
  const uniform = () => { seed ^= seed<<13; seed ^= seed>>>17; seed ^= seed<<5; return (seed>>>0)/4294967296; };
  const rms = Math.sqrt(audio.reduce((a,b) => a+b*b,0)/audio.length);
  const noisy = Float32Array.from(audio,x => x+rms*0.1*Math.sqrt(-2*Math.log(Math.max(1e-12,uniform())))*Math.cos(2*Math.PI*uniform()));
  const centers = [0.4,0.6,0.8].map(f => Math.floor(audio.length*f/160)*160);
  const scores:number[] = [], repaired:number[] = [];
  for (const geometry of grid) {
    const denoiser = new NonlocalAcousticDenoiser(noisy,geometry);
    let loss = 0, count = 0;
    for (const center of centers) {
      const from = center, to = Math.min(audio.length,from+2560);
      const result = denoiser.repair(from,to);
      count += result.patches.length;
      for (let i = from; i < to; i++) loss += ((result.samples[i]-audio[i])**2-(noisy[i]-audio[i])**2)/(rms*rms+1e-20);
    }
    scores.push(loss/(centers.length*2560)); repaired.push(count);
  }
  results.push({speaker:label.speaker,fold:folds.get(label.speaker)!,scores,repaired});
  console.log(JSON.stringify({done:results.length,total:labels.length,speaker:label.speaker}));
}
// Each speaker contributes equally, regardless of the number of submitted recordings.
const mean = (i:number, rows:typeof results) => {
  const ss = [...new Set(rows.map(r => r.speaker))];
  return ss.reduce((sum,s) => {
    const rr = rows.filter(r => r.speaker === s);
    return sum+rr.reduce((a,r) => a+r.scores[i],0)/rr.length;
  },0)/ss.length;
};
const best = (rows:typeof results) => grid.map((_,i) => i).sort((a,b) => mean(a,rows)-mean(b,rows)||a-b)[0];
const validation = [0,1,2].map(fold => {
  const train = results.filter(r => r.fold !== fold), held = results.filter(r => r.fold === fold);
  if (!train.length || !held.length) throw new Error("Missing speaker-grouped partition");
  const picked = best(train);
  return { fold,selected:grid[picked],heldLoss:mean(picked,held),heldSpeakers:[...new Set(held.map(r => r.speaker))] };
});
const selected = best(results);
writeFileSync(values.out,JSON.stringify({protocol:"nonlocal-independent-noise-v1",noiseRmsRatio:0.1,
  trainingSpeakers:speakers,recordings:results.length,grid,validation,selected:grid[selected],
  loss:mean(selected,results),eligiblePatches:results.reduce((a,r) => a+r.repaired[selected],0)},null,2));
