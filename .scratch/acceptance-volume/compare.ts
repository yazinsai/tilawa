import { readFileSync, writeFileSync } from 'node:fs';
import { createAcceptanceVolume, backgroundSamples, logAcceptanceVolume } from '../../packages/core/src/acceptance-volume.js';

const fit = JSON.parse(readFileSync('/scratch/tracker-reference-acceptance-volume/background.json','utf8'));
const target = JSON.parse(readFileSync(process.argv[2],'utf8'));
const at = Number(process.argv[3]);
if (!Number.isFinite(at)) throw new Error('Expected target snapshot file and time in seconds');
const frame = target.snapshots.find((s:any)=>Math.abs(s.time-at)<0.001);
if (!frame) throw new Error('No target restart snapshot');
const [near,restart] = frame.candidates;
const rawNear = Math.min(...near.states.filter((s:any)=>s.at>near.states[0].at).map((s:any)=>s.cost));
const rawRestart = Math.min(...restart.states.filter((s:any)=>s.at>restart.states[0].at).map((s:any)=>s.cost));
const z = createAcceptanceVolume(fit.background);
const lengths = Array.from({length:16},(_,i)=>i+1).map(n=>{
  const nearZ=z(near.pronunciation,n),restartZ=z(restart.pronunciation,n);
  const samples=backgroundSamples(fit.background,n);
  const blocks=Array.from({length:8},(_,i)=>{
    const b=samples.slice(i*512,(i+1)*512);
    return logAcceptanceVolume(restart.pronunciation,b)-logAcceptanceVolume(near.pronunciation,b);
  });
  return {n,nearZ,restartZ,difference:restartZ-nearZ,blockDifferences:blocks};
});
const result={rawNear,rawRestart,rawAdvantage:(rawNear-rawRestart)/10,
  nearPronunciation:near.pronunciation,restartPronunciation:restart.pronunciation,lengths};
writeFileSync(`${process.env.LAB_OUT}/acceptance-comparison.json`,JSON.stringify(result,null,1));
console.log(JSON.stringify(result));
