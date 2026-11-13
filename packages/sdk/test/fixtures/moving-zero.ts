import type { MovingZeroModel, RootFrame } from "../../src/moving-spectral-zero.js";

export function zeroModel():MovingZeroModel {
  const zeros=(components:number)=>({mean:Array.from({length:3},()=>Array.from({length:components},()=>[.5,.1,.8,.15])),
    variance:Array.from({length:3},()=>Array.from({length:components},()=>Array(4).fill(.1)))});
  return {schema:1,inventory:"test",tokens:"ab",frameMs:40,stay:.8,maxFrames:2048,phoneCounts:[5,5],
    center:Array(8).fill(0),scale:Array(8).fill(1),stationaryVariance:[.0036,.01],
    poles:{mean:Array.from({length:3},()=>Array(8).fill(0)),variance:Array.from({length:3},()=>Array(8).fill(1))},
    control:zeros(2),moving:{...zeros(1),alpha:Array.from({length:3},()=>Array(4).fill(.9)),noise:Array.from({length:3},()=>Array(4).fill(.0025))}};
}
export function rootFrames(n=6):RootFrame[] {
  return Array.from({length:n},(_,t)=>({poles:Array(8).fill(0),zeros:[[.2,.1],[.5+t*.02,.1],[.8-t*.01,.15]],valid:true}));
}
export function reversalModel():MovingZeroModel {
  const m=zeroModel();
  m.control.mean[1].forEach(v=>v.fill(100));m.control.mean[2].forEach(v=>v.fill(0));
  m.moving.mean[1][0].fill(0);m.moving.mean[2][0].fill(100);
  m.moving.alpha.forEach(v=>v.fill(0));
  return m;
}
