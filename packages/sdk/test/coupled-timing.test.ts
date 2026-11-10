import { describe, expect, it } from "vitest";
import { PosteriorFrames } from "@tilawa/core";
import { coupledTiming } from "../src/coupled-timing.js";

const rows = [
  [.2,.6,.1,.1], [.6,.2,.1,.1], [.1,.1,.5,.3],
  [.2,.1,.2,.5], [.6,.2,.1,.1], [.2,.6,.1,.1],
];
function frames(ps: number[][]) {
  const f = new PosteriorFrames(ps[0].length-1);
  f.append({startFrame:0,nFrames:ps.length,k:ps[0].length,
    ids:ps.flatMap(p=>p.map((_,i)=>i)),probs:ps.flat()});
  return f;
}

/** Enumerate complete raw CTC paths, not the production state recurrence. */
function paths(ps: number[][], target: number[]) {
  const out: { bounds: [number,number][]; score: number }[] = [], raw: number[] = [];
  const visit = () => {
    if (raw.length < ps.length) {
      for (let x = 0; x < ps[0].length; x++) { raw.push(x); visit(); raw.pop(); }
      return;
    }
    const labels: number[] = [], bounds: [number,number][] = [];
    for (let i = 0; i < raw.length; i++) {
      if (!raw[i]) continue;
      if (i && raw[i] === raw[i-1]) bounds.at(-1)![1] = i+1;
      else { labels.push(raw[i]); bounds.push([i,i+1]); }
    }
    if (JSON.stringify(labels) !== JSON.stringify(target)) return;
    out.push({bounds,score:raw.reduce((s,x,i)=>s+Math.log(ps[i][x]),0)});
  };
  visit(); return out;
}

function exhaustive(ps: number[][], w: number[], r: number[], maps: [number,number][][]) {
  const wp = paths(ps,w), rp = paths(ps,r);
  let mass = 0, weighted = 0;
  for (const map of maps) {
    const pairs = wp.flatMap(a=>rp.filter(b=>map.every(([i,j]) =>
      a.bounds[i][0] === b.bounds[j][0] && a.bounds[i][1] === b.bounds[j][1])).map(b=>({a,b})));
    for (const {a,b} of pairs) {
      const p = Math.exp((a.score+b.score)/2)/pairs.length/maps.length;
      mass += p; weighted += p*(a.score-b.score+Math.log(rp.length/wp.length));
    }
  }
  return weighted/mass;
}

describe("paired complete-pronunciation timing", () => {
  it("agrees with exhaustive complete CTC paths and jointly shared intervals", () => {
    const w = [1,2,1], r = [1,3,1];
    const result = coupledTiming(frames(rows),w,r,0,rows.length)!;
    expect(result.coupled).toBeCloseTo(exhaustive(rows,w,r,[[[0,0],[2,2]]]),10);
    expect(result.correspondences).toBe(1);
    const wp = paths(rows,w), rp = paths(rows,r);
    const sum = (xs: typeof wp) => xs.reduce((s,p)=>s+Math.exp(p.score),0);
    expect(result.normalizedIndependent).toBeCloseTo(Math.log(sum(wp)/wp.length)-Math.log(sum(rp)/rp.length),10);
    const mean = (xs: typeof wp) => xs.reduce((s,p)=>s+Math.exp(p.score/2)*p.score,0) /
      xs.reduce((s,p)=>s+Math.exp(p.score/2),0);
    expect(result.uncoupled).toBeCloseTo(mean(wp)-mean(rp)+Math.log(rp.length/wp.length),10);
  });

  it("marginalizes alternative shared-phone correspondences with normalized priors", () => {
    const w = [1,1,2], r = [1,3,2];
    const result = coupledTiming(frames(rows),w,r,0,rows.length)!;
    expect(result.correspondences).toBe(2);
    expect(result.coupled).toBeCloseTo(exhaustive(rows,w,r,[[[0,0],[2,2]],[[1,0],[2,2]]]),10);
  });

  it("is antisymmetric and leaves identical complete hypotheses tied", () => {
    const f = frames(rows), w = [1,2,1], r = [1,3,1];
    const a = coupledTiming(f,w,r,0,rows.length)!, b = coupledTiming(f,r,w,0,rows.length)!;
    expect(a.coupled).toBeCloseTo(-b.coupled,10);
    expect(a.normalizedIndependent).toBeCloseTo(-b.normalizedIndependent,10);
    expect(coupledTiming(f,w,w,0,rows.length)!.coupled).toBe(0);
  });

  it("requires every articulation, including repeated labels separated by blank", () => {
    const f = frames([[.1,.7,.1,.1],[.1,.7,.1,.1]]);
    expect(coupledTiming(f,[1,1],[1,2],0,2)).toBeNull();
    expect(coupledTiming(f,[1,1],[1,2],-1,2)).toBeNull();
    expect(coupledTiming(f,[1],[2],0,2)).toBeNull();
    expect(coupledTiming(f,[0],[1],0,2)).toBeNull();
  });

  it("keeps strong acoustic evidence for a real changed segment", () => {
    const ps = [[.1,.8,.05,.05],[.8,.1,.05,.05],[.01,.01,.01,.97],
      [.8,.1,.05,.05],[.1,.8,.05,.05]];
    const result = coupledTiming(frames(ps),[1,2,1],[1,3,1],0,ps.length)!;
    expect(result.independent).toBeLessThan(0);
    expect(result.coupled).toBeLessThan(0);
  });

  it("reverses a rival advantage supplied by independent shared-phone timing", () => {
    const ps = [
      [.13378794526775928,.1747308285347606,.4578575934391555,.23362363275832465],
      [.3731416754192928,.47951542205154785,.04588305451275701,.10145984801640236],
      [.1016383744397355,.22649346243426624,.5888832983642541,.08298486476174419],
      [.09081245231740456,.6592945080055808,.032458194729704015,.21743484494731066],
    ];
    const result = coupledTiming(frames(ps),[1,2],[1,3],0,ps.length)!;
    expect(result.independent).toBeLessThan(0);
    expect(result.normalizedIndependent).toBeLessThan(0);
    expect(result.uncoupled).toBeLessThan(0);
    expect(result.coupled).toBeCloseTo(.28306958241510405,10);
    expect(result.sharedBoundariesDiffer).toBe(true);
    expect(result.coupled).toBeCloseTo(exhaustive(ps,[1,2],[1,3],[[[0,0]]]),10);
  });
});
