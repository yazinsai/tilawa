import { describe, expect, it } from "vitest";
import { seamPhoneLoop } from "../src/seam-insertion.js";
import { PHRASE_SWITCH } from "../src/parallel-phrases.js";

function paint(text: number[]) {
  return { startFrame: 0, endFrame: text.length, row(f: number) {
    return Float64Array.from({ length: 6 }, (_, p) => Math.log(p === text[f] ? .9995 : .0001));
  } };
}
const score = (text: number[], interval: readonly [number, number]) =>
  seamPhoneLoop(paint(text), [[[1]]], [[[3]]], [1,2,3,4,5], [1,2,3,5],
    0, text.length, PHRASE_SWITCH, interval, false);

describe("tracker-bounded teacher insertion loop", () => {
  it("matches exhaustive CTC path mass with two consonant onsets", () => {
    const rows = [[.1,.5,.2,.1,.1],[.2,.1,.4,.1,.2],[.3,.1,.3,.1,.2],
      [.2,.1,.4,.1,.2],[.1,.1,.2,.5,.1]];
    let yes=0, no=0;
    const visit=(path:number[], mass:number) => {
      if (path.length < rows.length) {
        rows[path.length].forEach((p,id)=>visit([...path,id],mass*p)); return;
      }
      const text=path.filter((p,i)=>p && (!i || p!==path[i-1])).join("");
      if (text === "13") no+=mass;
      if (/^1[24]+3$/.test(text) && [...text.slice(1,-1)].filter(p=>p==="2").length>=2)
        yes+=mass*Math.exp(-PHRASE_SWITCH);
    };
    visit([],1);
    const frames={startFrame:0,endFrame:rows.length,row:(f:number)=>Float64Array.from(rows[f],Math.log)};
    expect(seamPhoneLoop(frames,[[[1]]],[[[3]]],[2,4],[2],0,rows.length,PHRASE_SWITCH,[0,rows.length],false)!.marginal)
      .toBeCloseTo(yes/(yes+no),12);
  });

  it("keeps every inserted phone within the interval while waiting through a later blank pause", () => {
    const result = score([1,0,2,2,0,4,5,0,0,0,3], [2,7])!;
    expect(result.marginal).toBeGreaterThan(.5);
    expect(result.phones).toEqual([2,4,5]);
    expect(result.fromFrame).toBe(2);
    expect(result.toFrame).toBe(7);
  });

  it("cannot buy consonants outside its interval", () => {
    const result = score([1,2,0,0,0,5,3], [3,4])!;
    expect(result.marginal).toBe(0);
    expect(result.phones).toEqual([]);
  });

  it("keeps neighbour-only speech quiet", () => {
    expect(score([1,0,0,0,3], [1,4])!.marginal).toBeLessThan(.5);
  });

  it("does not count held frames as two consonant onsets", () => {
    const text = [1,2,2,2,3];
    const frames = { startFrame:0,endFrame:text.length,row(f:number) {
      const row=new Float64Array(4).fill(-Infinity); row[text[f]]=0; return row;
    } };
    expect(seamPhoneLoop(frames, [[[1]]], [[[3]]], [2], [2], 0,5,PHRASE_SWITCH,[1,4],false)).toBeNull();
  });

  it("rejects empty, fractional and out-of-evidence intervals", () => {
    for (const interval of [[2,2],[-1,3],[1,6],[1.5,3]] as const)
      expect(score([1,0,2,5,3], interval)).toBeNull();
  });
});
