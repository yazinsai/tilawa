import { describe, expect, it } from "vitest";
import { PHRASE_SWITCH } from "../src/parallel-phrases.js";
import { PosteriorFrames } from "../src/ctc.js";
import { seamInsertionMarginal, seamPhoneLoop } from "../src/seam-insertion.js";

function framesOf(rows: number[][]) {
  const frames = new PosteriorFrames(rows[0].length - 1);
  frames.append({ startFrame: 0, nFrames: rows.length, k: rows[0].length,
    ids: rows.flatMap(r => r.map((_, i) => i)), probs: rows.flat() });
  return frames;
}
const collapse = (path: number[]) => path.filter((p, i) => p && (i === 0 || p !== path[i - 1])).join("");
function exhaustive(rows: number[][], cost: number, plain: string, extra: string) {
  let no = 0, yes = 0;
  const visit = (path: number[], p: number) => {
    if (path.length === rows.length) {
      const text = collapse(path);
      if (text === plain) no += p;
      if (text === extra) yes += p * Math.exp(-cost);
    } else rows[path.length].forEach((v, id) => visit([...path, id], p * v));
  };
  visit([], 1);
  return yes / (yes + no);
}
describe("whole-seam insertion marginal", () => {
  it("agrees with exhaustive CTC paths, including free blank timing", () => {
    const rows = [[.2, .4, .3, .1], [.3, .1, .5, .1], [.1, .2, .2, .5], [.5, .1, .1, .3]];
    const frames = framesOf(rows), cost = 1.7;
    expect(seamInsertionMarginal(frames, [[[1]]], [[2]], [[[3]]], 0, rows.length, cost, false))
      .toBeCloseTo(exhaustive(rows, cost, "13", "123"), 12);
  });
  it("requires a blank between equal labels on the no-word path", () => {
    const rows = [[.3, .4, .3], [.2, .4, .4], [.5, .3, .2], [.1, .6, .3]];
    expect(seamInsertionMarginal(framesOf(rows), [[[1]]], [[2]], [[[1]]], 0, rows.length, 2, false))
      .toBeCloseTo(exhaustive(rows, 2, "11", "121"), 12);
  });
  it("recovers extra speech with free boundaries and rejects neighbour-only speech", () => {
    const rows = (text: number[]) => text.flatMap(id => [Array.from({length: 4}, (_, i) => i === id ? .9997 : .0001), [.9997,.0001,.0001,.0001]]);
    const extra = framesOf(rows([1,2,3])), clean = framesOf(rows([1,3]));
    expect(seamInsertionMarginal(extra, [[[1]]], [[2]], [[[3]]], 0, extra.endFrame, 4)).toBeGreaterThan(.5);
    expect(seamInsertionMarginal(clean, [[[1]]], [[2]], [[[3]]], 0, clean.endFrame, 4)).toBeLessThan(.5);
  });
  it("lets contiguous repeats explain a restart without the insertion", () => {
    const rows = [1,1,3].flatMap(id => [Array.from({length: 4}, (_, i) => i === id ? .9997 : .0001), [.9997,.0001,.0001,.0001]]);
    const frames = framesOf(rows);
    expect(seamInsertionMarginal(frames, [[[1]]], [[2]], [[[3]]], 0, frames.endFrame, 4)).toBeLessThan(.5);
  });
  it("matches closed forward sums with word and contiguous side repeat arcs", () => {
    const rows = [1,2,3,4,3,4,5,6,7].flatMap(id => [
      Array.from({length: 8}, (_, i) => i === id ? .993 : .001),
      Array.from({length: 8}, (_, i) => i === 0 ? .993 : .001),
    ]);
    const frames = framesOf(rows), to = frames.endFrame;
    const no = -frames.nll([1,2,3,4,6,7],0,to,[[0,1],[2,3],[4,4],[5,5],[0,3],[4,5]]);
    const yes = -frames.nll([1,2,3,4,5,6,7],0,to,[[0,1],[2,3],[4,4],[5,5],[6,6],[0,3],[5,6]])-4;
    expect(seamInsertionMarginal(frames, [[[1,2]],[[3,4]]], [[5]], [[[6]],[[7]]],0,to,4))
      .toBeCloseTo(1/(1+Math.exp(no-yes)),12);
  });
  it("does not give duplicate pronunciation forms extra probability mass", () => {
    const rows = [[.2,.4,.3,.1],[.3,.1,.5,.1],[.1,.2,.2,.5],[.5,.1,.1,.3]], frames = framesOf(rows);
    expect(seamInsertionMarginal(frames, [[[1],[1]]], [[2],[2]], [[[3],[3]]], 0, rows.length, 4))
      .toBe(seamInsertionMarginal(frames, [[[1]]], [[2]], [[[3]]], 0, rows.length, 4));
  });
  it("returns no decision for missing evidence or an impossible read", () => {
    const frames = framesOf([[.25,.25,.25,.25]]);
    expect(seamInsertionMarginal(frames, [[[1]]], [[2]], [[[3]]], -1, 1, 4)).toBeNull();
    expect(seamInsertionMarginal(frames, [[[1]]], [[2]], [[[3]]], 0, 2, 4)).toBeNull();
    expect(seamInsertionMarginal(frames, [[[1]]], [[2]], [[[3]]], 0, 1, 4)).toBeNull();
  });
});


describe("identity-free seam phone loop", () => {
  it("sums exactly the CTC paths with two consonant onsets", () => {
    const rows = [
      [.1,.5,.2,.1,.1], [.2,.1,.4,.1,.2], [.3,.1,.3,.1,.2],
      [.2,.1,.4,.1,.2], [.1,.1,.2,.5,.1],
    ];
    let yes = 0, no = 0;
    const visit = (path: number[], mass: number) => {
      if (path.length < rows.length) {
        rows[path.length].forEach((p, id) => visit([...path, id], mass * p)); return;
      }
      const text = collapse(path);
      if (text === "13") no += mass;
      if (/^1[24]+3$/.test(text) && [...text.slice(1,-1)].filter(p => p === "2").length >= 2)
        yes += mass * Math.exp(-PHRASE_SWITCH);
    };
    visit([], 1);
    const result = seamPhoneLoop(framesOf(rows), [[[1]]], [[[3]]], [2,4], [2], 0, rows.length, PHRASE_SWITCH, false);
    expect(result!.marginal).toBeCloseTo(yes / (yes + no), 12);
  });
  const paint = (text: number[]) => framesOf(text.map(id => Array.from({ length: 6 }, (_, i) => i === id ? .9995 : .0001)));
  it("returns the inserted Viterbi phones and native free boundaries", () => {
    const frames = paint([1,0,2,2,0,4,0,5,0,3]);
    expect(seamPhoneLoop(frames, [[[1]]], [[[3]]], [1,2,3,4,5], [1,2,3,5],0,10,PHRASE_SWITCH))
      .toMatchObject({ phones: [2,4,5], fromFrame: 2, toFrame: 8 });
  });
  it("extracts a conditional insertion when many paths outweigh one no-word timing", () => {
    const rows = [
      [0,1,0,0,0], ...Array.from({length:20},()=>[.7,0,.15,0,.15]), [0,0,0,1,0],
    ];
    const frames={startFrame:0,endFrame:rows.length,row(f:number){return Float64Array.from(rows[f],Math.log)}};
    const result=seamPhoneLoop(frames,[[[1]]],[[[3]]],[2,4],[2,4],0,rows.length,PHRASE_SWITCH,false)!;
    expect(result.marginal).toBeGreaterThan(.5);
    expect(result.phones.length).toBeGreaterThanOrEqual(2);
    expect(result.fromFrame).toBeGreaterThanOrEqual(1);
    expect(result.toFrame).toBeLessThanOrEqual(21);
  });
  it("does not advance the consonant count on a held frame", () => {
    const text = [1,2,2,2,3];
    const frames = { startFrame:0,endFrame:text.length,row(f:number) {
      const row=new Float64Array(4).fill(-Infinity); row[text[f]]=0; return row;
    } };
    // No blank or other label can split this deterministic held run into
    // two onsets. Both the no-word and the constrained loop read are impossible.
    expect(seamPhoneLoop(frames, [[[1]]], [[[3]]], [2], [2],0,text.length,PHRASE_SWITCH)).toBeNull();
  });
  it("does not mistake blank or vowel-only mass for a word", () => {
    const text=[1,0,4,4,4,0,3];
    const frames={startFrame:0,endFrame:text.length,row(f:number){
      const row=new Float64Array(6).fill(-Infinity); row[0]=Math.log(.1); row[text[f]]=Math.log(.9); return row;
    }};
    expect(seamPhoneLoop(frames, [[[1]]], [[[3]]], [1,2,3,4,5], [1,2,3,5],0,text.length,PHRASE_SWITCH)!.marginal).toBeLessThan(.5);
    const silence=paint([1,0,0,0,3]);
    expect(seamPhoneLoop(silence, [[[1]]], [[[3]]], [1,2,3,4,5], [1,2,3,5],0,5,PHRASE_SWITCH)!.marginal).toBeLessThan(.5);
  });
  it("lets the existing contiguous repeat arcs explain restarts", () => {
    const frames = paint([1,0,2,0,1,0,2,0,3]);
    expect(seamPhoneLoop(frames, [[[1,2]]], [[[3]]], [1,2,3,4,5], [1,2,3,5],0,9,PHRASE_SWITCH)!.marginal).toBeLessThan(.5);
  });
});
