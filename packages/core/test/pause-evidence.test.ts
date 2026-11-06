import { describe,expect,it } from "vitest";
import type { PauseMaskEvidence,PhonemeEvidence } from "../src/evidence.js";
import { pauseEvidenceReader } from "../src/pause-evidence.js";

const scope: PauseMaskEvidence = { inventory: "test",frameMs: 40,fromFrame: 3,toFrame: 8,
  phonemes: [{ p: "ت",start: 4,end: 6,conf: .9 }],
  posteriors: { startFrame: 3,nFrames: 5,k: 1,ids: [0,2,2,2,0],probs: Array(5).fill(.9) } };
const primary: PhonemeEvidence = { inventory: "test",frameMs: 40,
  phonemes: [{ p: "ب",start: 1,end: 1,conf: .8 },{ p: "ن",start: 9,end: 9,conf: .8 }],
  posteriors: { startFrame: 0,nFrames: 12,k: 1,ids: [0,1,0,0,0,0,0,0,0,3,0,0],probs: Array(12).fill(.8) } };
const carried = () => JSON.parse(JSON.stringify({ ...primary,phonemes: primary.phonemes.map((p,i) =>
  i ? p : { ...p,pauseMaskEvidence: [scope] }) })) as PhonemeEvidence;

describe("pause-trained checker evidence", () => {
  it("survives JSON and primary rechunking, releases each run once and preserves all outside rows", () => {
    for (const width of [1,4,12]) {
      const input=carried(), original=structuredClone(input), read=pauseEvidenceReader(new Set("بتن"));
      const got: PhonemeEvidence[]=[];
      for (let from=0;from<12;from+=width) {
        const end=Math.min(12,from+width), r=input.posteriors!;
        got.push(read({ ...input,phonemes: input.phonemes.filter(p => p.start>=from && p.start<end),
          posteriors: { ...r,startFrame: from,nFrames: end-from,ids: Array.from(r.ids).slice(from,end),probs: Array.from(r.probs).slice(from,end) } }));
      }
      const phones=got.flatMap(e => e.phonemes).map(({ pauseMaskEvidence: _,...p }) => p);
      expect(phones).toEqual([primary.phonemes[0],scope.phonemes[0],primary.phonemes[1]]);
      expect(got.flatMap(e => Array.from(e.posteriors!.ids))).toEqual([0,1,0,0,2,2,2,0,0,3,0,0]);
      expect(input).toEqual(original);
      if (width===1) {
        expect(got[4].phonemes).toEqual([]);
        expect(got[5].phonemes).toEqual([]);
        expect(got[6].phonemes).toEqual(scope.phonemes);
      }
    }
  });
  it("clips boundary-crossing primary runs and ignores an invalid auxiliary clock", () => {
    const input=carried(); input.phonemes=[{ p: "ب",start: 1,end: 9,conf: .8,pauseMaskEvidence: [scope] }];
    const got=pauseEvidenceReader(new Set("بتن"))(input);
    expect(got.phonemes.map(({ pauseMaskEvidence: _,...p }) => p)).toEqual([
      { p: "ب",start: 1,end: 2,conf: .8 },scope.phonemes[0],{ p: "ب",start: 8,end: 9,conf: .8 },
    ]);
    const wrong=carried(); wrong.phonemes[0].pauseMaskEvidence![0].frameMs=80;
    expect(pauseEvidenceReader(new Set("بتن"))(wrong)).toBe(wrong);
    expect(pauseEvidenceReader(new Set("بتن"))(primary)).toBe(primary);
  });
});
