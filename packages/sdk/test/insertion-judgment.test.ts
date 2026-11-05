import { describe, expect, it, vi } from "vitest";
import type { PhonemeEvidence, ReferenceIndex, SeamInsertionCandidate, SeamInsertionWindow } from "@tilawa/core";
import type { LoadedModel } from "../src/model.js";

const candidates = vi.hoisted(() => [] as SeamInsertionCandidate[]);
const seams = vi.hoisted(() => [] as SeamInsertionWindow[]);
vi.mock("@tilawa/core", async original => ({
  ...await original<typeof import("@tilawa/core")>(),
  createTracker: () => ({ push: () => [], end: () => [] }),
  createChecker: (_: unknown, options: { seamInsertion(c: SeamInsertionCandidate): boolean; seamPhoneLoop(c: SeamInsertionWindow): void }) => ({
    push: () => [], end: () => { for (const c of candidates) expect(options.seamInsertion(c)).toBe(true); for (const c of seams) options.seamPhoneLoop(c); },
  }),
  lexicalWords: () => ({ surplusWord: (_: number, said: string) => said, fragmentOfSeam: () => false }),
}));
import { addInsertionEvidence } from "../src/seam-insertion.js";

describe("warm whole-seam teacher judgment", () => {
  it("proposes an identity-free insertion even with no student spelling", async () => {
    candidates.splice(0); seams.splice(0, seams.length, { word: 2, seamFromFrame:122,seamToFrame:130,prefixFrame:120 });
    const index = { inventoryId:"test",tokens:new Set("بتثجحخ"),phonemes:"بتثح",
      words:[..."بتثح"].map((_,i)=>({surah:1,ayah:i<2?1:2,word:i%2+1,start:i,end:i+1})) } as ReferenceIndex;
    const decode=vi.fn(async ():Promise<PhonemeEvidence>=>{
      const labels=new Map([[100,1],[102,2],[103,4],[105,6],[107,3],[109,5]]);
      const rows=Array.from({length:118},(_,f)=>Array.from({length:7},(_,id)=>id===(labels.get(f)??0)?.9994:.0001));
      return {inventory:"test",frameMs:40,phonemes:[],posteriors:{startFrame:0,nFrames:118,k:7,
        ids:rows.flatMap(r=>r.map((_,i)=>i)),probs:rows.flat()}};
    });
    const model={manifest:{sampleRate:16000},insertionTeacher:{decode}} as unknown as LoadedModel;
    const evidence:PhonemeEvidence[]=[{inventory:"test",frameMs:40,phonemes:[{p:"ب",start:120,end:120,conf:1}],
      posteriors:{startFrame:0,nFrames:140,k:1,ids:Array(140).fill(0),probs:Array(140).fill(1)}}];
    const enriched=await addInsertionEvidence(model,new Float32Array(140*640),evidence,index);
    expect(enriched[0].phonemes[0].seamInsertions).toMatchObject([{kind:"phone-loop",observed:"جخ",fromFrame:123,toFrame:126}]);
    expect(enriched[0].posteriors).toBe(evidence[0].posteriors);
    expect(decode).toHaveBeenCalledOnce();
    seams.splice(0);
  });
  it("uses the two-word prefix and native clock, sharing crops and freeing the insertion timing", async () => {
    const index = { inventoryId: "test", tokens: new Set("بتثجح"), phonemes: "بتثح",
      words: [..."بتثح"].map((_, i) => ({ surah: 1, ayah: i < 2 ? 1 : 2, word: i % 2 + 1, start: i, end: i + 1 })) } as ReferenceIndex;
    const c = { word: 2, said: "ج", fromFrame: 125, toFrame: 126, seamFromFrame: 122, seamToFrame: 130, prefixFrame: 120 };
    candidates.splice(0, candidates.length, c, { ...c });
    const decode = vi.fn(async (samples: Float32Array, k: number): Promise<PhonemeEvidence> => {
      expect(samples.length).toBe((138 - 20) * 640);
      expect(k).toBe(6);
      // Teacher hears the addition two frames earlier than the student's fixed candidate interval.
      const labels = new Map([[100, 1], [102, 2], [103, 4], [105, 3], [107, 5]]);
      const rows = Array.from({ length: 118 }, (_, f) => Array.from({ length: 6 }, (_, id) => id === (labels.get(f) ?? 0) ? .9995 : .0001));
      return { inventory: "test", frameMs: 40, phonemes: [], posteriors: {
        startFrame: 0, nFrames: rows.length, k, ids: rows.flatMap(r => r.map((_, i) => i)), probs: rows.flat() } };
    });
    const model = { manifest: { sampleRate: 16000 }, insertionTeacher: { decode } } as unknown as LoadedModel;
    const evidence: PhonemeEvidence[] = [{ inventory: "test", frameMs: 40,
      phonemes: [{ p: "ب", start: 120, end: 120, conf: 1 }],
      posteriors: { startFrame: 0, nFrames: 140, k: 1, ids: Array(140).fill(0), probs: Array(140).fill(1) } }];
    const enriched = await addInsertionEvidence(model, new Float32Array(140 * 640), evidence, index);
    expect(decode).toHaveBeenCalledOnce();
    expect(enriched[0].phonemes[0].seamInsertions).toMatchObject([{ fromFrame: 125, toFrame: 126 }]);
    expect(enriched[0].phonemes[0].seamInsertions![0].marginal).toBeGreaterThan(.5);
    expect(enriched[0].posteriors).toBe(evidence[0].posteriors);
  });
});
