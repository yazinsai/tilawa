import { describe, expect, it, vi } from "vitest";
import type { PhonemeEvidence, ReferenceIndex, SeamInsertionCandidate } from "@tilawa/core";
import type { LoadedModel } from "../src/model.js";

const candidates = vi.hoisted(() => [] as SeamInsertionCandidate[]);
const ear = vi.hoisted(() => ({ enabled: false }));
vi.mock("../src/word-ear.js", () => ({
  wordEarSpans: () => ear.enabled ? [{word:2,fromFrame:125,toFrame:138,seamFromFrame:122,seamToFrame:130,prefixFrame:120}] : [],
  wordEarCandidates: (_:unknown,s: {word:number;fromFrame:number;toFrame:number;seamFromFrame:number;seamToFrame:number;prefixFrame:number},heard:{text:string}[]) =>
    heard.length ? [{...s,said:heard[0].text}] : [],
}));
vi.mock("@tilawa/core", async original => ({
  ...await original<typeof import("@tilawa/core")>(),
  createTracker: () => ({ push: () => [], end: () => [] }),
  createChecker: (_: unknown, options: { seamInsertion(c: SeamInsertionCandidate): boolean }) => ({
    push: () => [], end: () => { for (const c of candidates) expect(options.seamInsertion(c)).toBe(true); },
  }),
  lexicalWords: () => ({ surplusWord: (_: number, said: string) => said }),
}));
import { addInsertionEvidence } from "../src/seam-insertion.js";

describe("warm whole-seam teacher judgment", () => {
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

  it("sends low-confidence word-ear proposals through the same teacher lattice even without a student candidate", async () => {
    candidates.splice(0); ear.enabled = true;
    const index = {inventoryId:"test",tokens:new Set("بتثجح"),phonemes:"بتثح",
      words:[..."بتثح"].map((_,i) => ({surah:1,ayah:i<2?1:2,word:i%2+1,start:i,end:i+1}))} as ReferenceIndex;
    const evidence: PhonemeEvidence[] = [{inventory:"test",frameMs:40,
      phonemes:[{p:"ب",start:120,end:120,conf:.1}],
      posteriors:{startFrame:0,nFrames:140,k:1,ids:Array(140).fill(0),probs:Array(140).fill(1)}}];
    const wordDecode = vi.fn(async () => [{text:"ج",logp:-Infinity}]);
    const judge = async (extra:boolean) => {
      const labels = new Map([[100,1],[102,2],...(extra?[[103,4]] as [number,number][]:[]),[105,3],[107,5]]);
      const rows = Array.from({length:118},(_,f) => Array.from({length:6},(_,id) => id===(labels.get(f)??0) ? .9995:.0001));
      const model = {manifest:{sampleRate:16000},wordEar:{decode:wordDecode},insertionTeacher:{decode:async () => ({
        inventory:"test",frameMs:40,phonemes:[],posteriors:{startFrame:0,nFrames:118,k:6,
          ids:rows.flatMap(r=>r.map((_,i)=>i)),probs:rows.flat()}})}} as unknown as LoadedModel;
      return addInsertionEvidence(model,new Float32Array(140*640),evidence,index);
    };
    try {
      const present = await judge(true), clean = await judge(false);
      expect(present[0].phonemes[0].seamInsertions![0]).toMatchObject({source:"word-ear",observed:"ج"});
      expect(present[0].phonemes[0].seamInsertions![0].marginal).toBeGreaterThan(.5);
      expect(clean[0].phonemes[0].seamInsertions![0].marginal).toBeLessThan(.5);
      expect(present[0].posteriors).toBe(evidence[0].posteriors);
      expect(present[0].phonemes[0].conf).toBe(.1);
    } finally { ear.enabled = false; }
  });
});
