import { beforeEach, describe, expect, it, vi } from "vitest";
import type { PhonemeEvidence, ReferenceIndex, SeamInsertionCandidate, SeamInsertionWindow, EngineEvent } from "@tilawa/core";
import type { LoadedModel } from "../src/model.js";

const candidates = vi.hoisted(() => [] as SeamInsertionCandidate[]);
const seams = vi.hoisted(() => [] as SeamInsertionWindow[]);
const steps = vi.hoisted(() => [] as EngineEvent[]);
vi.mock("@tilawa/core", async original => ({
  ...await original<typeof import("@tilawa/core")>(),
  createTracker: () => ({ push: () => [], end: () => steps }),
  createChecker: (_: unknown, options: { seamInsertion(c: SeamInsertionCandidate): boolean; seamPhoneLoop(c: SeamInsertionWindow): void }) => ({
    push: () => [], end: () => { for (const c of candidates) expect(options.seamInsertion(c)).toBe(true); for (const s of seams) options.seamPhoneLoop(s); },
  }),
  lexicalWords: () => ({ surplusWord: (_: number, said: string) => said, fragmentOfSeam: () => false }),
}));
import { addInsertionEvidence } from "../src/seam-insertion.js";

beforeEach(() => {
  candidates.length = 0;
  seams.length = 0;
  steps.length = 0;
});

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
});


describe("pause-onset cold agreement", () => {
  async function judge(warm: boolean, cold: boolean, disjoint = false, pause = true, step = true) {
    candidates.length = 0;
    const index = { inventoryId: "test", tokens: new Set("بتثجحخ"), phonemes: "بتثح",
      words: [..."بتثح"].map((_,i) => ({ surah:1,ayah:i<2?1:2,word:i%2+1,start:i,end:i+1 })) } as ReferenceIndex;
    seams.splice(0,seams.length,{ word:2,seamFromFrame:122,seamToFrame:134,prefixFrame:120,
      pauseFromFrame:pause?124:undefined });
    steps.splice(0,steps.length,...(step ? [{ type:"position",state:"locked",timeMs:128*40,
      observed:[{word:{surah:1,ayah:2,word:2}}] } as EngineEvent] : []));
    const decode = vi.fn(async (samples: Float32Array): Promise<PhonemeEvidence> => {
      const isCold = samples.length === (136-124)*640;
      expect(samples.length).toBe(isCold ? 12*640 : (142-20)*640);
      // The sample values encode absolute frame indices: cold input starts at
      // pause onset with no left padding; warm input retains its four seconds.
      const start = samples[0];
      expect(start).toBe(isCold?124:20);
      const labels = new Map([[120,1],[122,2],[129,3],[131,5]]);
      if (isCold && disjoint) { labels.delete(129); labels.delete(131); labels.set(132,3); labels.set(134,5); }
      if (isCold ? cold : warm) {
        labels.set(isCold && disjoint ? 130:125,4);
        labels.set(isCold && disjoint ? 131:127,6);
      }
      const rows = Array.from({length:samples.length/640},(_,f) => Array.from({length:7},(_,id) => id === (labels.get(start+f)??0) ? .9994:.0001));
      return {inventory:"test",frameMs:40,phonemes:[],posteriors:{startFrame:0,nFrames:rows.length,k:7,
        ids:rows.flatMap(r=>r.map((_,id)=>id)),probs:rows.flat()}};
    });
    const model = {manifest:{sampleRate:16000},insertionTeacher:{decode}} as unknown as LoadedModel;
    const evidence: PhonemeEvidence[] = [{inventory:"test",frameMs:40,
      phonemes:[{p:"ب",start:120,end:120,conf:1}],posteriors:{startFrame:0,nFrames:150,k:1,ids:Array(150).fill(0),probs:Array(150).fill(1)}}];
    const audio = Float32Array.from({length:150*640},(_,i)=>Math.floor(i/640));
    const enriched = await addInsertionEvidence(model,audio,evidence,index);
    expect(enriched[0].posteriors).toBe(evidence[0].posteriors);
    expect(enriched[0].phonemes.map(({seamInsertions:_,...p})=>p)).toEqual(evidence[0].phonemes);
    return { decode, certificates:enriched[0].phonemes[0].seamInsertions??[] };
  }
  it("emits only for warm and cold loop spans that overlap", async () => {
    const {decode,certificates} = await judge(true,true);
    expect(decode).toHaveBeenCalledTimes(2);
    expect(certificates).toMatchObject([{kind:"phone-loop",observed:"جخ",fromFrame:125,toFrame:128}]);
  });
  it("rejects warm-only, cold-only and disjoint loop evidence", async () => {
    expect((await judge(true,false)).certificates).toEqual([]);
    expect((await judge(false,true)).certificates).toEqual([]);
    expect((await judge(true,true,true)).certificates).toEqual([]);
  });
  it("abstains without a labelled pause or a committed word-2 step", async () => {
    expect((await judge(true,true,false,false)).certificates).toEqual([]);
    expect((await judge(true,true,false,true,false)).certificates).toEqual([]);
  });
});
