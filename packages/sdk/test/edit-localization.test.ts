import { describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { loadReferenceIndex } from "@tilawa/core";
import { syntheticFixture, withPosteriors } from "../../core/src/fixtures/index.js";
import type { LexicalChangeEvidence, PhonemeEvidence, ReferenceIndex } from "@tilawa/core";
import { addEditLocalizationEvidence, canonicalOccupancy, carryLexicalChangeEvidence, poolWordFeatures, noteLocalizationContext, encodeFrozenWords } from "../src/edit-localization.js";
import type { WordLocalizationContext } from "@tilawa/core";
import { insertionCells } from "../src/seam-insertion.js";
import type { LoadedModel } from "../src/model.js";
import type { InferenceBackend, ModelManifest, Tensor } from "../src/model.js";

const probabilities = [[.5,.3,.2],[.2,.4,.4],[.3,.2,.5],[.6,.3,.1]];
const logp = Float32Array.from(probabilities.flat(),Math.log);

describe("canonical CTC word features", () => {
  it("sums all valid paths, including repeated-label blank separation", () => {
    for (const labels of [[1],[1,1],[1,2]]) {
      const states = [0]; for (const p of labels) states.push(p,0);
      const expected = labels.map(() => Array(4).fill(0)); let total = 0;
      const visit = (path: number[]) => {
        if (path.length === 4) {
          if (path.at(-1)! < states.length-2) return;
          const mass = path.reduce((p,s,t) => p*probabilities[t][states[s]],1); total += mass;
          path.forEach((s,t) => {if (s%2) expected[(s-1)/2][t] += mass;}); return;
        }
        const previous = path.at(-1);
        for (let s = previous ?? 0; s < states.length && s <= (previous === undefined ? 1 : previous+2); s++) {
          if (previous !== undefined && s === previous+2 && (!states[s] || states[s] === states[previous])) continue;
          visit([...path,s]);
        }
      };
      visit([]);
      const actual = canonicalOccupancy(logp,3,labels)!;
      expected.forEach((row,p) => row.forEach((v,t) => expect(actual[p][t]).toBeCloseTo(v/total,6)));
    }
  });

  it("abstains on impossible or unknown pronunciation paths", () => {
    expect(canonicalOccupancy(Float32Array.from([Math.log(.5),Math.log(.5)]),2,[1,1])).toBeNull();
    expect(poolWordFeatures({logp,encoder:new Float32Array(8),frames:4,encoderDim:2,vocab:3},["unknown"],"bc")).toBeNull();
  });

  it("keeps one feature row per word and matches the Python pooling recipe", () => {
    const result = poolWordFeatures({logp,encoder:Float32Array.from([1,2,2,1,3,0,0,3]),frames:4,encoderDim:2,vocab:3},["b","c"],"bc")!;
    expect(result.length).toBe(24); expect(Array.from(result).every(Number.isFinite)).toBe(true);
    // Text histograms remain distinct; both words can get a change score.
    expect(Array.from(result.slice(7,10))).toEqual([0,1,0]);
    expect(Array.from(result.slice(19,22))).toEqual([0,0,1]);
    expect(result[11]).toBeCloseTo(Math.log(2),6);
  });
});

describe("localization certificate transport", () => {
  const chunks: PhonemeEvidence[] = [{inventory:"test",frameMs:40,phonemes:[{p:"b",start:1,end:2,conf:.9}],
    posteriors:{k:1,startFrame:0,nFrames:4,ids:[0,1,1,0],probs:[1,1,1,1]}}];
  const certificate: LexicalChangeEvidence = {word:{surah:1,ayah:1,word:1},expected:"b",observed:"c",fromFrame:1,toFrame:3,score:.1,preservation:.2};
  it("survives serialization and native-frame rechunking without changing primary evidence", () => {
    const before = structuredClone(chunks), result = carryLexicalChangeEvidence(chunks,[certificate]);
    expect(result[0].posteriors).toBe(chunks[0].posteriors);
    expect(chunks).toEqual(before);
    expect(result[0].phonemes.map(({lexicalChanges:_,...p}) => p)).toEqual(chunks[0].phonemes);
    const cells = insertionCells(JSON.parse(JSON.stringify(result)));
    expect(cells.flatMap(c => c.phonemes)[0].lexicalChanges).toEqual([certificate]);
    expect(cells.flatMap(c => Array.from(c.posteriors!.ids))).toEqual([0,1,1,0]);
  });

  it("does not load a side model without primary rows and an existing candidate", async () => {
    const encode = vi.fn(), model = {manifest:{sampleRate:16000,frameMs:40},editLocalization:{encode}} as unknown as LoadedModel;
    expect(await addEditLocalizationEvidence(model,new Float32Array(16000),[],{} as ReferenceIndex)).toEqual([]);
    expect(encode).not.toHaveBeenCalled();
  });
});

describe("complete contextual occurrences", () => {
  it("merges overlapping views without treating a later re-read as the same occurrence", () => {
    const contexts: WordLocalizationContext[] = [];
    noteLocalizationContext(contexts,{words:[10,11,12],coveredWords:[10,11],fromFrame:0,toFrame:20});
    noteLocalizationContext(contexts,{words:[10,11,12],coveredWords:[11,12],fromFrame:10,toFrame:30});
    expect(contexts).toEqual([{words:[10,11,12],coveredWords:[11,12,10],fromFrame:0,toFrame:30}]);
    noteLocalizationContext(contexts,{words:[10,11,12],coveredWords:[10,11,12],fromFrame:40,toFrame:70});
    expect(contexts).toHaveLength(2);
  });
});

describe("frozen feature streaming state", () => {
  it("uses a separate zero-initialized state and retains the final output count", async () => {
    const states: number[] = [], manifest = {tokens:"bc",fbank:{mels:80,padValue:0},window:11,shift:8,
      states:[{name:"cache",shape:[1],dtype:"float32"}]} as ModelManifest;
    const backend = {run:vi.fn(async (feeds: Record<string,Tensor>) => {
      const state = Number(feeds.cache.data[0]); states.push(state);
      return {log_probs:{type:"float32",dims:[1,2,3],data:new Float32Array(6).fill(Math.log(1/3))},
        encoder_features:{type:"float32",dims:[1,2,2],data:Float32Array.from([state,0,state+1,0])},
        new_cache:{type:"float32",dims:[1],data:Float32Array.of(state+2)}};
    })} as unknown as InferenceBackend;
    const view = await encodeFrozenWords(backend,manifest,new Float32Array(400+160*30));
    expect(states).toEqual([0,2,4]); expect(view.frames).toBe(6); expect(view.encoder.length).toBe(12);
    await encodeFrozenWords(backend,manifest,new Float32Array(400+160*30));
    expect(states).toEqual([0,2,4,0,2,4]);
  });
});

describe("SDK discovery through certificate enrichment", () => {
  it("scores a full contextual occurrence while preserving the primary rows", async () => {
    const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json",import.meta.url),"utf8")));
    const from = {surah:2,ayah:255,word:1}, to = {surah:2,ayah:256,word:10}, target = {surah:2,ayah:255,word:23};
    const find = (r: typeof from) => index.words.find(w => w.surah === r.surah && w.ayah === r.ayah && w.word === r.word)!;
    const word = find(target), start = find(from), rival = find({surah:108,ayah:1,word:3});
    const fixture = syntheticFixture(index,{id:"localization",from,to,edits:[
      {kind:"del",at:word.start-start.start,count:word.end-word.start},
      {kind:"ins",at:word.start-start.start,text:index.phonemes.slice(rival.start,rival.end)},
    ]});
    const chunks = [withPosteriors(index,fixture.evidence[0])], n = chunks[0].posteriors!.nFrames, vocab = index.tokens.size+1;
    const score = vi.fn(async (_: Float32Array, words: number) => ({contextual:Array(words).fill(.1),independent:Array(words).fill(.8)}));
    const model = {manifest:{sampleRate:16000,frameMs:40,tokens:[...index.tokens].join("")},editLocalization:{
      preservation:.2,independentPreservation:.3,score,async encode() {return {frames:n,encoderDim:2,vocab,
        encoder:new Float32Array(n*2),logp:new Float32Array(n*vocab).fill(Math.log(1/vocab))};},
    }} as unknown as LoadedModel;
    const enriched = await addEditLocalizationEvidence(model,new Float32Array(16000),chunks,index);
    expect(score).toHaveBeenCalledOnce();
    expect(score.mock.calls[0][1]).toBeGreaterThan(1);
    expect(enriched[0].posteriors).toBe(chunks[0].posteriors);
    expect(enriched[0].phonemes[0].lexicalChanges).toHaveLength(1);
    expect(enriched[0].phonemes[0].lexicalChanges![0]).toMatchObject({word:target,score:.1,preservation:.2});
    expect(chunks[0].phonemes[0].lexicalChanges).toBeUndefined();
  });
});
