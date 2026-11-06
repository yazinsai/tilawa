import { readFileSync } from "node:fs";
import { expect, it, vi } from "vitest";
import { loadReferenceIndex, seamWordCompetitors } from "@tilawa/core";
import type { PhonemeEvidence } from "@tilawa/core";
import type { LoadedModel } from "../src/model.js";
const fixed = vi.hoisted(() => ({ spans: [] as any[] }));
vi.mock("../src/word-ear.js", async original => ({
  ...await original<typeof import("../src/word-ear.js")>(), wordEarSpans: () => fixed.spans,
}));
import { addWordSslEvidence, confirmedWordInterval } from "../src/word-ssl.js";
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")));
const word = index.words.findIndex(w => w.surah === 1 && w.ayah === 2 && w.word === 1);
const phones = (s: string, start = 0) => [...s].map((p, i) => ({ p, start: start + i * 2, end: start + i * 2, conf: .1 }));
it("searches native SSL intervals without giving the proposed word free ends", () => {
  expect(confirmedWordInterval("سَلِم", phones("بتَلِمب", 100), ["وَقَل"]))
    .toMatchObject({ nativeFrom: 102, nativeTo: 111, cost: 10, hard: 1 });
  expect(confirmedWordInterval("سَلِم", phones("تَلِب"), ["وَقَل"])).toBeUndefined();
  expect(confirmedWordInterval("سَلِم", phones("تَلِم"), ["تَلِم"])).toBeUndefined();
});
it("stores rank evidence without changing primary tokens, posterior rows or teacher judgments", async () => {
  fixed.spans = [{ word, fromFrame: 10, toFrame: 40, seamFromFrame: 0, seamToFrame: 32 }];
  const teacher = { word: { surah: 1, ayah: 2, word: 1 }, observed: "بت", fromFrame: 1, toFrame: 2, marginal: .7 };
  const primary: PhonemeEvidence[] = [{ inventory: index.inventoryId, frameMs: 40,
    phonemes: [{ p: "ب", start: 0, end: 0, conf: 1, seamInsertions: [teacher] }],
    posteriors: { startFrame: 0, nFrames: 1, k: 1, ids: [1], probs: [1] } }];
  const wordEar = { decode: vi.fn().mockResolvedValue([{ text: "كريم", logp: -Infinity }]) };
  const sslEar = { decodeSeams: vi.fn().mockResolvedValue(phones("كَرِۦم", 24)) };
  const model = { manifest: { sampleRate: 16000 }, wordEar, sslEar } as unknown as LoadedModel;
  const enriched = await addWordSslEvidence(model, new Float32Array(30000), primary, index);
  expect(enriched[0].phonemes[0].seamWordAgreements?.length).toBeGreaterThan(0);
  expect(enriched[0].posteriors).toBe(primary[0].posteriors);
  expect(enriched[0].phonemes[0].seamInsertions).toBe(primary[0].phonemes[0].seamInsertions);
  const { seamWordAgreements, ...unchanged } = enriched[0].phonemes[0];
  expect(unchanged).toEqual(primary[0].phonemes[0]);
  expect(primary[0].phonemes[0].seamWordAgreements).toBeUndefined();
  expect(sslEar.decodeSeams.mock.calls[0][1]).toContainEqual({ fromFrame: 20, toFrame: 80 });
  expect(confirmedWordInterval("كَرِۦم", phones("كَرِۦم", 24), seamWordCompetitors(index, word))).toBeDefined();
  const absent = { ...model, sslEar: undefined };
  expect(await addWordSslEvidence(absent, new Float32Array(30000), primary, index)).toBe(primary);
});
