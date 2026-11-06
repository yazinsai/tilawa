import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import { loadReferenceIndex } from "@tilawa/core";
import type { PhonemeEvidence, SeamInsertionEvidence } from "@tilawa/core";
import { addInsertionEvidence, carryInsertionEvidence, insertionCells } from "../src/seam-insertion.js";
import type { LoadedModel } from "../src/model.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")));
const metadata: SeamInsertionEvidence[] = [{ word: { surah: 1, ayah: 2, word: 1 }, observed: "بت", fromFrame: 2, toFrame: 5, marginal: .8 }];
const evidence: PhonemeEvidence[] = [{ inventory: "test", frameMs: 40,
  phonemes: [{ p: "ب", start: 1, end: 3, conf: .9 }, { p: "ت", start: 5, end: 5, conf: .8 }],
  posteriors: { startFrame: 0, nFrames: 7, k: 1, ids: [0, 1, 1, 1, 0, 2, 0], probs: Array(7).fill(1) } }];

describe("seam insertion evidence transport", () => {
  it("preserves primary phones and posterior rows, including through JSON and frame rechunking", () => {
    const original = structuredClone(evidence);
    const enriched = carryInsertionEvidence(evidence, metadata);
    expect(enriched[0].posteriors).toBe(evidence[0].posteriors);
    expect(enriched[0].phonemes.map(({ seamInsertions: _, ...p }) => p)).toEqual(evidence[0].phonemes);
    expect(evidence).toEqual(original);
    const cells = insertionCells(JSON.parse(JSON.stringify(enriched)));
    expect(cells.map(e => [e.posteriors!.startFrame, e.posteriors!.nFrames])).toEqual([[0,1],[1,3],[4,1],[5,1],[6,1]]);
    expect(cells.flatMap(e => e.phonemes)[0].seamInsertions).toEqual(metadata);
    expect(cells.flatMap(e => Array.from(e.posteriors!.ids))).toEqual(evidence[0].posteriors!.ids);
  });

  it("requires primary posteriors and a seam before consulting the teacher", async () => {
    const decode = vi.fn();
    const model = { manifest: { sampleRate: 16000 }, insertionTeacher: { decode } } as unknown as LoadedModel;
    expect(await addInsertionEvidence(model, new Float32Array(16000), [], index)).toEqual([]);
    const noRows = evidence.map(({ posteriors: _, ...e }) => e);
    expect(await addInsertionEvidence(model, new Float32Array(16000), noRows, index)).toBe(noRows);
    expect(decode).not.toHaveBeenCalled();
  });
});
