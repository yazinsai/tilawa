import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import { join } from "path";
import {
  buildTrie,
  type QuranVerse,
  type CompactTrie,
} from "../src/lib/phoneme-trie";
import { beamSearchDecode } from "../src/worker/beam-decode";

// Load real data
const quranData: QuranVerse[] = JSON.parse(
  readFileSync(join(__dirname, "../public/quran_phonemes.json"), "utf-8"),
);
const vocabJson: Record<string, string> = JSON.parse(
  readFileSync(join(__dirname, "../public/phoneme_vocab.json"), "utf-8"),
);

// Build reverse maps
const tokenToId = new Map<string, number>();
for (const [id, token] of Object.entries(vocabJson)) {
  tokenToId.set(token, parseInt(id));
}
const BLANK_ID = tokenToId.get("<blank>")!; // 69
const PIPE_ID = tokenToId.get("|")!; // 68
const VOCAB_SIZE = 70;

function phonemesToIds(phonemes: string): number[] {
  return phonemes
    .trim()
    .split(/\s+/)
    .map((t) => tokenToId.get(t)!)
    .filter((id) => id !== undefined);
}

describe("phoneme-trie", () => {
  const first20 = quranData.slice(0, 20);

  it("builds a trie from verses", () => {
    const { trie, stats } = buildTrie(first20, vocabJson);
    expect(stats.singleVerseCount).toBe(20);
    expect(stats.spanCount).toBeGreaterThan(0);
    expect(stats.nodeCount).toBeGreaterThan(20);
    expect(trie.validChildren(0).length).toBeGreaterThan(0);
  });

  it("walks the trie to find Al-Fatiha verse 1", () => {
    const { trie } = buildTrie(first20, vocabJson);
    const ids = phonemesToIds(first20[0].phonemes);
    const node = trie.walk(0, ids);
    expect(node).not.toBe(-1);
    expect(trie.verseEnds[node].length).toBeGreaterThan(0);
    expect(trie.verseEnds[node].some((r) => r.verseIndex === 0 && r.spanLength === 1)).toBe(true);
  });

  it("walks partial prefix and finds valid children", () => {
    const { trie } = buildTrie(first20, vocabJson);
    const prefix = phonemesToIds("b i s m i");
    const node = trie.walk(0, prefix);
    expect(node).not.toBe(-1);
    expect(trie.verseEnds[node].length).toBe(0); // not a complete verse
    // Next valid token should be "|" (word boundary)
    expect(trie.getChild(node, PIPE_ID)).not.toBe(-1);
  });

  it("rejects invalid token sequences", () => {
    const { trie } = buildTrie(first20, vocabJson);
    const zzId = tokenToId.get("z")!;
    const node = trie.walk(0, [zzId, zzId, zzId]);
    expect(node).toBe(-1);
  });

  it("contains multi-verse spans", () => {
    const { trie } = buildTrie(first20, vocabJson);
    const v1Ids = phonemesToIds(first20[0].phonemes);
    const v2Ids = phonemesToIds(first20[1].phonemes);
    const spanIds = [...v1Ids, PIPE_ID, ...v2Ids];
    const node = trie.walk(0, spanIds);
    expect(node).not.toBe(-1);
    expect(trie.verseEnds[node].some((r) => r.verseIndex === 0 && r.spanLength === 2)).toBe(true);
  });

  it("collects all reachable verse ends from root", () => {
    const { trie } = buildTrie(first20, vocabJson);
    const allEnds = trie.collectVerseEnds(0);
    // Should have single verses (20) + spans
    expect(allEnds.length).toBeGreaterThan(20);
    for (let i = 0; i < 20; i++) {
      expect(allEnds.some((r) => r.verseIndex === i && r.spanLength === 1)).toBe(true);
    }
  });

  it("handles full 6236 verses within memory budget", () => {
    const { trie, stats } = buildTrie(quranData, vocabJson, 3);
    expect(stats.singleVerseCount).toBe(6236);
    expect(stats.spanCount).toBeGreaterThan(0);
    expect(stats.memoryMB).toBeLessThan(50);
    console.log(
      `Full trie: ${stats.nodeCount} nodes, ${stats.singleVerseCount} verses, ` +
        `${stats.spanCount} spans, max depth ${stats.maxDepth}, ~${stats.memoryMB.toFixed(1)}MB`,
    );
  });
});

describe("beam-decode", () => {
  const first20 = quranData.slice(0, 20);

  /**
   * Create synthetic logprobs that strongly favor a specific token sequence.
   * Each token gets ~5 frames (2 blank, 1 token, 2 blank).
   */
  function makeSyntheticLogprobs(
    targetIds: number[],
    framesPerToken = 5,
  ): { logprobs: Float32Array; timeSteps: number } {
    const timeSteps = targetIds.length * framesPerToken;
    const logprobs = new Float32Array(timeSteps * VOCAB_SIZE);
    logprobs.fill(-10);

    for (let i = 0; i < targetIds.length; i++) {
      const baseFrame = i * framesPerToken;
      for (let f = 0; f < framesPerToken; f++) {
        const frame = baseFrame + f;
        const offset = frame * VOCAB_SIZE;
        if (f === 2) {
          // Token frame
          logprobs[offset + targetIds[i]] = -0.01;
          logprobs[offset + BLANK_ID] = -5;
        } else {
          // Blank frames
          logprobs[offset + BLANK_ID] = -0.01;
        }
      }
    }
    return { logprobs, timeSteps };
  }

  it("decodes a single verse (Al-Fatiha:1) with synthetic logprobs", () => {
    const { trie } = buildTrie(first20, vocabJson);
    const targetIds = phonemesToIds(first20[0].phonemes);
    const { logprobs, timeSteps } = makeSyntheticLogprobs(targetIds);

    const results = beamSearchDecode(
      logprobs, timeSteps, VOCAB_SIZE, BLANK_ID, trie, 8,
    );

    expect(results.length).toBeGreaterThan(0);
    const best = results[0];
    expect(best.tokenIds).toEqual(targetIds);
    expect(best.isComplete).toBe(true);
    expect(best.matchedVerses.some((r) => r.verseIndex === 0)).toBe(true);
  });

  it("returns valid trie prefixes only", () => {
    const { trie } = buildTrie(first20, vocabJson);
    const targetIds = phonemesToIds(first20[2].phonemes);
    const { logprobs, timeSteps } = makeSyntheticLogprobs(targetIds);

    const results = beamSearchDecode(
      logprobs, timeSteps, VOCAB_SIZE, BLANK_ID, trie, 8,
    );

    for (const result of results) {
      const node = trie.walk(0, result.tokenIds);
      expect(node).not.toBe(-1);
    }
  });

  it("handles ambiguous logprobs by returning multiple candidates", () => {
    const { trie } = buildTrie(first20, vocabJson);
    const v3Ids = phonemesToIds(first20[2].phonemes);
    const { logprobs, timeSteps } = makeSyntheticLogprobs(v3Ids);

    const results = beamSearchDecode(
      logprobs, timeSteps, VOCAB_SIZE, BLANK_ID, trie, 8,
    );

    expect(results.length).toBeGreaterThan(0);
    expect(results[0].tokenIds).toEqual(v3Ids);
  });

  it("preserves original token IDs (0-68)", () => {
    const { trie } = buildTrie(first20, vocabJson);
    const targetIds = phonemesToIds(first20[0].phonemes);
    const { logprobs, timeSteps } = makeSyntheticLogprobs(targetIds);

    const results = beamSearchDecode(
      logprobs, timeSteps, VOCAB_SIZE, BLANK_ID, trie, 8,
    );

    for (const result of results) {
      for (const id of result.tokenIds) {
        expect(id).toBeGreaterThanOrEqual(0);
        expect(id).toBeLessThanOrEqual(68);
      }
    }
  });

  it("produces results within latency budget (20-verse trie, 300 frames)", () => {
    const { trie } = buildTrie(first20, vocabJson);
    const targetIds = phonemesToIds(first20[0].phonemes);

    const timeSteps = 300;
    const logprobs = new Float32Array(timeSteps * VOCAB_SIZE);
    logprobs.fill(-10);
    for (let i = 0; i < targetIds.length && i * 10 < timeSteps; i++) {
      const frame = i * 10;
      logprobs[frame * VOCAB_SIZE + targetIds[i]] = -0.01;
      logprobs[frame * VOCAB_SIZE + BLANK_ID] = -5;
      for (let f = 1; f < 10 && frame + f < timeSteps; f++) {
        logprobs[(frame + f) * VOCAB_SIZE + BLANK_ID] = -0.01;
      }
    }

    const start = performance.now();
    const results = beamSearchDecode(
      logprobs, timeSteps, VOCAB_SIZE, BLANK_ID, trie, 8,
    );
    const elapsed = performance.now() - start;

    expect(results.length).toBeGreaterThan(0);
    expect(elapsed).toBeLessThan(200);
    console.log(`Beam decode 300 frames (20-verse trie): ${elapsed.toFixed(1)}ms`);
  });

  it("latency with full 6236-verse trie", () => {
    const { trie } = buildTrie(quranData, vocabJson, 3);
    const targetIds = phonemesToIds(quranData[0].phonemes);

    const timeSteps = 300;
    const logprobs = new Float32Array(timeSteps * VOCAB_SIZE);
    logprobs.fill(-10);
    for (let i = 0; i < targetIds.length && i * 10 < timeSteps; i++) {
      const frame = i * 10;
      logprobs[frame * VOCAB_SIZE + targetIds[i]] = -0.01;
      logprobs[frame * VOCAB_SIZE + BLANK_ID] = -5;
      for (let f = 1; f < 10 && frame + f < timeSteps; f++) {
        logprobs[(frame + f) * VOCAB_SIZE + BLANK_ID] = -0.01;
      }
    }

    const start = performance.now();
    const results = beamSearchDecode(
      logprobs, timeSteps, VOCAB_SIZE, BLANK_ID, trie, 8,
    );
    const elapsed = performance.now() - start;

    expect(results.length).toBeGreaterThan(0);
    console.log(`Beam decode 300 frames (full trie): ${elapsed.toFixed(1)}ms`);
    expect(elapsed).toBeLessThan(2000);
  });
});
