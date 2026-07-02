import { describe, it, expect } from "vitest";
import { createTilawaSession, type SessionRunner, type TilawaAssets } from "../src/index";
import type { SessionOutput } from "../src/session";

/**
 * Deterministic decode+match test with NO ONNX.
 *
 * Strategy: the SDK is self-consistent — the same TextCTCDecoder that decodes
 * runtime logprobs also builds each verse's `phonemes_joined` from
 * `quranCtcTokens`. So if a mock runner emits one-hot logprobs that greedy-decode
 * to the exact token id sequence stored for a verse, the decoded text equals that
 * verse's `phonemes_joined` → Levenshtein ratio 1.0 → trusted champion.
 *
 * Vocab tokens are Latin (normalizeArabic leaves ASCII alone) with `\u2581`
 * word-prefix separators, so the mapping stays legible.
 */

const WORD_PREFIX = "\u2581";

// id -> token. blank id = 10.
const vocab: Record<string, string> = {
  "0": WORD_PREFIX + "bis",
  "1": "mi",
  "2": WORD_PREFIX + "lah",
  "3": WORD_PREFIX + "rah",
  "4": "man",
  "5": WORD_PREFIX + "qul",
  "6": "hu",
  "7": WORD_PREFIX + "wa",
  "8": "l",
  "9": "hu2",
  "10": "<blank>",
};
const BLANK_ID = 10;
const VOCAB_SIZE = 11;

// verse "1:1:1" decodes tokens [0,1,2,3] -> "bis mi lah rah" (WORD_PREFIX -> space)
// verse "112:1:1" decodes tokens [5,6,7,8] -> "qul hu wa l"
const quranCtcTokens = {
  "1:1:1": [0, 1, 2, 3],
  "112:1:1": [5, 6, 7, 8],
};

// Raw verse records — only the fields the adapter reads.
const quran = [
  {
    surah: 1,
    ayah: 1,
    text_uthmani: "بسم الله الرحمن",
    surah_name: "الفاتحة",
    surah_name_en: "Al-Fatihah",
  },
  {
    surah: 112,
    ayah: 1,
    text_uthmani: "قل هو الله",
    surah_name: "الإخلاص",
    surah_name_en: "Al-Ikhlas",
  },
];

const assets: TilawaAssets = { vocab, quranCtcTokens, quran, blankId: BLANK_ID };

/**
 * Build a logprob buffer whose greedy CTC decode yields exactly `tokenIds`.
 * Consecutive equal ids get a blank frame between them so collapse keeps both.
 * Each emitted frame is one-hot (0 for target, -1e9 elsewhere).
 */
function logprobsFor(tokenIds: number[]): SessionOutput {
  const frames: number[] = [];
  let prev = -1;
  for (const id of tokenIds) {
    if (id === prev) frames.push(BLANK_ID); // separator so collapse doesn't merge
    frames.push(id);
    prev = id;
  }
  const timeSteps = frames.length;
  const logprobs = new Float32Array(timeSteps * VOCAB_SIZE).fill(-1e9);
  frames.forEach((id, t) => {
    logprobs[t * VOCAB_SIZE + id] = 0;
  });
  return { logprobs, timeSteps, vocabSize: VOCAB_SIZE };
}

function runnerFor(tokenIds: number[]): SessionRunner {
  const out = logprobsFor(tokenIds);
  return { run: async () => out };
}

describe("createTilawaSession decode+match (no ONNX)", () => {
  it("decodes canned logprobs to the exact verse and matches surah:ayah with score 1.0", async () => {
    const session = createTilawaSession(runnerFor(quranCtcTokens["1:1:1"]), assets);
    const pred = await session.transcribe(new Float32Array(16000));

    expect(pred.transcript).toBe("bismi lah rah");
    expect(pred.surah).toBe(1);
    expect(pred.ayah).toBe(1);
    expect(pred.score).toBeGreaterThanOrEqual(0.8);
    expect(pred.score).toBeCloseTo(1.0, 5);
  });

  it("resolves a different verse from a different logprob sequence", async () => {
    const session = createTilawaSession(runnerFor(quranCtcTokens["112:1:1"]), assets);
    const pred = await session.transcribe(new Float32Array(16000));

    expect(pred.transcript).toBe("qulhu wal");
    expect(pred.surah).toBe(112);
    expect(pred.ayah).toBe(1);
    expect(pred.score).toBeCloseTo(1.0, 5);
  });

  it("collapses CTC repeats via blank separators", () => {
    // tokenIds with a repeat -> logprobsFor inserts a blank so both survive
    const out = logprobsFor([4, 4]);
    // frames: [4, blank, 4] -> 3 steps
    expect(out.timeSteps).toBe(3);
  });

  it("returns surah=0 when the decode matches nothing in the db", async () => {
    // token 9 ("hu2") never appears in any verse's ctc tokens
    const session = createTilawaSession(runnerFor([9, 9, 9, 9, 9, 9, 9, 9]), assets);
    const pred = await session.transcribe(new Float32Array(16000));

    expect(pred.surah).toBe(0);
    expect(pred.ayah).toBe(0);
    expect(pred.score).toBe(0);
  });
});
