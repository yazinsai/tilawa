import { describe, expect, it } from "vitest";
import { ZipformerRunner, type TensorLike, type ZipformerIo } from "../../src/recitation/zipformerRunner";
import {
  A0W_SLIP_HEAD,
  ENCODER_DIM,
  EncoderFrames,
  poolEncoderSpan,
  slipProbability,
  wordSpansFromTrail,
  type SlipHead,
  type TrailLike,
} from "../../src/recitation/slipHead";

class FakeTensor implements TensorLike {
  constructor(
    readonly type: string,
    readonly data: Float32Array | BigInt64Array | Int32Array,
    readonly dims: readonly number[],
  ) {}
}

function tinyIo(): ZipformerIo {
  return {
    T: 4,
    hop: 2,
    featureDim: 2,
    vocabSize: 3,
    encoderFrames: "encoder_out",
    inputs: [
      { name: "x", dims: [1, 4, 2], dtype: "float32" },
      { name: "processed_lens", dims: [1], dtype: "int64" },
    ],
  };
}

describe("slip head", () => {
  it("splits a skipped gap the way the probe does", () => {
    // words 0 (frames 0, 4) and 3 (frames 10, 12). Cells 1 and 5 map to those words.
    const local = [0, 0, 0, 0, 3];
    const tracker: TrailLike = {
      heard: [{ frame: 0 }, { frame: 4 }, { frame: 10 }, { frame: 12 }],
      trail: [1, 1, 5, 5],
      firstWord: 0,
      len: 8,
      localWordOfPos: local,
      corpus: {
        wordSurah: [2, 2, 2, 2],
        wordAyah: [5, 5, 5, 5],
        wordInAyah: [0, 1, 2, 3],
      },
    };
    const by = new Map(wordSpansFromTrail(tracker).map((s) => [s.word, s]));
    expect([...by.keys()].sort()).toEqual([0, 1, 2, 3]);
    expect(by.get(0)).toMatchObject({ a: 0, b: 5, skipped: false });
    expect(by.get(1)).toMatchObject({ a: 5, b: 7, skipped: true });
    expect(by.get(2)).toMatchObject({ a: 7, b: 10, skipped: true });
    expect(by.get(3)).toMatchObject({ a: 10, b: 13, skipped: false });
    expect(by.get(1)!.b).toBeLessThanOrEqual(by.get(2)!.a);
  });

  it("pools mean and max with a one-frame pad, then a logistic", () => {
    const enc = new EncoderFrames(4, 16);
    const rows = new Float32Array([
      0, 0, 0, 0,
      1, 2, 0, 0,
      3, 0, 0, 0,
      1, 4, 0, 0,
      0, 0, 0, 0,
    ]);
    enc.push(rows, 5, 0);
    // span [1, 4) pads to [0, 5)
    const pooled = poolEncoderSpan(enc, 1, 4, 1)!;
    expect(pooled.mean[0]).toBeCloseTo(1, 5);
    expect(pooled.mean[1]).toBeCloseTo(1.2, 5);
    expect(pooled.max[0]).toBe(3);
    expect(pooled.max[1]).toBe(4);
    expect(poolEncoderSpan(enc, 0, 1, 1)).not.toBeNull();
    // A span the ring has dropped is not scored.
    const short = new EncoderFrames(4, 2);
    short.push(rows, 5, 0);
    expect(short.has(0, 1)).toBe(false);
    expect(poolEncoderSpan(short, 0, 1, 1)).toBeNull();

    const n = 8;
    const head: SlipHead = {
      mu: new Float32Array(n),
      sd: new Float32Array(n).fill(1),
      coef: new Float32Array(n),
      intercept: 0,
      strict: 0.8,
      high: 0.5,
    };
    head.coef[0] = 1;
    head.coef[4] = 1;
    const mean = new Float32Array(4);
    const max = new Float32Array(4);
    mean[0] = 1;
    max[0] = 1;
    const p = slipProbability(head, mean, max);
    expect(p).toBeCloseTo(1 / (1 + Math.exp(-2)), 10);
  });

  it("ships a 1024-d a0w head with a stricter cutoff than the high one", () => {
    expect(A0W_SLIP_HEAD.mu.length).toBe(ENCODER_DIM * 2);
    expect(A0W_SLIP_HEAD.coef.length).toBe(ENCODER_DIM * 2);
    expect(A0W_SLIP_HEAD.strict).toBeGreaterThanOrEqual(A0W_SLIP_HEAD.high);
    expect(A0W_SLIP_HEAD.strict).toBeGreaterThan(0);
    expect(A0W_SLIP_HEAD.high).toBeLessThanOrEqual(1);
  });

  it("collects encoder frames and ignores a model that has none", async () => {
    const io = tinyIo();
    const encDim = 4;
    const make = (withEnc: boolean) => ({
      async run(feeds: Record<string, TensorLike>) {
        const out: Record<string, TensorLike> = {
          log_probs: new FakeTensor("float32", Float32Array.from([1, 2, 3, 4, 5, 6]), [1, 2, 3]),
          new_processed_lens: feeds.processed_lens!,
        };
        if (withEnc) {
          out.encoder_out = new FakeTensor("float32", Float32Array.from([9, 8, 7, 6, 5, 4, 3, 2]), [1, 2, encDim]);
        }
        return out;
      },
    });
    const frames = () => [0, 1, 2, 3].map(() => new Float32Array(2));
    const on = ZipformerRunner.fromSession(make(true), io, FakeTensor);
    const hit = await on.accept(frames());
    expect([...hit.logProbs]).toEqual([1, 2, 3, 4, 5, 6]);
    expect(hit.encoderFrames).toBe(2);
    expect([...(hit.encoder ?? [])]).toEqual([9, 8, 7, 6, 5, 4, 3, 2]);

    const off = ZipformerRunner.fromSession(make(false), io, FakeTensor);
    const miss = await off.accept(frames());
    expect([...miss.logProbs]).toEqual([...hit.logProbs]);
    expect(miss.encoder).toBeUndefined();
    expect(miss.frames).toBe(hit.frames);
  });
});
