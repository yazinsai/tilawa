import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { Fbank, type FrequencyPooling } from "../src/fbank.js";
import { decodeWav } from "../src/wav.js";

const fixture = (p: string) => new URL(`./fixtures/${p}`, import.meta.url);
const reference = JSON.parse(readFileSync(fixture("tiny/reference.json"), "utf8")) as { fbank: number[]; fbankFrames: number };
const wav = decodeWav(readFileSync(fixture("tlog-clean-00010-630.wav")));

describe("fbank", () => {
  it("matches torchaudio's kaldi fbank on a recitation clip", () => {
    const frames = new Fbank().push(wav.samples);
    expect(frames.length).toBe(reference.fbankFrames);
    let worst = 0;
    for (let t = 0; t < 40; t++) for (let m = 0; m < 80; m++) worst = Math.max(worst, Math.abs(frames[t][m] - reference.fbank[t * 80 + m]));
    expect(worst).toBeLessThan(1e-2);
  });

  it("gives the same frames however the audio is split", () => {
    const whole = new Fbank().push(wav.samples);
    const f = new Fbank();
    const parts: Float32Array[] = [];
    for (let a = 0; a < wav.samples.length; a += 1234) parts.push(...f.push(wav.samples.subarray(a, a + 1234)));
    expect(parts.length).toBe(whole.length);
    expect(Array.from(parts[100])).toEqual(Array.from(whole[100]));
  });

  it("applies exported power filters before the log and keeps chunk timing", () => {
    // All outputs read one FFT bin, with known relative energy scales.
    const pooling: FrequencyPooling = { schema: 1, fftSize: 512,
      filters: Array.from({ length: 80 }, (_, m) => ({ start: 20, weights: [1], scale: m + 1 })) };
    const pcm = Float32Array.from({ length: 16000 }, (_, i) => 0.1 * Math.sin(2 * Math.PI * 625 * i / 16000));
    const whole = new Fbank({ frequencyPooling: pooling }).push(pcm);
    const fbank = new Fbank({ frequencyPooling: pooling });
    pooling.filters[0].scale = 100; // A stream owns its filters.
    const parts = [...fbank.push(pcm.subarray(0, 333)), ...fbank.push(pcm.subarray(333))];
    expect(parts).toEqual(whole);
    expect(whole.length).toBe(new Fbank().push(pcm).length);
    for (let m = 1; m < 80; m++) expect(whole[0][m] - whole[0][0]).toBeCloseTo(Math.log(m + 1), 5);
    const silence = new Fbank({ frequencyPooling: pooling }).push(new Float32Array(400));
    expect(new Set(silence[0]).size).toBe(1);
  });

  it("rejects incompatible, negative, noncontiguous or unnormalized pooling", () => {
    const base: FrequencyPooling = { schema: 1, fftSize: 512,
      filters: Array.from({ length: 80 }, () => ({ start: 10, weights: [0.5, 0.5], scale: 1 })) };
    for (const filter of [
      { start: 256, weights: [1], scale: 1 }, { start: 1.5, weights: [1], scale: 1 },
      { start: 10, weights: [1, 0, 1], scale: 1 }, { start: 10, weights: [-1, 2], scale: 1 },
      { start: 10, weights: [0.5], scale: 1 }, { start: 10, weights: [1], scale: NaN },
    ]) expect(() => new Fbank({ frequencyPooling: { ...base, filters: [filter, ...base.filters.slice(1)] } })).toThrow();
    expect(() => new Fbank({ frequencyPooling: { ...base, filters: [] } })).toThrow();
    expect(() => new Fbank({ frequencyPooling: base, highFrequency: 8000 })).toThrow();
  });
});
