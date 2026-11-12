import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { Fbank } from "../../src/runtime/fbank.js";
import { decodeWav } from "../../src/runtime/wav.js";

const fixture = (p: string) => new URL(`../fixtures/${p}`, import.meta.url);
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
});
