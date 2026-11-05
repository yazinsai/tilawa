import { describe, expect, it } from "vitest";
import { PitchTrack } from "../src/pitch.js";
const tone = (hz: number, seconds: number) => Float32Array.from({ length: seconds * 16000 }, (_, i) => 0.2 * Math.sin(2 * Math.PI * hz * i / 16000));
function track(audio: Float32Array, step: number) {
  const pitch = new PitchTrack(16000, 40);
  for (let i = 0; i < audio.length; i += step) pitch.push(audio.subarray(i, i + step));
  pitch.end();
  return pitch.slice(0, Infinity);
}
describe("raw audio pitch", () => {
  it("recovers a harmonic contour independently of push boundaries", () => {
    const audio = tone(160, 1);
    const a = track(audio, audio.length);
    expect(track(audio, 317)).toEqual(a);
    expect(a).toHaveLength(25);
    for (const hz of a.slice(0, -1)) expect(hz).toBeCloseTo(160, 0);
  });
  it("keeps internal silence unvoiced between pitch contours", () => {
    const audio = new Float32Array(16000);
    audio.set(tone(140, 0.3));
    audio.set(tone(210, 0.3), 11200);
    const hz = track(audio, 320);
    expect(hz.slice(9, 15)).toEqual(Array(6).fill(0));
    expect(hz[2]).toBeCloseTo(140, 0);
    expect(hz[20]).toBeCloseTo(210, 0);
  });
});
