import { afterEach, describe, expect, it, vi } from "vitest";
import { Fbank } from "../src/runtime/fbank.js";
import type { EngineEvent } from "../src/index.js";
import { tinyTilawa, wav } from "./helpers.js";

const tilawa = await tinyTilawa({ recordEvidence: true });
const clip = await wav();

const concat = (parts: Float32Array[]) => {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) out.set(p, o), (o += p.length);
  return out;
};
const upsample = (x: Float32Array, from: number, to: number) => {
  const up = new Float32Array(Math.floor((x.length * to) / from));
  for (let i = 0; i < up.length; i++) up[i] = x[Math.min(x.length - 1, Math.round((i * from) / to))];
  return up;
};

async function take(samples: Float32Array, rate: number, chunk: number, tap: boolean) {
  const s = tilawa.startFreeRecitation();
  const events: EngineEvent[] = [];
  s.subscribe((e) => events.push(e));
  const tapped: Float32Array[] = [];
  const starts: number[] = [];
  if (tap) s.onAudio((pcm, startMs) => (tapped.push(pcm), starts.push(startMs)));
  // What the listener feeds its fbank: exactly the 16 kHz samples it received.
  const heard: Float32Array[] = [];
  const push = Fbank.prototype.push;
  const spy = vi.spyOn(Fbank.prototype, "push").mockImplementation(function (this: Fbank, x: Float32Array) {
    heard.push(Float32Array.from(x));
    return push.call(this, x);
  });
  const calls = [];
  for (let a = 0; a < samples.length; a += chunk) calls.push(s.pushAudio(samples.subarray(a, a + chunk), rate));
  calls.push(s.end());
  const returned = (await Promise.all(calls)).flat();
  spy.mockRestore();
  return { events, returned, tapped, starts, heard: concat(heard), evidence: s.evidence(), now: s.now() };
}

afterEach(() => vi.restoreAllMocks());

describe("session.onAudio tap", () => {
  for (const [label, rate, chunk] of [["16 kHz", 16000, 3000], ["44.1 kHz", 44100, 4410], ["8 kHz", 8000, 1234]] as const) {
    it(`at ${label}: fires exactly the samples the listener got, contiguously, and changes no event`, async () => {
      const input = rate === 16000 ? clip.samples : upsample(clip.samples, 16000, rate);
      const off = await take(input, rate, chunk, false);
      const on = await take(input, rate, chunk, true);
      expect(on.events).toEqual(off.events);
      expect(on.returned).toEqual(off.returned);
      expect(on.evidence).toEqual(off.evidence);
      const tapped = concat(on.tapped);
      expect(tapped.length).toBe(on.heard.length);
      expect(Array.from(tapped)).toEqual(Array.from(on.heard));
      // startMs is the session clock: each chunk starts where the previous one ended.
      let at = 0;
      on.tapped.forEach((p, i) => {
        expect(on.starts[i]).toBeCloseTo(at, 9);
        at += p.length / 16;
      });
      expect(on.now).toBeCloseTo(at, 9);
    });
  }

  it("hands each tap a copy, so a tap that mutates it can't change inference", async () => {
    const s = tilawa.startFreeRecitation();
    const base = await take(clip.samples, 16000, 4000, false);
    s.onAudio((pcm) => pcm.fill(0));
    const events: EngineEvent[] = [];
    s.subscribe((e) => events.push(e));
    for (let a = 0; a < clip.samples.length; a += 4000) await s.pushAudio(clip.samples.subarray(a, a + 4000), 16000);
    await s.end();
    expect(events).toEqual(base.events);
  });

  it("stops after unsubscribe and after dispose", async () => {
    const s = tilawa.startFreeRecitation();
    let n = 0;
    const off = s.onAudio(() => n++);
    await s.pushAudio(clip.samples.subarray(0, 1600), 16000);
    off();
    await s.pushAudio(clip.samples.subarray(0, 1600), 16000);
    expect(n).toBe(1);
    s.onAudio(() => n++);
    s.dispose();
    expect(s.onAudio(() => n++)).toBeTypeOf("function");
    expect(n).toBe(1);
  });
});
