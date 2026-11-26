import { beforeEach, expect, it, vi } from "vitest";

let tones = 0;
class FakeContext {
  state = "running";
  currentTime = 0;
  destination = {};
  resume = () => Promise.resolve();
  createOscillator() {
    tones++;
    const param = { setValueAtTime() {}, exponentialRampToValueAtTime() {} };
    return { frequency: param, connect: (n: unknown) => n, start() {}, stop() {} };
  }
  createGain() {
    const param = { setValueAtTime() {}, exponentialRampToValueAtTime() {} };
    return { gain: param, connect: (n: unknown) => n };
  }
}

beforeEach(() => {
  tones = 0;
  const store = new Map<string, string>();
  vi.stubGlobal("AudioContext", FakeContext);
  vi.stubGlobal("localStorage", { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => void store.set(k, v) });
  vi.resetModules();
});

it("plays one tone per burst of flags, and none before a tap arms it", async () => {
  const { armCue, cue } = await import("../src/lib/cue");
  cue(0);
  expect(tones).toBe(0);
  armCue();
  cue(1000);
  cue(1100); // same burst
  cue(1500);
  expect(tones).toBe(2);
});

it("stays silent when switched off, and remembers the choice", async () => {
  const first = await import("../src/lib/cue");
  first.setCueEnabled(false);
  first.armCue();
  first.cue(1000);
  expect(tones).toBe(0);
  vi.resetModules();
  expect((await import("../src/lib/cue")).cueEnabled()).toBe(false);
});
