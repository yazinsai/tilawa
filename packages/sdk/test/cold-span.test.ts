import { describe, expect, it, vi } from "vitest";
import type { LoadedModel } from "../src/model.js";
import { decodeColdSpan } from "../src/gap-decode.js";

const mock = vi.hoisted(() => ({ listeners: [] as any[], phones: [] as any[] }));
vi.mock("../src/listener.js", () => ({
  createListener: () => {
    const listener = {
      push: vi.fn(async () => null),
      end: vi.fn(async () => ({ inventory: "test", frameMs: 40, phonemes: mock.phones })),
    };
    mock.listeners.push(listener);
    return listener;
  },
}));
const model = {} as LoadedModel;
const audio = new Float32Array(3000).map((_, i) => i);

describe("cold decoding of an already selected spelled span", () => {
  it("uses 300 ms edge context and only keeps centres inside the original span", async () => {
    // The crop starts at 0.7 s, mapped to frame 18 on the 40 ms clock.
    mock.phones = [
      { p: "ل", start: 1, end: 3, conf: 0.9 },
      { p: "و", start: 6, end: 8, conf: 0.9 },
      { p: "م", start: 10, end: 10, conf: 0.9 },
      { p: "ن", start: 12, end: 12, conf: 0.9 },
    ];
    const result = await decodeColdSpan(model, audio, 1000, 25, 30, 40, 4);
    expect(result.map(p => [p.p, p.start, p.end])).toEqual([["و", 24, 26], ["م", 28, 28]]);
    const pushed = mock.listeners.at(-1).push.mock.calls.flatMap((c: any[]) => Array.from(c[0]));
    expect(pushed).toEqual(Array.from(audio.slice(700, 1500)));
    expect(mock.listeners.at(-1).end).toHaveBeenCalledOnce();
  });

  it("starts fresh for each crop, clips padding at the recording edge, and keeps blank crops blank", async () => {
    mock.phones = [];
    const before = mock.listeners.length;
    expect(await decodeColdSpan(model, audio, 1000, 0, 5, 40, 4)).toEqual([]);
    expect(await decodeColdSpan(model, audio, 1000, 0, 5, 40, 4)).toEqual([]);
    expect(mock.listeners.length - before).toBe(2);
    const pushed = mock.listeners.at(-1).push.mock.calls.flatMap((c: any[]) => Array.from(c[0]));
    expect(pushed).toEqual(Array.from(audio.slice(0, 500)));
  });

  it("fails closed for missing or invalid audio bounds", async () => {
    const before = mock.listeners.length;
    for (const [from, to] of [[-1, 5], [5, 5], [10, 100]]) {
      expect(await decodeColdSpan(model, audio, 1000, from, to, 40, 4)).toEqual([]);
    }
    expect(mock.listeners).toHaveLength(before);
  });
});
