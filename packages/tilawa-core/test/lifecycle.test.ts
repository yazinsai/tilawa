import { describe, expect, it, vi } from "vitest";
import { SDK_VERSION, SessionEndedError } from "../src/index.js";
import type { EngineEvent } from "../src/index.js";
import { loadNodeModel } from "../src/node.js";
import { tinyDir, tinyTilawa, wav } from "./helpers.js";

const tilawa = await tinyTilawa();
const clip = await wav();
const second = clip.samples.subarray(0, 16000);

describe("session lifecycle (contract §3)", () => {
  it("has a uuid v4 id, a wall clock at the first sample and a session clock", async () => {
    const s = tilawa.startFreeRecitation();
    expect(s.id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    expect(s.id).not.toBe(tilawa.startFreeRecitation().id);
    expect(s.startedAt).toBeNull();
    expect(s.now()).toBe(0);
    const before = Date.now();
    await s.pushAudio(second, 16000);
    expect(Date.parse(s.startedAt!)).toBeGreaterThanOrEqual(before - 1);
    expect(s.now()).toBe(1000);
    // At 8 kHz, half the input samples' duration in 16 kHz samples.
    await s.pushAudio(new Float32Array(8000), 8000);
    expect(s.now()).toBeGreaterThan(1900);
    expect(s.now()).toBeLessThanOrEqual(2000);
    await s.end();
  });

  it("goes active → ending → ended; end() is idempotent and later pushes reject", async () => {
    const s = tilawa.startFreeRecitation();
    expect(s.state).toBe("active");
    const p = s.pushAudio(second, 16000);
    const e1 = s.end();
    expect(s.state).toBe("ending");
    expect(s.end()).toBe(e1);
    await expect(s.pushAudio(second, 16000)).rejects.toBeInstanceOf(SessionEndedError);
    await p;
    await e1;
    expect(s.state).toBe("ended");
    expect(s.end()).toBe(e1);
    expect(() => s.pushEvidence({ inventory: "x", frameMs: 40, phonemes: [] })).toThrow(SessionEndedError);
  });

  it("runs pushes and end in call order without awaits, and returns the same events subscribers get", async () => {
    const s = tilawa.startMemorization({ surah: 1, fromAyah: 1, toAyah: 7 });
    const seen: EngineEvent[] = [];
    s.subscribe((e) => seen.push(e));
    const calls = [];
    for (let a = 0; a < clip.samples.length; a += 3000) calls.push(s.pushAudio(clip.samples.subarray(a, a + 3000), clip.sampleRate));
    calls.push(s.end());
    const returned = (await Promise.all(calls)).flat();
    expect(seen[0]).toEqual(s.capabilities);
    expect(seen.slice(1)).toEqual(returned);
  });

  it("delivers capabilities first, with the V1 fields", () => {
    const s = tilawa.startFreeRecitation();
    const seen: EngineEvent[] = [];
    s.subscribe((e) => seen.push(e));
    expect(seen).toHaveLength(1);
    expect(seen[0]).toMatchObject({
      type: "capabilities", v: 1, modelVersion: "tiny-random", referenceVersion: "ref-v2", sdkVersion: SDK_VERSION, backend: "onnxruntime-node",
    });
    expect(tilawa.capabilities).toEqual(seen[0]);
    s.dispose();
  });

  it("resolves the policy with every default filled in", () => {
    expect(tilawa.startFreeRecitation().policy).toEqual({
      countSelfCorrections: false, confirm: { after: "delay", ms: 2500 }, minConfidence: 0.6, evidenceMs: 40,
    });
    const s = tilawa.startMemorization({ surah: 18, fromAyah: 60, toAyah: 70 }, { confirm: { after: "next-word" } });
    expect(s.policy.confirm).toEqual({ after: "next-word" });
    expect(s.mode).toBe("memorization");
    expect(s.passage).toMatchObject({ surah: 18, fromAyah: 60, toAyah: 70, from: { surah: 18, ayah: 60, word: 1 } });
    expect(tilawa.startFreeRecitation().passage).toBeNull();
  });

  it("dispose() drops subscribers and rejects queued pushes", async () => {
    const s = tilawa.startFreeRecitation();
    const seen: EngineEvent[] = [];
    s.subscribe((e) => seen.push(e));
    const running = s.pushAudio(second, 16000);
    const queued = [s.pushAudio(second, 16000), s.pushAudio(second, 16000), s.end()];
    let ended = 0;
    s.onEnded(() => ended++);
    s.dispose();
    expect(s.state).toBe("disposed");
    expect(ended).toBe(1);
    for (const q of queued) await expect(q).rejects.toBeInstanceOf(SessionEndedError);
    await running;
    await s.idle();
    expect(seen).toHaveLength(1);
    await expect(s.pushAudio(second, 16000)).rejects.toBeInstanceOf(SessionEndedError);
    await expect(s.end()).rejects.toBeInstanceOf(SessionEndedError);
    expect(s.mistakes()).toEqual([]);
    s.dispose();
  });

  it("an inference failure rejects that push with the backend error and ends the session", async () => {
    const err = new Error("backend exploded");
    const loaded = await loadNodeModel(tinyDir);
    await loaded.backend.release();
    loaded.backend = { name: "broken", run: async () => { throw err; }, release: async () => {} };
    const broken = await tinyTilawa({ model: loaded });
    const b = broken.startFreeRecitation();
    let ended = 0;
    b.onEnded(() => ended++);
    const first = b.pushAudio(second, 16000);
    const next = b.pushAudio(second, 16000);
    await expect(first).rejects.toBe(err);
    await expect(next).rejects.toBeInstanceOf(SessionEndedError);
    expect(b.state).toBe("ended");
    expect(ended).toBe(1);
    await expect(b.end()).resolves.toEqual([]);
    await broken.dispose();
  });

  it("idle() resolves once queued work is done; onEnded fires once when end() completes", async () => {
    const s = tilawa.startFreeRecitation();
    await s.idle();
    let done = 0;
    s.pushAudio(second, 16000).then(() => done++);
    s.pushAudio(second, 16000).then(() => done++);
    await s.idle();
    expect(done).toBe(2);
    let ended = 0;
    s.onEnded(() => ended++);
    const off = s.onEnded(() => ended += 100);
    off();
    await s.end();
    expect(ended).toBe(1);
    s.dispose();
    expect(ended).toBe(1);
    s.onEnded(() => ended++);
    expect(ended).toBe(2);
  });

  it("a throwing subscriber doesn't stop other subscribers or the session", async () => {
    // Subscriber errors are rethrown from a microtask; catch them here instead of failing the run.
    const errors: unknown[] = [];
    const spy = vi.spyOn(globalThis, "queueMicrotask").mockImplementation((f) => {
      try {
        f();
      } catch (e) {
        errors.push(e);
      }
    });
    const s = tilawa.startFreeRecitation();
    let first = true;
    s.subscribe(() => {
      if (first) return void (first = false);
      throw new Error("subscriber bug");
    });
    const seen: EngineEvent[] = [];
    s.subscribe((e) => seen.push(e));
    const out: EngineEvent[] = [];
    for (let a = 0; a < clip.samples.length; a += 8000) out.push(...(await s.pushAudio(clip.samples.subarray(a, a + 8000), clip.sampleRate)));
    out.push(...(await s.end()));
    spy.mockRestore();
    expect(out.length).toBeGreaterThan(0);
    expect(seen.slice(1)).toEqual(out);
    expect(errors).toHaveLength(out.length);
  });

  it("tilawa.dispose() disposes open sessions and refuses new ones", async () => {
    const t = await tinyTilawa();
    const s = t.startFreeRecitation();
    await t.dispose();
    expect(s.state).toBe("disposed");
    expect(() => t.startFreeRecitation()).toThrow();
    await t.dispose();
  });
});
