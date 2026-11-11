import { afterEach, describe, expect, it, vi } from "vitest";
import { ConsentError, RecorderError, createRecorder, encodeReport, validateReport } from "../../src/report/index.js";
import type { TilawaReport } from "../../src/report/index.js";
import { wavSamples } from "../../src/report/wav.js";
import { CONSENT, FakeSession, rampInt16 } from "./fake-session.js";

const pending = () => new Promise((r) => setTimeout(r, 0));

function expectRamp(report: TilawaReport) {
  const pcm = wavSamples(encodeReport(report).wav);
  expect(pcm.length).toBe(report.audio.durationMs * 16);
  const first = report.audio.startMs * 16;
  for (let i = 0; i < pcm.length; i += 97) expect(pcm[i]).toBe(rampInt16(first + i));
  expect(pcm[pcm.length - 1]).toBe(rampInt16(first + pcm.length - 1));
}

async function expectValid(report: TilawaReport) {
  const { json, wav } = encodeReport(report);
  const v = await validateReport(json, wav);
  expect(v.errors).toEqual([]);
}

afterEach(() => {
  vi.useRealTimers();
});

describe("createRecorder: timing and cropping", () => {
  it("crops [t - beforeMs, t + afterMs] exactly from chunks that are not ms-aligned", async () => {
    const s = new FakeSession();
    const rec = createRecorder(s, { source: "test", consent: CONSENT });
    s.pushMs(25_000, 333); // 333 samples = 20.8125 ms
    const r = await rec.report({ kind: "other", atMs: 15_000.4 });
    expect(r.audio.requested).toEqual({ startMs: 5000, endMs: 20001 });
    expect(r.audio).toMatchObject({ startMs: 5000, durationMs: 15001, truncatedStart: false, truncatedEnd: false });
    expectRamp(r);
    await expectValid(r);
  });

  it("does not call a request that starts before session start truncated", async () => {
    const s = new FakeSession();
    const rec = createRecorder(s, { source: "test", consent: CONSENT });
    s.pushMs(10_000, 160);
    const r = await rec.report({ kind: "stumble", atMs: 3000 });
    expect(r.audio).toMatchObject({ startMs: 0, durationMs: 8000, truncatedStart: false, requested: { startMs: 0, endMs: 8000 } });
    expectRamp(r);
  });

  it("records report fields from the session, capabilities and options", async () => {
    const s = new FakeSession({ mode: "memorization" });
    const rec = createRecorder(s, { source: "tilawa.dev/help", contributor: "spk_01", consent: CONSENT, clock: () => Date.UTC(2026, 10, 11, 12) });
    s.pushMs(6000);
    const r = await rec.report({ kind: "other", atMs: 500, claim: { answer: "no", note: "fine" } });
    expect(r.schema).toBe("tilawa.report/1");
    expect(r.sessionId).toBe(s.id);
    expect(r.createdAt).toBe("2026-11-11T12:00:00.000Z");
    expect(r.contributor).toBe("spk_01");
    expect(r.engine).toEqual({
      sdkVersion: "1.0.0", modelId: "zipformer-tajweed", modelVersion: "2026.11.1", modelSha256: "a".repeat(64),
      referenceVersion: "ref-v2", eventSchema: 1, tier: "full", backend: "wasm",
    });
    expect(r.session).toEqual({ mode: "memorization", passage: { surah: 18, fromAyah: 60, toAyah: 70 }, policy: { minConfidence: 0.6 }, startedAt: s.startedAt, vad: false });
    expect(r.incidents[0].claim).toEqual({ answer: "no", note: "fine" });
    expect(r.incidents[0].markedAt).toEqual({ sessionMs: 6000, wallMs: Date.UTC(2026, 10, 11, 12) });
    expect(r.consent).toEqual(CONSENT);
    expect(JSON.parse(encodeReport(r).json)).toEqual(JSON.parse(JSON.stringify(r)));
  });
});

describe("createRecorder: tap misuse", () => {
  it("fails without disturbing the session when the tap reports a non-16 kHz rate", async () => {
    const s = new FakeSession();
    const rec = createRecorder(s, { source: "test", consent: CONSENT });
    s.pushMs(1000);
    expect(() => [...s.audioSubs].forEach((fn) => fn(new Float32Array(4800), s.now(), 48_000))).not.toThrow();
    expect(rec.state).toBe("failed");
    expect(rec.error?.message).toMatch(/48000 Hz/);
    expect(s.audioSubs.size).toBe(0);
    await expect(rec.report({ kind: "other" })).rejects.toMatchObject({ name: "RecorderError", reason: "failed" });
  });

  it("flags 48 kHz chunks fed through a 16 kHz tap by their clock", () => {
    const s = new FakeSession();
    const rec = createRecorder(s, { source: "test", consent: CONSENT });
    const tap = [...s.audioSubs][0];
    tap(new Float32Array(4800), 0); // 100 ms at 48 kHz, read as 300 ms
    tap(new Float32Array(4800), 100);
    expect(rec.state).toBe("failed");
    expect(rec.error?.message).toMatch(/clock jumped/);
  });
});

describe("createRecorder: missing and late context", () => {
  it("clamps a start that left the ring (truncatedStart)", async () => {
    const s = new FakeSession();
    const rec = createRecorder(s, { source: "test", consent: CONSENT, retain: { beforeMs: 10_000, afterMs: 5000 } });
    s.pushMs(60_000, 1000); // ring holds the last 45 s: [15 s, 60 s]
    expect(rec.stats().retainedFromMs).toBe(15_000);
    const r = await rec.report({ kind: "other", atMs: 20_000 });
    expect(r.audio).toMatchObject({ startMs: 15_000, durationMs: 10_000, truncatedStart: true, truncatedEnd: false, requested: { startMs: 10_000, endMs: 25_000 } });
    expectRamp(r);
    await expectValid(r);
  });

  it("marks truncatedStart when the recorder joined mid-session", async () => {
    const s = new FakeSession();
    s.pushMs(30_000);
    const rec = createRecorder(s, { source: "test", consent: CONSENT });
    s.pushMs(20_000);
    const r = await rec.report({ kind: "other", atMs: 32_000 });
    expect(r.audio).toMatchObject({ startMs: 30_000, truncatedStart: true, durationMs: 7000 });
    expectRamp(r);
  });

  it("waits for the after-context during a take", async () => {
    const s = new FakeSession();
    const rec = createRecorder(s, { source: "test", consent: CONSENT, retain: { take: true, maxMs: 120_000 } });
    s.pushMs(10_000);
    let done = false;
    const p = rec.report({ kind: "stumble" }).then((r) => ((done = true), r));
    await pending();
    expect(done).toBe(false);
    s.pushMs(4999);
    await pending();
    expect(done).toBe(false);
    s.pushMs(1);
    const r = await p;
    expect(r.audio).toMatchObject({ startMs: 0, durationMs: 15_000, truncatedEnd: false, requested: { startMs: 0, endMs: 15_000 } });
    expect(r.incidents[0]).toMatchObject({ startMs: 10_000, endMs: 10_000, markedAt: { sessionMs: 10_000 } });
    expectRamp(r);
  });

  it("waits on session.idle before cropping, so late events are in", async () => {
    const s = new FakeSession();
    let release!: () => void;
    (s as unknown as { idle: () => Promise<void> }).idle = () => new Promise<void>((r) => (release = r));
    const rec = createRecorder(s, { source: "test", consent: CONSENT });
    s.pushMs(20_000);
    const p = rec.report({ kind: "other", atMs: 5000 });
    await pending();
    s.checker("late", 6000); // inference caught up after the audio arrived
    release();
    const r = await p;
    expect(r.observations.events.map((e) => e.type === "checker" && e.id)).toContain("late");
  });

  it("reports after the session ended", async () => {
    const s = new FakeSession();
    const rec = createRecorder(s, { source: "test", consent: CONSENT });
    s.pushMs(20_000);
    s.end();
    expect(rec.state).toBe("ended");
    expect(s.audioSubs.size).toBe(0);
    const r = await rec.report({ kind: "other", atMs: 10_000 });
    expect(r.audio).toMatchObject({ startMs: 0, durationMs: 15_000, truncatedStart: false, truncatedEnd: false });
  });

  it("resolves at session end when the after-context never arrives (truncatedEnd)", async () => {
    const s = new FakeSession();
    const rec = createRecorder(s, { source: "test", consent: CONSENT });
    s.pushMs(10_000);
    const p = rec.report({ kind: "missed_mistake" });
    s.pushMs(2000);
    s.end();
    const r = await p;
    expect(r.audio).toMatchObject({ startMs: 0, durationMs: 12_000, truncatedEnd: true, requested: { startMs: 0, endMs: 15_000 } });
    await expectValid(r);
  });

  it("detects session end by polling state when the session has no onEnded", async () => {
    vi.useFakeTimers();
    const s = new FakeSession({ onEnded: false });
    const rec = createRecorder(s, { source: "test", consent: CONSENT });
    s.pushMs(10_000);
    const p = rec.report({ kind: "other" });
    s.end();
    await vi.advanceTimersByTimeAsync(300);
    expect(rec.state).toBe("ended");
    const r = await p;
    expect(r.audio.truncatedEnd).toBe(true);
  });
});

describe("createRecorder: events and incidents", () => {
  it("handles multiple reports with overlapping clips and links events outside the span", async () => {
    const s = new FakeSession();
    const rec = createRecorder(s, { source: "test", consent: CONSENT });
    s.pushMs(3000);
    s.checker("c1", 3000, "suspected");
    s.pushMs(5000);
    s.position(7000);
    s.checker("c2", 8000, "confirmed");
    s.pushMs(24_000);
    s.position(25_000);
    s.checker("c1", 30_000, "confirmed");
    s.pushMs(10_000);
    const [a, b] = await Promise.all([
      rec.report({ kind: "false_warning", eventId: "c1", atMs: 30_000, claim: { answer: "no" } }),
      rec.report({ kind: "false_warning", eventId: "c2", claim: { answer: "yes" } }),
    ]);
    expect(a.reportId).not.toBe(b.reportId);
    expect(a.audio).toMatchObject({ startMs: 20_000, durationMs: 15_000 });
    expect(b.audio).toMatchObject({ startMs: 0, durationMs: 13_000 });
    const summary = (r: TilawaReport) => r.observations.events.map((e) => (e.type === "checker" ? `${e.id}@${e.timeMs}` : `${e.type}${"timeMs" in e ? "@" + e.timeMs : ""}`));
    expect(summary(a)).toEqual(["capabilities", "c1@3000", "position@25000", "c1@30000"]);
    expect(summary(b)).toEqual(["capabilities", "c1@3000", "position@7000", "c2@8000"]);
    expect(b.incidents[0]).toMatchObject({ startMs: 8000, endMs: 8000, eventIds: ["c2"] });
    expect(rec.incidents().map((i) => i.eventIds[0])).toEqual(["c1", "c2"]);

    const m1 = rec.mark({ kind: "stumble" });
    const m2 = rec.mark({ kind: "stumble", startMs: 40_000, endMs: 41_000 });
    expect(rec.incidents()).toHaveLength(4);
    const take = await rec.contribution({ incidents: [m1, m2] });
    expect(take.incidents.map((i) => i.id)).toEqual([m1.id, m2.id]);
    expect(take.audio).toMatchObject({ startMs: 0, durationMs: 42_000, truncatedStart: false, truncatedEnd: false, requested: { startMs: 0, endMs: 42_000 } });
    await expectValid(take);
  });

  it("caps incidents at 100", () => {
    const s = new FakeSession();
    const rec = createRecorder(s, { source: "test", consent: CONSENT });
    for (let i = 0; i < 100; i++) rec.mark({ kind: "stumble" });
    expect(() => rec.mark({ kind: "stumble" })).toThrow(RecorderError);
  });

  it("clips the evidence slice when evidence: true, and omits it otherwise", async () => {
    const evidence = [
      {
        inventory: "tajweed-v1",
        frameMs: 40,
        phonemes: [
          { p: "a", start: 10, end: 12, conf: 0.9 }, // 400-520 ms
          { p: "b", start: 140, end: 150, conf: 0.8 }, // 5600-6040 ms
        ],
        posteriors: { k: 2, startFrame: 0, nFrames: 200, ids: Array.from({ length: 400 }, (_, i) => i % 7), probs: Array.from({ length: 400 }, () => 0.5) },
      },
    ];
    const s = new FakeSession({ evidence });
    const on = createRecorder(s, { source: "test", consent: CONSENT, evidence: true, retain: { beforeMs: 1000, afterMs: 1000 } });
    const off = createRecorder(s, { source: "test", consent: CONSENT, retain: { beforeMs: 1000, afterMs: 1000 } });
    s.pushMs(10_000);
    const r = await on.report({ kind: "other", atMs: 6000 });
    expect(r.observations.evidence).toHaveLength(1);
    const ev = r.observations.evidence![0];
    expect(ev.phonemes.map((p) => p.p)).toEqual(["b"]);
    expect(ev.posteriors).toMatchObject({ k: 2, startFrame: 125, nFrames: 50 });
    expect(Array.from(ev.posteriors!.ids)).toHaveLength(100);
    expect((await off.report({ kind: "other", atMs: 6000 })).observations.evidence).toBeUndefined();
  });

  it("requires consent with send: true, and a source", async () => {
    const s = new FakeSession();
    expect(() => createRecorder(s, {} as never)).toThrow(/source/);
    expect(() => createRecorder(s, { source: "x", contributor: "bad id!" })).toThrow(/contributor/);
    const rec = createRecorder(s, { source: "test" });
    s.pushMs(1000);
    await expect(rec.report({ kind: "other" })).rejects.toBeInstanceOf(ConsentError);
    await expect(rec.contribution()).rejects.toBeInstanceOf(ConsentError);
    await expect(rec.contribution({ consent: { ...CONSENT, send: false as unknown as true } })).rejects.toBeInstanceOf(ConsentError);
    expect(rec.incidents()).toHaveLength(0);
    const ok = await rec.contribution({ consent: CONSENT });
    expect(ok.incidents).toEqual([]);
  });
});

describe("createRecorder: lifetime and memory", () => {
  it("dispose stops the tap, drops buffers and rejects pending reports", async () => {
    const s = new FakeSession();
    const rec = createRecorder(s, { source: "test", consent: CONSENT });
    s.pushMs(1000);
    const p = rec.report({ kind: "other" });
    rec.dispose();
    rec.dispose();
    await expect(p).rejects.toMatchObject({ reason: "disposed" });
    expect(s.audioSubs.size).toBe(0);
    expect(s.eventSubs.size).toBe(0);
    expect(rec.stats()).toMatchObject({ state: "disposed", capacityBytes: 0 });
    expect(() => rec.mark({ kind: "other" })).toThrow(/disposed/);
  });

  it("keeps buffers retainAfterEndMs after end, then expires them", async () => {
    vi.useFakeTimers();
    const s = new FakeSession();
    const rec = createRecorder(s, { source: "test", consent: CONSENT, retainAfterEndMs: 60_000 });
    s.pushMs(20_000);
    s.end();
    await vi.advanceTimersByTimeAsync(59_000);
    expect((await rec.report({ kind: "other", atMs: 10_000 })).audio.durationMs).toBe(15_000);
    await vi.advanceTimersByTimeAsync(1001);
    expect(rec.state).toBe("expired");
    expect(rec.stats().capacityBytes).toBe(0);
    expect(s.eventSubs.size).toBe(0);
    await expect(rec.report({ kind: "other", atMs: 10_000 })).rejects.toMatchObject({ reason: "expired" });
  });

  it("defaults retainAfterEndMs to 10 minutes", async () => {
    vi.useFakeTimers();
    const s = new FakeSession();
    const rec = createRecorder(s, { source: "test", consent: CONSENT });
    s.pushMs(1000);
    s.end();
    await vi.advanceTimersByTimeAsync(599_999);
    expect(rec.state).toBe("ended");
    await vi.advanceTimersByTimeAsync(2);
    expect(rec.state).toBe("expired");
  });

  it("never grows the ring past its bound over a 20-minute stream", async () => {
    const s = new FakeSession();
    const rec = createRecorder(s, { source: "test", consent: CONSENT });
    const bound = (10_000 + 5000 + 30_000) * 32; // contract: (before + after + 30 s) x 32 KB/s
    expect(rec.stats().boundBytes).toBe(bound);
    let maxBytes = 0;
    let maxEvents = 0;
    for (let t = 0; t < 20 * 60_000; t += 1000) {
      s.pushMs(1000, 1234);
      for (let k = 40; k <= 1000; k += 40) s.position(t + k);
      if (t % 60_000 === 0) s.checker(`c${t}`, t);
      const st = rec.stats();
      maxBytes = Math.max(maxBytes, st.capacityBytes);
      maxEvents = Math.max(maxEvents, st.events);
    }
    expect(maxBytes).toBe(bound);
    expect(bound).toBeLessThanOrEqual(1.44e6);
    expect(maxEvents).toBeLessThan(46 * 25 + 64 + 25);
    const r = await rec.report({ kind: "other", atMs: 20 * 60_000 - 5000 });
    expect(r.audio).toMatchObject({ durationMs: 15_000, truncatedStart: false, truncatedEnd: false });
    expectRamp(r);
  });

  it("hard-caps take mode at maxMs and marks truncatedEnd", async () => {
    const s = new FakeSession();
    const rec = createRecorder(s, { source: "test", consent: CONSENT, retain: { take: true, maxMs: 60_000 } });
    s.pushMs(90_000, 4000);
    expect(rec.stats().capacityBytes).toBeLessThanOrEqual(60_000 * 32);
    expect(rec.stats().retainedToMs).toBe(60_000);
    const take = await rec.contribution();
    expect(take.audio).toMatchObject({ startMs: 0, durationMs: 60_000, truncatedEnd: true, requested: { startMs: 0, endMs: 90_000 } });
    expectRamp(take);
    await expectValid(take);
    const near = await rec.report({ kind: "other", atMs: 58_000 });
    expect(near.audio).toMatchObject({ startMs: 48_000, durationMs: 12_000, truncatedEnd: true });
    await expect(rec.report({ kind: "other", atMs: 80_000 })).rejects.toMatchObject({ reason: "empty" });
  });

  it("defaults take mode to 300 s and rejects a cap above it", () => {
    const s = new FakeSession();
    expect(createRecorder(s, { source: "t", retain: { take: true } }).stats().boundBytes).toBe(300_000 * 32);
    expect(() => createRecorder(s, { source: "t", retain: { take: true, maxMs: 300_001 } })).toThrow(RangeError);
  });
});

describe("opt-out baseline", () => {
  it("adds no tap subscriber and does no network I/O without a recorder", async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    try {
      const s = new FakeSession();
      s.pushMs(5000);
      expect(s.audioSubs.size).toBe(0);
      const rec = createRecorder(s, { source: "test", consent: CONSENT });
      expect(s.audioSubs.size).toBe(1);
      s.pushMs(5000);
      encodeReport(await rec.report({ kind: "other", atMs: 5000 }));
      rec.dispose();
      expect(s.audioSubs.size).toBe(0);
      expect(fetchSpy).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
