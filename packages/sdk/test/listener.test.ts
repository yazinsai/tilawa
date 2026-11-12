import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { PhonemeEvidence } from "@tilawa/core";
import { createListener } from "../src/listener.js";
import type { InferenceBackend, LoadedModel, ModelManifest, Tensor } from "../src/model.js";
import { loadNodeModel } from "../src/node.js";
import { decodeWav } from "../src/wav.js";

const fixture = (p: string) => new URL(`./fixtures/${p}`, import.meta.url);
const reference = JSON.parse(readFileSync(fixture("tiny/reference.json"), "utf8")) as { logProbs: number[]; frames: number };
const wav = decodeWav(readFileSync(fixture("tlog-clean-00010-630.wav")));
const VOCAB = reference.logProbs.length / reference.frames;

async function run(step: number) {
  const listener = createListener(await loadNodeModel(fixture("tiny").pathname), { posteriors: true });
  const out: PhonemeEvidence[] = [];
  for (let a = 0; a < wav.samples.length; a += step) {
    const e = await listener.push(wav.samples.subarray(a, a + step), wav.sampleRate);
    if (e) out.push(e);
  }
  const last = await listener.end();
  if (last) out.push(last);
  return out;
}

describe("streaming listener on onnxruntime-node", () => {
  it("emits the same per-frame posteriors as the reference streaming loop over the same graph", async () => {
    const chunks = await run(4000);
    const frames = chunks.reduce((n, c) => n + c.posteriors!.nFrames, 0);
    expect(frames).toBe(reference.frames);
    let worst = 0;
    for (const c of chunks) {
      const { k, startFrame, nFrames, ids, probs } = c.posteriors!;
      for (let t = 0; t < nFrames; t++)
        for (let j = 0; j < k; j++) {
          const want = Math.exp(reference.logProbs[(startFrame + t) * VOCAB + ids[t * k + j]]);
          worst = Math.max(worst, Math.abs(want - probs[t * k + j]));
        }
    }
    expect(worst).toBeLessThan(1e-3);
  });

  it("emits contiguous chunks whose phonemes are the greedy CTC runs, whatever the push size", async () => {
    const a = await run(4000);
    const b = await run(16000 * 3);
    const flat = (cs: PhonemeEvidence[]) => cs.flatMap((c) => c.phonemes);
    expect(flat(a)).toEqual(flat(b));
    let next = 0;
    for (const c of a) {
      expect(c.posteriors!.startFrame).toBe(next);
      next += c.posteriors!.nFrames;
      for (const p of c.phonemes) expect(p.end).toBeLessThan(next);
    }
    const greedy: string[] = [];
    let prev = 0;
    for (let t = 0; t < reference.frames; t++) {
      const row = reference.logProbs.slice(t * VOCAB, (t + 1) * VOCAB);
      const id = row.indexOf(Math.max(...row));
      if (id !== 0 && id !== prev) greedy.push(String(id));
      prev = id;
    }
    expect(flat(a).length).toBe(greedy.length);
  });

  it("holds nothing back past the last complete window: a token run still open there is returned with it, once", async () => {
    const tiny = await loadNodeModel(fixture("tiny").pathname);
    const m = tiny.manifest;
    const vocab = m.tokens.length + 1;
    // Every frame is token 1: one run across every window.
    const backend: InferenceBackend = {
      name: "stub",
      async run(feeds) {
        const out: Record<string, Tensor> = Object.fromEntries(m.states.map((s) => [`new_${s.name}`, feeds[s.name]]));
        const lp = new Float32Array((m.chunk / 2) * vocab).fill(-10);
        for (let t = 0; t < m.chunk / 2; t++) lp[t * vocab + 1] = 0;
        out.log_probs = { type: "float32", data: lp, dims: [1, m.chunk / 2, vocab] };
        return out;
      },
    };
    const listener = createListener({ ...tiny, backend });
    const got: PhonemeEvidence[] = [];
    for (let a = 0; a < wav.samples.length; a += 1600) {
      const e = await listener.push(wav.samples.subarray(a, a + 1600), wav.sampleRate);
      if (e) got.push(e);
      expect(listener.decoded % (m.chunk / 2)).toBe(0);
    }
    expect(got[0].phonemes.map((p) => p.p)).toEqual([m.tokens[0]]);
    const last = await listener.end();
    expect([...got, ...(last ? [last] : [])].flatMap((e) => e.phonemes)).toHaveLength(1);
  });

  it("resamples 44.1 kHz input to the model rate", async () => {
    const listener = createListener(await loadNodeModel(fixture("tiny").pathname));
    const up = new Float32Array(Math.floor((wav.samples.length * 44100) / 16000));
    for (let i = 0; i < up.length; i++) up[i] = wav.samples[Math.min(wav.samples.length - 1, Math.round((i * 16000) / 44100))];
    let frames = 0;
    const count = (e: PhonemeEvidence | null) => {
      if (e?.phonemes.length) frames = Math.max(frames, e.phonemes.at(-1)!.end + 1);
    };
    count(await listener.push(up, 44100));
    count(await listener.end());
    expect(frames).toBeGreaterThan(0);
    expect(frames).toBeLessThanOrEqual(reference.frames);
  });
});

describe("primary integrate-and-fire listener", () => {
  function model(silenceAfter?: number): LoadedModel {
    const manifest = JSON.parse(readFileSync(fixture("tiny/manifest.json"), "utf8")) as ModelManifest;
    manifest.decoder = { kind: "cif", mass: "phone_mass", scores: "phone_scores" };
    const vocab = manifest.tokens.length + 1;
    const n = manifest.shift / 4;
    let frame = 0;
    return {
      manifest, supported: [], backend: {
        name: "cif-stub",
        async release() {},
        async run(feeds) {
          const out: Record<string, Tensor> = Object.fromEntries(manifest.states.map((s) => [`new_${s.name}`, feeds[s.name]]));
          const lp = new Float32Array(n * vocab).fill(-10);
          const scores = new Float32Array(n * vocab).fill(-2);
          for (let t = 0; t < n; t++) lp[t * vocab] = 0, scores[t * vocab + 1] = 2;
          out.log_probs = { type: "float32", data: lp, dims: [1, n, vocab] };
          const masses = Float32Array.from({ length: n }, () => silenceAfter === undefined ? 0.3 : frame++ < silenceAfter ? 0.7 : 0);
          out.phone_mass = { type: "float32", data: masses, dims: [1, n] };
          out.phone_scores = { type: "float32", data: scores, dims: [1, n, vocab] };
          return out;
        },
      },
    };
  }

  async function collect(step: number, input = model()) {
    const listener = createListener(input, { posteriors: true });
    const chunks: PhonemeEvidence[] = [];
    for (let a = 0; a < wav.samples.length; a += step) {
      const chunk = await listener.push(wav.samples.subarray(a, a + step), wav.sampleRate);
      if (chunk) {
        expect(chunk.decodedTo).toBe(listener.decoded);
        expect(chunk.phonemes.every((p) => p.start < listener.decoded)).toBe(true);
        chunks.push(chunk);
      }
    }
    const tail = await listener.end();
    if (tail) chunks.push(tail);
    return chunks;
  }

  it("uses pooled event identities and keeps independent CTC rows, invariant to push size", async () => {
    const a = await collect(1600);
    const b = await collect(wav.samples.length);
    const events = a.flatMap((c) => c.phonemes);
    expect(events).toEqual(b.flatMap((c) => c.phonemes));
    expect(events.length).toBe(Math.round(reference.frames * Math.fround(0.3)));
    let next = 0;
    for (const chunk of a) {
      const post = chunk.posteriors!;
      expect(post.startFrame).toBe(next);
      for (let t = 0; t < post.nFrames; t++) {
        expect(post.ids[t * post.k]).toBe(0);
        expect(post.probs[t * post.k]).toBe(1);
      }
      next += post.nFrames;
    }
    expect(next).toBe(reference.frames);
    expect(events.at(-1)!.integration!.contributions.at(-1)!.toSample).toBeLessThanOrEqual(next * 640);
    for (const chunk of a) for (const event of chunk.phonemes) {
      expect(event.start).toBeGreaterThanOrEqual(chunk.posteriors!.startFrame);
      expect(event.end).toBeLessThan(chunk.posteriors!.startFrame + chunk.posteriors!.nFrames);
    }
  });

  it("delivers a terminal partial event inside its posterior chunk even after trailing silence", async () => {
    const chunks = await collect(1600, model(4));
    const events = chunks.flatMap((c) => c.phonemes);
    expect(events).toHaveLength(3);
    const tail = chunks.at(-1)!;
    expect(tail.phonemes.at(-1)!.start).toBe(reference.frames - 1);
    expect(tail.phonemes.at(-1)!.integration!.contributions.at(-1)!.toSample).toBe(4 * 640);
    expect(tail.phonemes.at(-1)!.start).toBeGreaterThanOrEqual(tail.posteriors!.startFrame);
  });
});
