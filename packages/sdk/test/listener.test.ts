import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { PhonemeEvidence } from "@tilawa/core";
import { createListener } from "../src/listener.js";
import type { InferenceBackend, Tensor } from "../src/model.js";
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

describe("causal session conditioning", () => {
  it("uses completed audio after one second and is independent of push size and other sessions", async () => {
    const manifest = JSON.parse(readFileSync(fixture("tiny/manifest.json"), "utf8"));
    manifest.sessionConditioning = { dimensions: 64, frameMs: 20, startMs: 1000 };
    const execute = async (pushSize: number) => {
      const embeddings: number[][] = [];
      const backend: InferenceBackend = {
        name: "conditioned-stub",
        async run(feeds) {
          embeddings.push(Array.from(feeds.session_embedding.data as Float32Array));
          const call = embeddings.length;
          const n = manifest.shift / 4;
          return {
            ...Object.fromEntries(manifest.states.map((s: { name: string }) => [`new_${s.name}`, feeds[s.name]])),
            log_probs: { type: "float32", data: new Float32Array(n * (manifest.tokens.length + 1)).fill(-10), dims: [1, n, manifest.tokens.length + 1] },
            session_features: { type: "float32", data: new Float32Array(n * 2 * 64).fill(call), dims: [1, n * 2, 64] },
          };
        },
      };
      const listener = createListener({ manifest, backend, supported: [] });
      const audio = new Float32Array(16000 * 3);
      for (let a = 0; a < audio.length; a += pushSize) await listener.push(audio.subarray(a, a + pushSize), 16000);
      await listener.end();
      return embeddings;
    };
    const small = await execute(1600), large = await execute(48000);
    expect(small).toEqual(large);
    expect(small.slice(0, 4).every(e => e.every(v => v === 0))).toBe(true);
    expect(small[4]).toEqual(new Array(64).fill(2.5));
    expect(small[5]).toEqual(new Array(64).fill(3));
    // A new listener starts from zero even when the model is shared.
    const empty: number[][] = [];
    const backend: InferenceBackend = {
      name: "missing-features",
      async run(feeds) {
        empty.push(Array.from(feeds.session_embedding.data as Float32Array));
        return { log_probs: { type: "float32", data: new Float32Array(8 * (manifest.tokens.length + 1)), dims: [1, 8, manifest.tokens.length + 1] },
          ...Object.fromEntries(manifest.states.map((s: { name: string }) => [`new_${s.name}`, feeds[s.name]])) };
      },
    };
    await expect(createListener({ manifest, backend, supported: [] }).push(new Float32Array(16000), 16000)).rejects.toThrow("session_features");
    expect(empty[0]).toEqual(new Array(64).fill(0));
  });
});
