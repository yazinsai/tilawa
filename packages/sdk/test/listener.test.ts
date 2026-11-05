import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { PhonemeEvidence } from "@tilawa/core";
import { createListener } from "../src/listener.js";
import type { InferenceBackend, Tensor } from "../src/model.js";
import { loadNodeModel } from "../src/node.js";
import { decodeWav } from "../src/wav.js";
import { Fbank } from "../src/fbank.js";

const fixture = (p: string) => new URL(`./fixtures/${p}`, import.meta.url);
const reference = JSON.parse(readFileSync(fixture("tiny/reference.json"), "utf8")) as { logProbs: number[]; frames: number };
const wav = decodeWav(readFileSync(fixture("tlog-clean-00010-630.wav")));
const VOCAB = reference.logProbs.length / reference.frames;

async function run(step: number, seamGrid = true) {
  const listener = createListener(await loadNodeModel(fixture("tiny").pathname), { posteriors: true, seamGrid });
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
  it("keeps primary evidence unchanged while retaining the whole shifted grid", async () => {
    const a = await run(4000), b = await run(4000, false);
    const primary = (chunks: PhonemeEvidence[]) => ({
      phonemes: chunks.flatMap(c => c.phonemes.map(({ seamGrid: _, ...p }) => p)),
      ids: chunks.flatMap(c => Array.from(c.posteriors!.ids)),
      probs: chunks.flatMap(c => Array.from(c.posteriors!.probs)),
    });
    expect(primary(a)).toEqual(primary(b));
    const grid = a.flatMap(c => c.phonemes.flatMap(p => p.seamGrid ? [p.seamGrid] : []));
    let until = 0;
    for (const rows of grid) { expect(rows.startFrame).toBe(until); until += rows.ids.length; }
    expect(until).toBeGreaterThan(a.flatMap(c => c.phonemes).at(-1)!.end);
    expect(grid[0].ids.slice(0, 4)).toEqual([-1, -1, -1, -1]);
  });

  it("uses independent states and shifted features without replacing primary rows", async () => {
    const tiny = await loadNodeModel(fixture("tiny").pathname), m = tiny.manifest;
    const nRows = m.shift / 4, vocab = m.tokens.length + 1;
    const calls: { x: Float32Array; processed: bigint }[] = [];
    const backend: InferenceBackend = {
      name: "grid-geometry-stub",
      async run(feeds) {
        const processed = (feeds.processed_lens.data as BigInt64Array)[0];
        calls.push({ x: (feeds.x.data as Float32Array).slice(), processed });
        const out: Record<string, Tensor> = Object.fromEntries(m.states.map(s => [`new_${s.name}`, feeds[s.name]]));
        out.new_processed_lens = { type: "int64", data: BigInt64Array.of(processed + 1n), dims: [1] };
        const lp = new Float32Array(nRows * vocab).fill(-10);
        for (let t = 0; t < nRows; t++) lp[t * vocab + (calls.length % 2 ? 1 : 2 + t % 2)] = 0;
        out.log_probs = { type: "float32", data: lp, dims: [1, nRows, vocab] };
        return out;
      },
    };
    const audio = wav.samples.subarray(0, 16000), feats = new Fbank().push(audio);
    const listener = createListener({ ...tiny, backend }, { posteriors: true });
    const chunks = [await listener.push(audio, wav.sampleRate), await listener.end()].filter((e): e is PhonemeEvidence => !!e);
    const total = Math.floor((Math.floor((feats.length - 7) / 2) + 1) / 2);
    expect(chunks.reduce((n, c) => n + c.posteriors!.nFrames, 0)).toBe(total);
    for (let c = 0; c < calls.length; c++) {
      expect(calls[c].processed).toBe(BigInt(Math.floor(c / 2)));
      const from = Math.floor(c / 2) * m.shift + (c % 2) * m.shift / 2;
      for (let f = 0; f < m.window; f++) expect(calls[c].x.subarray(f * m.fbank.mels, (f + 1) * m.fbank.mels))
        .toEqual(feats[from + f] ?? new Float32Array(m.fbank.mels).fill(m.fbank.padValue));
    }
    expect(chunks.flatMap(c => c.phonemes).map(p => p.p)).toEqual([m.tokens[0]]);
    const grid = chunks[0].phonemes[0].seamGrid!;
    expect(grid.ids.slice(0, nRows / 2)).toEqual(Array(nRows / 2).fill(-1));
    expect(grid.ids.slice(nRows / 2)).toEqual(Array.from({ length: grid.ids.length - nRows / 2 }, (_, i) => 2 + i % 2));
    await tiny.backend.release();
  });

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
