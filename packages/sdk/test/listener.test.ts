import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { PhonemeEvidence } from "@tilawa/core";
import { createListener } from "../src/listener.js";
import { PitchTrack } from "../src/pitch.js";
import type { InferenceBackend, LoadedModel, Tensor } from "../src/model.js";
import { loadNodeModel } from "../src/node.js";
import { decodeWav } from "../src/wav.js";

const fixture = (p: string) => new URL(`./fixtures/${p}`, import.meta.url);
const reference = JSON.parse(readFileSync(fixture("tiny/reference.json"), "utf8")) as { logProbs: number[]; frames: number };
const wav = decodeWav(readFileSync(fixture("tlog-clean-00010-630.wav")));
const VOCAB = reference.logProbs.length / reference.frames;

async function run(step: number, model?: LoadedModel) {
  const listener = createListener(model ?? await loadNodeModel(fixture("tiny").pathname), { posteriors: true });
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

  it("preserves the raw pitch clock through blank frames, chunk boundaries and JSON recording", async () => {
    const model = await loadNodeModel(fixture("tiny").pathname);
    let frame = 0;
    // The tiny graph has no blank argmaxes. Alternate two token frames with
    // two blank frames so this test actually exercises inter-token silence.
    const backend: InferenceBackend = {
      name: "blank-runs",
      release: () => model.backend.release(),
      async run(feeds) {
        const out = await model.backend.run(feeds);
        const lp = new Float32Array(out.log_probs.data.length).fill(-10);
        const vocab = model.manifest.tokens.length + 1;
        for (let t = 0; t < lp.length / vocab; t++) lp[t * vocab + (frame++ % 4 < 2 ? 1 : 0)] = 0;
        return { ...out, log_probs: { ...out.log_probs, data: lp } };
      },
    };
    const chunks = await run(317, { ...model, backend });
    const pitch = new PitchTrack(wav.sampleRate, chunks[0].frameMs);
    pitch.push(wav.samples);
    pitch.end();
    const phones = chunks.flatMap(c => c.phonemes);
    expect(JSON.parse(JSON.stringify(phones))).toEqual(phones);
    let until = 0;
    let blankFrames = 0;
    for (const p of phones) {
      expect(p.pitch).toEqual(pitch.slice(until, p.end + 1));
      expect(p.pitch).toHaveLength(p.end + 1 - until);
      blankFrames += p.start - until;
      until = p.end + 1;
    }
    expect(blankFrames).toBeGreaterThan(0);
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
