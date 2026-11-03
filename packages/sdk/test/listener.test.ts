import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { minGapFrames } from "@tilawa/core";
import type { DecodedPhoneme, PhonemeEvidence } from "@tilawa/core";
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

describe("pause-onset hamza", () => {
  const spec = (id: number, n: number, prob = 1) => Array.from({ length: n }, () => ({ id, prob }));

  async function decode(frames: { id: number; prob: number }[]) {
    const tiny = await loadNodeModel(fixture("tiny").pathname);
    const m = tiny.manifest;
    const vocab = m.tokens.length + 1;
    const per = m.chunk / 2;
    let cursor = 0;
    const backend: InferenceBackend = {
      name: "stub",
      async run(feeds) {
        const out: Record<string, Tensor> = Object.fromEntries(m.states.map((s) => [`new_${s.name}`, feeds[s.name]]));
        const lp = new Float32Array(per * vocab);
        const floor = Math.log(1e-6);
        for (let t = 0; t < per; t++) {
          const row = frames[cursor] ?? { id: 0, prob: 1 };
          if (cursor < frames.length) cursor++;
          lp.fill(floor, t * vocab, (t + 1) * vocab);
          lp[t * vocab + row.id] = Math.log(row.prob);
        }
        out.log_probs = { type: "float32", data: lp, dims: [1, per, vocab] };
        return out;
      },
    };
    const listener = createListener({ ...tiny, backend });
    const silence = new Float32Array(m.sampleRate * 3);
    const phonemes: DecodedPhoneme[] = [];
    const take = (e: PhonemeEvidence | null) => { if (e) phonemes.push(...e.phonemes); };
    take(await listener.push(silence, m.sampleRate));
    take(await listener.end());
    return { phonemes, pause: minGapFrames(m.frameMs), hamza: m.tokens[0], other: m.tokens[1] };
  }

  it("tags the hamza that is the first token after a blank-owned pause, including across a window", async () => {
    const pause = minGapFrames(40);
    // Three frames of a consonant, then the pause, so the blank run starts mid-window and crosses the next one.
    const { phonemes, hamza } = await decode([...spec(2, 3), ...spec(0, pause), ...spec(1, 2), ...spec(2, 2)]);
    const marked = phonemes.filter((p) => p.pauseOnset);
    expect(hamza).toBe("ء");
    expect(marked.map((p) => p.p)).toEqual(["ء"]);
    expect(phonemes.filter((p) => p.p === "ء")).toHaveLength(1);
  });

  it("does not tag a hamza after a shorter gap, a blank run below 0.5, a non-hamza, or no gap", async () => {
    const pause = minGapFrames(40);
    const short = await decode([...spec(0, pause - 1), ...spec(1, 2)]);
    expect(short.phonemes.find((p) => p.p === "ء")?.pauseOnset).toBeUndefined();
    const weak = await decode([...spec(0, pause, 0.4), ...spec(1, 2)]);
    expect(weak.phonemes.find((p) => p.p === "ء")?.pauseOnset).toBeUndefined();
    const consonant = await decode([...spec(0, pause), ...spec(2, 2)]);
    expect(consonant.phonemes.some((p) => p.p === consonant.other)).toBe(true);
    expect(consonant.phonemes.some((p) => p.pauseOnset)).toBe(false);
    const bare = await decode(spec(1, 3));
    expect(bare.phonemes[0]).toMatchObject({ p: "ء" });
    expect(bare.phonemes[0].pauseOnset).toBeUndefined();
  });
});
