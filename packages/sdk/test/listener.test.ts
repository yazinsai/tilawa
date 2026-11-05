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

  it("carries separator frames on main tokens without changing phones or posteriors", async () => {
    async function decode(aux: boolean) {
      const tiny = await loadNodeModel(fixture("tiny").pathname);
      let frame = 0;
      const backend: InferenceBackend = {
        name: "separator-stub",
        async run(feeds) {
          const out = await tiny.backend.run(feeds);
          const n = out.log_probs.data.length / (tiny.manifest.tokens.length + 1);
          if (aux) {
            const mask = new BigInt64Array(n);
            for (let t = 0; t < n; t++) if ((frame + t) % 7 === 3) mask[t] = 1n;
            out.word_separator = { type: "int64", data: mask, dims: [1, n] };
          }
          frame += n;
          return out;
        },
        release: () => tiny.backend.release(),
      };
      const listener = createListener({ ...tiny, backend }, { posteriors: true });
      const a = await listener.push(wav.samples, wav.sampleRate);
      const b = await listener.end();
      await backend.release();
      return [a, b].filter((e): e is PhonemeEvidence => !!e);
    }
    const before = await decode(false), after = await decode(true);
    const withoutHead = (cs: PhonemeEvidence[]) => cs.map(e => ({ ...e,
      phonemes: e.phonemes.map(({ wordSeparators: _, ...p }) => p),
    }));
    expect(withoutHead(after)).toEqual(before);
    const proofs = after.flatMap(e=>e.phonemes.flatMap(p=>p.wordSeparators ?? []));
    expect(proofs.length).toBeGreaterThan(0);
    expect(proofs[0].fromFrame).toBe(0);
    const end = proofs.at(-1)!.toFrame;
    expect(proofs.flatMap(p=>p.frames)).toEqual(Array.from({length:end}, (_,f)=>f).filter(f=>f%7===3));
    expect(end).toBeLessThanOrEqual(reference.frames);
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
