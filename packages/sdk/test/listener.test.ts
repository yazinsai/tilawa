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

  it("carries unrounded auxiliary rows without changing main phones or posteriors", async () => {
    const tiny=await loadNodeModel(fixture("tiny").pathname);
    const head={tokens:tiny.manifest.tokens+"|",cleanMaxMargin:2};
    const wrapped: InferenceBackend={name:"head",release:()=>tiny.backend.release(),async run(feeds){
      const out=await tiny.backend.run(feeds),n=out.log_probs.data.length/(tiny.manifest.tokens.length+1);
      return {...out,word_count_log_probs:{type:"float32",dims:[1,n,head.tokens.length+1],
        data:new Float32Array(n*(head.tokens.length+1)).fill(-0.123456789)}};
    }};
    const primary=createListener(tiny,{posteriors:true});
    const extra=createListener({...tiny,manifest:{...tiny.manifest,wordCount:head},backend:wrapped},{posteriors:true});
    const a=[await primary.push(wav.samples,wav.sampleRate),await primary.end()].filter(Boolean) as PhonemeEvidence[];
    const b=[await extra.push(wav.samples,wav.sampleRate),await extra.end()].filter(Boolean) as PhonemeEvidence[];
    const strip=(cs:PhonemeEvidence[])=>cs.map(c=>({...c,phonemes:c.phonemes.map(({wordCountPosteriors:_,...p})=>p)}));
    expect(strip(b)).toEqual(a);
    const packets=b.flatMap(c=>c.phonemes.flatMap(p=>p.wordCountPosteriors??[]));
    expect(packets.length).toBeGreaterThan(0);
    expect(packets[0].startFrame).toBe(0);
    expect(packets[0].logProbs[0]).toBe(Math.fround(-0.123456789));
    expect(JSON.parse(JSON.stringify(packets))).toEqual(packets);
    for(let i=1;i<packets.length;i++)expect(packets[i].startFrame).toBe(packets[i-1].startFrame+packets[i-1].nFrames);
    await tiny.backend.release();
  });

  it("fails when an export declares a head but omits its output or calibration", async () => {
    const tiny=await loadNodeModel(fixture("tiny").pathname);
    const manifest={...tiny.manifest,wordCount:{tokens:tiny.manifest.tokens+"|",cleanMaxMargin:2}};
    const listener=createListener({...tiny,manifest});
    await expect(listener.push(wav.samples,wav.sampleRate)).rejects.toThrow("head output");
    expect(()=>createListener({...tiny,manifest:{...manifest,wordCount:{...manifest.wordCount,cleanMaxMargin:NaN}}}))
      .toThrow("training clean envelope");
    await tiny.backend.release();
  });
});
