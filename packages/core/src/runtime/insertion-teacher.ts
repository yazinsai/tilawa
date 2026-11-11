import type { DecodedPhoneme, PhonemeEvidence } from "../engine/index.js";
import { Fbank } from "./fbank.js";
import { topK } from "./listener.js";
import type { InferenceBackend } from "./model.js";

/** The KD teacher's SeamlessM4T front end: Kaldi mel bins to Nyquist, per-bin unbiased variance, stride two. */
export function teacherFeatures(samples: Float32Array): { data: Float32Array; frames: number; mask: BigInt64Array } {
  const pcm = Float32Array.from(samples, x => x * 32768);
  const fb = new Fbank({ highFrequency: 8000 }).push(pcm);
  if (fb.length < 2) throw new Error("Teacher crop is shorter than two feature frames");
  const n = fb.length;
  const frames = Math.ceil(n / 2);
  // The saved teacher processor pads a single unpaired feature frame with ones.
  const data = new Float32Array(frames * 160).fill(1);
  for (let m = 0; m < 80; m++) {
    const mean = fb.reduce((sum, f) => sum + f[m], 0) / n;
    const variance = fb.reduce((sum, f) => sum + (f[m] - mean) ** 2, 0) / (n - 1);
    const scale = Math.sqrt(variance + 1e-7);
    for (let t = 0; t < n; t++) data[t * 80 + m] = (fb[t][m] - mean) / scale;
  }
  const mask = new BigInt64Array(frames).fill(1n);
  if (n % 2) mask[frames - 1] = 0n;
  return { data, frames, mask };
}

export function createInsertionTeacher(backend: InferenceBackend, tokens: string, inventory: string) {
  return {
    async decode(samples: Float32Array, k: number): Promise<PhonemeEvidence> {
      const features = teacherFeatures(samples);
      const out = await backend.run({ input_features: { type: "float32", data: features.data, dims: [1, features.frames, 160] },
        attention_mask: { type: "int64", data: features.mask, dims: [1, features.frames] } });
      const lp = out.log_probs.data as Float32Array, vocab = tokens.length + 1;
      if (lp.length % vocab) throw new Error("Teacher output differs from the listener inventory");
      const phonemes: DecodedPhoneme[] = [], ids: number[] = [], probs: number[] = [];
      let prev = 0;
      for (let f = 0; f < lp.length / vocab; f++) {
        const top = topK(lp.subarray(f * vocab, (f + 1) * vocab), k);
        for (const [id, p] of top) ids.push(id), probs.push(p);
        const [id, conf] = top[0];
        if (id && id === prev) {
          phonemes.at(-1)!.end = f;
          phonemes.at(-1)!.conf = Math.max(phonemes.at(-1)!.conf, conf);
        } else if (id) phonemes.push({ p: tokens[id - 1], start: f, end: f, conf });
        prev = id;
      }
      return { inventory, frameMs: 40, phonemes, posteriors: { k, startFrame: 0, nFrames: lp.length / vocab, ids, probs } };
    },
  };
}
