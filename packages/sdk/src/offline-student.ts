import type { DecodedPhoneme, PhonemeEvidence } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { topK } from "./listener.js";
import type { InferenceBackend } from "./model.js";

/** Full bidirectional CTC: a complete crop per call, no streaming state. */
export function createOfflineStudent(backend: InferenceBackend, tokens: string, inventory: string) {
  return {
    async decode(samples: Float32Array, k: number): Promise<PhonemeEvidence> {
      const fb = new Fbank().push(samples);
      if (fb.length < 11) throw new Error("Offline crop is shorter than one output frame");
      const data = new Float32Array(fb.length * 80);
      fb.forEach((f, i) => data.set(f, i * 80));
      const out = await backend.run({ x: { type: "float32", data, dims: [1, fb.length, 80] },
        x_lens: { type: "int64", data: BigInt64Array.of(BigInt(fb.length)), dims: [1] } });
      const lp = out.log_probs.data as Float32Array, vocab = tokens.length + 1;
      const n = Number((out.out_lens.data as BigInt64Array)[0]);
      if (!Number.isInteger(n) || n < 1 || n * vocab > lp.length)
        throw new Error("Offline output differs from its declared length or inventory");
      const phonemes: DecodedPhoneme[] = [], ids: number[] = [], probs: number[] = [];
      let prev = 0;
      for (let f = 0; f < n; f++) {
        const top = topK(lp.subarray(f * vocab, (f + 1) * vocab), k);
        for (const [id, p] of top) ids.push(id), probs.push(p);
        const [id, conf] = top[0];
        if (id && id === prev) {
          phonemes.at(-1)!.end = f;
          phonemes.at(-1)!.conf = Math.max(phonemes.at(-1)!.conf, conf);
        } else if (id) phonemes.push({ p: tokens[id - 1], start: f, end: f, conf });
        prev = id;
      }
      return { inventory, frameMs: 40, phonemes, decodedTo: n,
        posteriors: { k, startFrame: 0, nFrames: n, ids, probs } };
    },
  };
}
