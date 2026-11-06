import { expect, it, vi } from "vitest";
import { createWordDecoder, decodeWords } from "../src/word-decoder.js";
import type { InferenceBackend } from "../src/model.js";

it("rejoins split UTF-8 and BPE pieces into separate whole words", () => {
  const b = [...new TextEncoder().encode("كريم وقال")];
  const words = decodeWords([{ bytes: b.slice(0, 1), logp: -.1 }, { bytes: b.slice(1, 8), logp: -.3 },
    { bytes: b.slice(8), logp: -.5 }]);
  expect(words).toEqual([{ text: "كريم", logp: -.2 }, { text: "وقال", logp: -.5 }]);
});

it("runs the autoregressive decoder without any expected text prompt", async () => {
  const encoder = { run: vi.fn(async () => ({ encoder_hidden_states: { type: "float32", data: new Float32Array(1), dims: [1,1,1] } })) } as unknown as InferenceBackend;
  let step = 0;
  const decoder = { run: vi.fn(async () => ({ logits: { type: "float32", data: Float32Array.from(step++ ? [10, 0, 0] : [0, 10, 0]), dims: [1,1,3] } })) } as unknown as InferenceBackend;
  const model = createWordDecoder(encoder, decoder, { schema: 1, samples: 16, prefix: [2], eos: 0, maxTokens: 4, suppressed: [], vocabulary: ["", "a", ""] });
  const heard = await model.decode(new Float32Array(8));
  expect(heard[0].text).toBe("a");
  expect((decoder.run as ReturnType<typeof vi.fn>).mock.calls[0][0].input_ids.data).toEqual(BigInt64Array.from([2n]));
  expect(encoder.run).toHaveBeenCalledTimes(1);
});

it("abstains on truncated decoding and overlong audio", async () => {
  const backend = { run: vi.fn(async () => ({ encoder_hidden_states: { type: "float32", data: new Float32Array(1), dims: [1,1,1] },
    logits: { type: "float32", data: Float32Array.from([0, 10]), dims: [1,1,2] } })) } as unknown as InferenceBackend;
  const model = createWordDecoder(backend, backend, { schema: 1, samples: 8, prefix: [0], eos: 0, maxTokens: 2, suppressed: [], vocabulary: ["", "a"] });
  expect(await model.decode(new Float32Array(9))).toEqual([]);
  expect(backend.run).not.toHaveBeenCalled();
  expect(await model.decode(new Float32Array(8))).toEqual([]);
});
