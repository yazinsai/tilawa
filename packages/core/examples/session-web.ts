/**
 * Reference SessionRunner for the browser (onnxruntime-web / WASM).
 *
 * Copy this into your app. `@tilawa/core` never imports onnxruntime — you own
 * the `ort` dependency and wire it in behind the `SessionRunner` interface.
 *
 *   npm i onnxruntime-web
 *
 * Usage:
 *   const runner = await createWebSessionRunner(modelBuffer);
 *   const session = createTilawaSession(runner, { vocab, quranCtcTokens, quran });
 */
import * as ort from "onnxruntime-web/wasm";
import type { SessionRunner, SessionOutput } from "@tilawa/core";

export async function createWebSessionRunner(
  modelBuffer: ArrayBuffer,
): Promise<SessionRunner> {
  // Single-threaded is the reliable default: browser pthread startup can hang
  // in deployed worker contexts even when COOP/COEP headers look correct.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;

  const session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ["wasm"],
  });

  return {
    async run(audio: Float32Array): Promise<SessionOutput> {
      const inputTensor = new ort.Tensor("float32", audio, [1, audio.length]);
      const lengthTensor = new ort.Tensor(
        "int64",
        BigInt64Array.from([BigInt(audio.length)]),
        [1],
      );

      const results = await session.run({
        audio_signal: inputTensor,
        length: lengthTensor,
      });

      const output = results[session.outputNames[0]];
      const [, timeSteps, vocabSize] = output.dims as number[];

      return {
        logprobs: output.data as Float32Array,
        timeSteps,
        vocabSize,
      };
    },
  };
}
