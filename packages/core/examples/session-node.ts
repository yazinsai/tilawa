/**
 * Reference SessionRunner for Node.js (onnxruntime-node).
 *
 *   npm i onnxruntime-node
 *
 * Usage:
 *   const modelBuffer = await readFile("model.onnx");
 *   const runner = await createNodeSessionRunner(modelBuffer);
 *   const session = createTilawaSession(runner, { vocab, quranCtcTokens, quran });
 */
import * as ort from "onnxruntime-node";
import type { SessionRunner, SessionOutput } from "@tilawa/core";

export async function createNodeSessionRunner(
  modelBuffer: Uint8Array,
): Promise<SessionRunner> {
  const session = await ort.InferenceSession.create(modelBuffer);

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
