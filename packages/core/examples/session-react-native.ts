/**
 * Reference SessionRunner for React Native (onnxruntime-react-native).
 *
 *   npm i onnxruntime-react-native
 *
 * RN can't hand the model to ORT as an ArrayBuffer — load from a file path on
 * device (bundle the .onnx as an asset, copy to documents dir, pass the path).
 *
 * Usage:
 *   const runner = await createRNSessionRunner(modelPath);
 *   const session = createTilawaSession(runner, { vocab, quranCtcTokens, quran });
 */
import * as ort from "onnxruntime-react-native";
import type { SessionRunner, SessionOutput } from "@tilawa/core";

export async function createRNSessionRunner(
  modelPath: string,
): Promise<SessionRunner> {
  const session = await ort.InferenceSession.create(modelPath);

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
