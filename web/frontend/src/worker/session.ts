import * as ort from "onnxruntime-web/wasm";

let session: ort.InferenceSession | null = null;

export async function createSession(modelBuffer: ArrayBuffer): Promise<void> {
  // Browser pthread startup can hang in deployed worker contexts even when
  // COOP/COEP headers look correct. Prefer a reliable single-threaded session;
  // inference can be tuned separately once production init is stable.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;

  session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ["wasm"],
  });
}

export async function runInference(
  audio: Float32Array,
): Promise<{ logprobs: Float32Array; timeSteps: number; vocabSize: number }> {
  if (!session) throw new Error("Session not initialized");

  const inputTensor = new ort.Tensor("float32", audio, [1, audio.length]);
  const lengthTensor = new ort.Tensor(
    "int64",
    BigInt64Array.from([BigInt(audio.length)]),
    [1],
  );

  const feeds: Record<string, ort.Tensor> = {
    audio_signal: inputTensor,
    length: lengthTensor,
  };

  const results = await session.run(feeds);
  const outputTensor = results[session.outputNames[0]];
  const [_batch, timeSteps, vocabSize] = outputTensor.dims as number[];

  return {
    logprobs: outputTensor.data as Float32Array,
    timeSteps,
    vocabSize,
  };
}
