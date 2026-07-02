/**
 * The ONNX injection seam.
 *
 * `@tilawa/core` is runtime-agnostic: it never imports `onnxruntime-*`.
 * Instead the app developer wires in their runtime of choice (onnxruntime-web,
 * onnxruntime-node, onnxruntime-react-native, ...) behind this interface.
 *
 * The contract mirrors the model graph exactly: feed mono 16kHz float32 PCM
 * as `audio_signal` `[1, N]` plus its `length` `[1]`, receive `[1, T, vocab]`
 * log-probabilities flattened row-major into `logprobs` (length `T * vocab`).
 * Preprocessing (mel, normalization) is baked into the graph.
 *
 * Reference implementations for each runtime live in the README. The web demo's
 * adapter is `web/frontend/src/worker/session.ts`.
 */
export interface SessionRunner {
  /**
   * Run one forward pass of the acoustic model.
   *
   * @param audio - mono 16kHz PCM, float32, shape `[N]`.
   * @returns flattened log-probs (`timeSteps * vocabSize`) plus the two dims
   *   needed to reshape them into `[T, vocab]`.
   */
  run(audio: Float32Array): Promise<SessionOutput>;
}

export interface SessionOutput {
  /** Row-major `[timeSteps, vocabSize]` log-probabilities. */
  logprobs: Float32Array;
  timeSteps: number;
  vocabSize: number;
}
