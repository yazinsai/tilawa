import type { ErrorClass, PhonemeEvidence } from "../engine/index.js";

export interface StateSpec {
  name: string;
  shape: number[];
  dtype: "float32" | "int64";
}

/** `manifest.json` next to each exported model (written by `modal/student_export.py`). */
export interface ModelManifest {
  schema: 1;
  id: string;
  format: string;
  arch: "zipformer2-streaming-ctc";
  /** Must equal the reference index inventory id. */
  inventory: string;
  /** Token `i + 1` is `tokens[i]`; 0 is the CTC blank. */
  tokens: string;
  blank: 0;
  sampleRate: number;
  frameMs: number;
  fbank: { mels: number; padValue: number };
  /** Fbank frames per call and frames to advance by. */
  window: number;
  shift: number;
  states: StateSpec[];
  file: string;
  bytes: number;
  sha256: string;
}

export interface Tensor {
  type: "float32" | "int64";
  data: Float32Array | BigInt64Array;
  dims: readonly number[];
}

/** One ONNX session, whatever the runtime. */
export interface InferenceBackend {
  readonly name: string;
  run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
  release(): Promise<void>;
}

export interface LoadedModel {
  manifest: ModelManifest;
  backend: InferenceBackend;
  /** Error classes this model's evidence supports; the rest are reported `unsupported`. */
  supported: ErrorClass[];
  /** Declared fp32 KD teacher; primary streaming outputs are preserved. */
  insertionTeacher?: { decode(samples: Float32Array, k: number): Promise<PhonemeEvidence> };
}
