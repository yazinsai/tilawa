import type { InferenceBackend, Tensor } from "./model.js";

/** The subset of the API that onnxruntime-node, onnxruntime-web and onnxruntime-react-native share. */
export interface OrtModule {
  Tensor: new (type: "float32" | "int64", data: Float32Array | BigInt64Array, dims: readonly number[]) => unknown;
  InferenceSession: {
    create(model: string | Uint8Array, options?: Record<string, unknown>): Promise<OrtSession>;
  };
}

interface OrtSession {
  run(feeds: Record<string, unknown>): Promise<Record<string, { type: string; data: unknown; dims: readonly number[] }>>;
  release?(): Promise<void>;
}

export async function ortBackend(ort: OrtModule, model: string | Uint8Array, name: string, options: Record<string, unknown> = {}): Promise<InferenceBackend> {
  const session = await ort.InferenceSession.create(model, { graphOptimizationLevel: "all", ...options });
  return {
    name,
    async run(feeds) {
      const input: Record<string, unknown> = {};
      for (const [k, t] of Object.entries(feeds)) input[k] = new ort.Tensor(t.type, t.data, t.dims);
      const out = await session.run(input);
      const res: Record<string, Tensor> = {};
      for (const [k, t] of Object.entries(out)) res[k] = { type: t.type as Tensor["type"], data: t.data as Tensor["data"], dims: t.dims };
      return res;
    },
    async release() {
      await session.release?.();
    },
  };
}
