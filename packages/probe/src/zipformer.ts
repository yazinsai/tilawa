import * as ort from "onnxruntime-web/all";

export interface ModelSpec {
  id: string;
  label: string;
  dir: string;
  encoder: string;
  decoder: string;
  joiner: string;
  /** Encoder frames consumed per call (decode_chunk_len); the window is the `x` input length. */
  shift: number;
}

export const MODELS: ModelSpec[] = [
  {
    id: "20m-int8",
    label: "Zipformer 20M int8 (~43 MB)",
    dir: "models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
    encoder: "encoder-epoch-99-avg-1.int8.onnx",
    decoder: "decoder-epoch-99-avg-1.int8.onnx",
    joiner: "joiner-epoch-99-avg-1.int8.onnx",
    shift: 32,
  },
  {
    id: "70m-int8",
    label: "Zipformer 2023-06-26 int8 (~71 MB)",
    dir: "models/sherpa-onnx-streaming-zipformer-en-2023-06-26",
    encoder: "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    decoder: "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    joiner: "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    shift: 32,
  },
  {
    id: "20m-fp32",
    label: "Zipformer 20M fp32 (~92 MB)",
    dir: "models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
    encoder: "encoder-epoch-99-avg-1.onnx",
    decoder: "decoder-epoch-99-avg-1.onnx",
    joiner: "joiner-epoch-99-avg-1.onnx",
    shift: 32,
  },
];

const fetchBytes = async (url: string) => {
  const cache = await caches.open("probe-models");
  let res = await cache.match(url);
  if (!res) {
    res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} fetching ${url}`);
    await cache.put(url, res.clone());
  }
  return new Uint8Array(await res.arrayBuffer());
};

const dims = (shape: readonly (number | string)[]) => shape.map((d) => (typeof d === "number" ? d : 1));

const zeros = (type: string, shape: number[]) => {
  const n = shape.reduce((a, b) => a * b, 1);
  return type === "int64" ? new ort.Tensor("int64", new BigInt64Array(n), shape) : new ort.Tensor("float32", new Float32Array(n), shape);
};

const meta = (s: ort.InferenceSession) =>
  s.inputMetadata.map((m) => (m.isTensor ? { name: m.name, type: m.type as string, shape: dims(m.shape) } : { name: m.name, type: "float32", shape: [1] }));

export class Zipformer {
  private states: Record<string, ort.Tensor> = {};
  private context = [0n, 0n];
  private decOut!: ort.Tensor;
  readonly tokens: string[] = [];
  readonly window: number;
  readonly melDim: number;

  private constructor(
    readonly spec: ModelSpec,
    private encoder: ort.InferenceSession,
    private decoder: ort.InferenceSession,
    private joiner: ort.InferenceSession,
    private vocab: string[],
  ) {
    const x = meta(encoder).find((m) => m.name === "x")!;
    this.window = x.shape[1];
    this.melDim = x.shape[2];
  }

  static async load(spec: ModelSpec, ep: "wasm" | "webgpu") {
    const opts: ort.InferenceSession.SessionOptions = { executionProviders: [ep], graphOptimizationLevel: "all" };
    const base = `${import.meta.env.BASE_URL}${spec.dir}/`;
    const [enc, dec, join, tokens] = await Promise.all([
      fetchBytes(base + spec.encoder),
      fetchBytes(base + spec.decoder),
      fetchBytes(base + spec.joiner),
      fetch(base + "tokens.txt").then((r) => r.text()),
    ]);
    const z = new Zipformer(
      spec,
      await ort.InferenceSession.create(enc, opts),
      await ort.InferenceSession.create(dec, opts),
      await ort.InferenceSession.create(join, opts),
      tokens
        .trim()
        .split("\n")
        .map((l) => l.split(" ")[0]),
    );
    await z.reset();
    return z;
  }

  async reset() {
    this.states = {};
    for (const m of meta(this.encoder)) if (m.name !== "x") this.states[m.name] = zeros(m.type, m.shape);
    this.context = [0n, 0n];
    this.tokens.length = 0;
    await this.runDecoder();
  }

  private async runDecoder() {
    const r = await this.decoder.run({ y: new ort.Tensor("int64", BigInt64Array.from(this.context), [1, 2]) });
    this.decOut = r.decoder_out;
  }

  /** Runs one encoder window plus greedy transducer search over its output frames. */
  async step(frames: Float32Array[]) {
    const x = new Float32Array(this.window * this.melDim);
    frames.forEach((f, i) => x.set(f, i * this.melDim));
    const out = await this.encoder.run({ x: new ort.Tensor("float32", x, [1, this.window, this.melDim]), ...this.states });
    for (const name of Object.keys(this.states)) this.states[name] = out[`new_${name}`];
    const enc = out.encoder_out;
    const [, t, d] = enc.dims;
    const data = (await enc.getData()) as Float32Array;
    for (let i = 0; i < t; i++) {
      const logit = (
        await this.joiner.run({
          encoder_out: new ort.Tensor("float32", data.slice(i * d, (i + 1) * d), [1, d]),
          decoder_out: this.decOut,
        })
      ).logit;
      const l = (await logit.getData()) as Float32Array;
      let best = 0;
      for (let k = 1; k < l.length; k++) if (l[k] > l[best]) best = k;
      if (best !== 0) {
        this.tokens.push(this.vocab[best]);
        this.context = [this.context[1], BigInt(best)];
        await this.runDecoder();
      }
    }
  }

  text() {
    return this.tokens.join("").replaceAll("▁", " ").trim();
  }
}
