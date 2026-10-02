import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
const require = createRequire("/workspace/web/frontend/node_modules/");
const which = process.argv[2] ?? "node";
const ort = which === "web" ? require("onnxruntime-web") : require("onnxruntime-node");
if (which === "web") { ort.env.wasm.numThreads = 1; }
const io = JSON.parse(readFileSync("/workspace/packages/core/src/recitation/zipformer-io.json", "utf8"));
const N = Number(process.argv[3] ?? 150);
for (const v of ["base", "enc", "head_lin", "head_mlp256", "base"]) {
  const opts = which === "web" ? { executionProviders: ["wasm"], graphOptimizationLevel: "all" }
    : { executionProviders: ["cpu"], graphOptimizationLevel: "all", intraOpNumThreads: 1, interOpNumThreads: 1 };
  const s = await ort.InferenceSession.create(new Uint8Array(readFileSync(`/tmp/headprobe/${v}.onnx`)), opts);
  const states = {};
  for (const i of io.inputs) if (i.name !== "x") {
    const n = i.dims.reduce((a, b) => a * b, 1);
    states[i.name] = i.dtype === "int64" ? new ort.Tensor("int64", new BigInt64Array(n), i.dims) : new ort.Tensor("float32", new Float32Array(n), i.dims);
  }
  const ts = [];
  for (let k = 0; k < N + 10; k++) {
    const x = new Float32Array(61 * 80); for (let j = 0; j < x.length; j++) x[j] = Math.random() * 4 - 8;
    const t0 = performance.now();
    const out = await s.run({ x: new ort.Tensor("float32", x, [1, 61, 80]), ...states });
    const dt = performance.now() - t0;
    for (const n of Object.keys(states)) states[n] = out[`new_${n}`];
    if (k >= 10) ts.push(dt);
  }
  ts.sort((a, b) => a - b);
  const med = ts[Math.floor(ts.length / 2)], p90 = ts[Math.floor(ts.length * 0.9)];
  console.log(which, v.padEnd(12), "median", med.toFixed(2), "ms  p90", p90.toFixed(2), "ms  outputs", s.outputNames.filter(n => !n.startsWith("new_")).join(","));
}
