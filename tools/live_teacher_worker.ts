/** Persistent worker serving the folded teacher through the SDK's exact frontend. */
import { createInterface } from "node:readline";
import * as ort from "onnxruntime-node";
import { ortBackend } from "../packages/sdk/src/ort.js";
import type { OrtModule } from "../packages/sdk/src/ort.js";
import { createInsertionTeacher } from "../packages/sdk/src/insertion-teacher.js";
import { MODEL_REGISTRY } from "../packages/sdk/src/registry.js";

const manifest = MODEL_REGISTRY.tiers.find(t => t.tier === "full")!.manifest;
const backend = await ortBackend(ort as unknown as OrtModule, process.argv[2], "onnxruntime-node", {
  executionProviders: ["cpu"], intraOpNumThreads: 4, graphOptimizationLevel: "disabled",
});
const teacher = createInsertionTeacher(backend, manifest.tokens, manifest.inventory);
process.stdout.write('{"ready":true}\n');
for await (const line of createInterface({ input: process.stdin })) {
  try {
    const request = JSON.parse(line);
    const bytes = Buffer.from(request.audio, "base64");
    const samples = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    const evidence = await teacher.decode(samples, request.k);
    process.stdout.write(JSON.stringify({ evidence }) + "\n");
  } catch (error) {
    process.stdout.write(JSON.stringify({ error: String(error) }) + "\n");
  }
}
