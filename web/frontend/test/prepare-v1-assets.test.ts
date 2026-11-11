import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, expect, it } from "vitest";
// @ts-expect-error plain .mjs script
import { prepare } from "../scripts/prepare-v1-assets.mjs";

const sha = (b: Buffer | string) => createHash("sha256").update(b).digest("hex");
let dir: string;

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "v1-assets-"));
});
afterEach(() => rm(dir, { recursive: true, force: true }));

async function fixture(modelBytes = Buffer.from("fake onnx bytes")) {
  const good = Buffer.from("fake onnx bytes");
  await mkdir(join(dir, "models/spliced-clean-int8"), { recursive: true });
  await writeFile(join(dir, "models/spliced-clean-int8/model.onnx"), modelBytes);
  await writeFile(join(dir, "models/spliced-clean-int8/manifest.json"), "{\"not\":\"copied\"}");
  await mkdir(join(dir, "ort"), { recursive: true });
  await writeFile(join(dir, "ort/ort.all.bundle.min.mjs"), 'x="ort-wasm-simd-threaded.jsep.mjs";y="ort-wasm-simd-threaded.jsep.wasm"');
  await writeFile(join(dir, "ort/ort-wasm-simd-threaded.jsep.mjs"), "mjs");
  await writeFile(join(dir, "ort/ort-wasm-simd-threaded.jsep.wasm"), "wasm");
  await writeFile(join(dir, "ort/ort-wasm-simd-threaded.wasm"), "not needed");
  const ref = Buffer.from('{"version":"ref-v2"}');
  await writeFile(join(dir, "ref-v2.json"), ref);
  const manifest = { id: "spliced-clean", file: "model.onnx", bytes: good.length, sha256: sha(good) };
  return {
    out: join(dir, "out"),
    modelsDir: join(dir, "models"),
    registry: { schema: 1, tiers: [{ tier: "full", id: "spliced-clean", version: "1-int8", bytes: good.length, sha256: sha(good), manifest }] },
    reference: { path: join(dir, "ref-v2.json"), rel: "reference/ref-v2.json", bytes: ref.length, sha256: sha(ref) },
    ort: { version: "1.30.0", dist: join(dir, "ort") },
    log: () => {},
  };
}

it("lays out models, reference and runtime, and lists every file", async () => {
  const o = await fixture();
  const manifest = await prepare(o);
  expect(manifest.files.map((f: { path: string }) => f.path)).toEqual([
    "models/spliced-clean/1-int8/manifest.json",
    "models/spliced-clean/1-int8/model.onnx",
    "reference/ref-v2.json",
    "runtime/ort/1.30.0/ort-wasm-simd-threaded.jsep.mjs",
    "runtime/ort/1.30.0/ort-wasm-simd-threaded.jsep.wasm",
  ]);
  const model = await readFile(join(o.out, "models/spliced-clean/1-int8/model.onnx"));
  expect(sha(model)).toBe(o.registry.tiers[0].sha256);
  // The tier manifest comes from the registry, not from the models dir.
  expect(JSON.parse(await readFile(join(o.out, "models/spliced-clean/1-int8/manifest.json"), "utf8"))).toEqual(o.registry.tiers[0].manifest);
  const listed = JSON.parse(await readFile(join(o.out, "manifest.json"), "utf8"));
  for (const f of listed.files) expect(sha(await readFile(join(o.out, f.path)))).toBe(f.sha256);
  // Idempotent.
  expect(await prepare(o)).toEqual(manifest);
});

it("fails loudly on a model that doesn't match the registry", async () => {
  const o = await fixture(Buffer.from("tampered onnx b"));
  await expect(prepare(o)).rejects.toThrow(/sha256 mismatch/);
});

it("refuses to overwrite an immutable file with different bytes", async () => {
  const o = await fixture();
  await mkdir(join(o.out, "reference"), { recursive: true });
  await writeFile(join(o.out, "reference/ref-v2.json"), "older");
  await expect(prepare(o)).rejects.toThrow(/immutable/);
});
