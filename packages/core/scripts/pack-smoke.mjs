#!/usr/bin/env node
/**
 * Packs @tilawa/core, installs the tarball into a fresh Node project, imports every subpath (types too) and runs a
 * Node session on the tiny fixture model + WAV. Checks the tarball holds only dist, assets and the docs.
 *
 *   node scripts/pack-smoke.mjs [--keep]
 *
 * Installs onnxruntime-node/web at this package's dev versions (from the npm cache when present). A stub stands in
 * for onnxruntime-react-native, whose real package needs react-native.
 */
import { execFileSync } from "node:child_process";
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const pkgDir = join(dirname(fileURLToPath(import.meta.url)), "..");
const pkg = JSON.parse(readFileSync(join(pkgDir, "package.json"), "utf8"));
const keep = process.argv.includes("--keep");
const tmp = mkdtempSync(join(tmpdir(), "tilawa-pack-"));
const run = (cmd, args, cwd) => execFileSync(cmd, args, { cwd, stdio: ["ignore", "pipe", "inherit"], maxBuffer: 64 << 20 }).toString();
const fail = (msg) => {
  console.error(`pack-smoke: ${msg}`);
  process.exit(1);
};

try {
  // 1. Pack (prepack syncs the docs and rebuilds dist).
  const [packed] = JSON.parse(run("npm", ["pack", "--json", "--pack-destination", tmp], pkgDir));
  const tarball = join(tmp, packed.filename);
  const files = packed.files.map((f) => f.path).sort();
  const allowed = /^(dist\/.+\.(js|d\.ts|json)|assets\/(ref-v2|phrases-v1)\.json|README\.md|LICENSE|NOTICE\.md|CHANGELOG\.md|package\.json)$/;
  const stray = files.filter((f) => !allowed.test(f) || /(^|\/)(test|bench|relabel|review|e2e|buckets|fixtures)\//.test(f));
  if (stray.length) fail(`unexpected files in the tarball:\n  ${stray.join("\n  ")}`);
  for (const need of ["dist/index.js", "dist/web.js", "dist/node.js", "dist/react-native.js", "dist/report/index.js", "assets/ref-v2.json", "LICENSE", "NOTICE.md", "README.md", "CHANGELOG.md"])
    if (!files.includes(need)) fail(`missing ${need}`);

  // 2. A consumer project.
  const app = join(tmp, "app");
  // Inside the app, so the (symlinked) stub resolves onnxruntime-common from the app's node_modules.
  const stub = join(app, "stub-onnxruntime-react-native");
  mkdirSync(stub, { recursive: true });
  writeFileSync(join(stub, "package.json"), JSON.stringify({ name: "onnxruntime-react-native", version: "1.30.0", type: "module", exports: { ".": { types: "./index.d.ts", default: "./index.js" } } }));
  writeFileSync(join(stub, "index.js"), 'export * from "onnxruntime-common";\n');
  writeFileSync(join(stub, "index.d.ts"), 'export * from "onnxruntime-common";\n');
  writeFileSync(join(app, "package.json"), JSON.stringify({ name: "smoke", private: true, type: "module" }));
  const dev = pkg.devDependencies;
  run("npm", ["install", "--no-audit", "--no-fund", "--prefer-offline", tarball, `onnxruntime-node@${dev["onnxruntime-node"]}`,
    `onnxruntime-web@${dev["onnxruntime-web"]}`, `onnxruntime-common@${dev["onnxruntime-web"]}`, stub], app);
  const installed = JSON.parse(readFileSync(join(app, "node_modules/@tilawa/core/package.json"), "utf8"));
  if (installed.version !== pkg.version) fail(`installed ${installed.version}, expected ${pkg.version}`);
  for (const dep of Object.keys(installed.dependencies ?? {})) fail(`unexpected runtime dependency ${dep}`);

  cpSync(join(pkgDir, "test/fixtures/tiny"), join(app, "tiny"), { recursive: true });
  cpSync(join(pkgDir, "test/fixtures/tlog-clean-00010-630.wav"), join(app, "clip.wav"));

  // 3. Runtime: every subpath, then a Node session.
  writeFileSync(join(app, "smoke.mjs"), `
import assert from "node:assert/strict";
import * as root from "@tilawa/core";
import * as web from "@tilawa/core/web";
import * as node from "@tilawa/core/node";
import * as rn from "@tilawa/core/react-native";
import * as report from "@tilawa/core/report";
import ref from "@tilawa/core/assets/ref-v2.json" with { type: "json" };
import phrases from "@tilawa/core/assets/phrases-v1.json" with { type: "json" };
import manifest from "@tilawa/core/package.json" with { type: "json" };

for (const k of ["createEngine", "loadReferenceIndex", "ayahsRecited", "DEFAULTS", "MODEL_REGISTRY", "fetchModel", "withBase", "SessionEndedError", "TilawaLoadError"])
  assert.ok(k in root, "root." + k);
for (const k of ["createTilawa", "loadWebModel", "startMicrophone", "cacheStorage", "decodeWav"]) assert.equal(typeof web[k], "function", "web." + k);
for (const k of ["createTilawa", "loadNodeModel", "loadNodeTier", "fsStorage", "readWav"]) assert.equal(typeof node[k], "function", "node." + k);
for (const k of ["createTilawa", "loadNativeModel"]) assert.equal(typeof rn[k], "function", "react-native." + k);
for (const name of ["createRecorder", "encodeReport", "decodeReport", "validateReport", "sendReport", "REPORT_SCHEMA"]) assert.ok(name in report, "report export missing: " + name);
assert.equal(ref.version, "ref-v2");
assert.equal(phrases.version, "phrases-v1");
assert.equal(manifest.version, root.SDK_VERSION);

const tilawa = await node.createTilawa({ modelDir: new URL("./tiny", import.meta.url).pathname });
const session = tilawa.startFreeRecitation();
const events = [];
session.subscribe((e) => events.push(e));
const wav = await node.readWav(new URL("./clip.wav", import.meta.url).pathname);
for (let a = 0; a < wav.samples.length; a += 8000) session.pushAudio(wav.samples.subarray(a, a + 8000), wav.sampleRate);
await session.end();
assert.equal(events[0].type, "capabilities");
assert.equal(events[0].sdkVersion, root.SDK_VERSION);
assert.ok(events.some((e) => e.type === "position"), "position events");
assert.equal(session.state, "ended");
assert.ok(session.now() > 5000);
await tilawa.dispose();
console.log("runtime ok: " + events.length + " events, " + Math.round(session.now()) + " ms of audio");
`);
  process.stdout.write(run(process.execPath, ["smoke.mjs"], app));

  // 4. Types: the subpaths resolve with declarations under NodeNext.
  writeFileSync(join(app, "types.ts"), `
import { ayahsRecited, type EngineEvent, type TilawaSession } from "@tilawa/core";
import { createTilawa as web } from "@tilawa/core/web";
import { createTilawa as node, type NodeTilawaOptions } from "@tilawa/core/node";
import { createTilawa as rn } from "@tilawa/core/react-native";
import "@tilawa/core/report";
const o: NodeTilawaOptions = { cacheDir: "/tmp" };
export async function f(s: TilawaSession) {
  const off: () => void = s.onAudio((pcm: Float32Array, startMs: number) => void [pcm, startMs]);
  const e: EngineEvent[] = await s.end();
  off();
  return [ayahsRecited(e), web, node(o), rn, s.id, s.startedAt, s.now(), s.policy.confirm] as const;
}
`);
  writeFileSync(join(app, "tsconfig.json"), JSON.stringify({
    compilerOptions: { target: "ES2022", module: "NodeNext", moduleResolution: "NodeNext", strict: true, noEmit: true, lib: ["ES2022", "DOM"], types: [], skipLibCheck: false },
    files: ["types.ts"],
  }));
  run(process.execPath, [join(pkgDir, "node_modules/typescript/bin/tsc"), "-p", "tsconfig.json"], app);
  console.log("types ok");

  console.log(`${packed.filename}: ${files.length} files, ${(statSync(tarball).size / 1024 / 1024).toFixed(2)} MB packed, ${(packed.unpackedSize / 1024 / 1024).toFixed(2)} MB unpacked`);
  if (keep) console.log(`kept ${tmp}`);
} finally {
  if (!keep) rmSync(tmp, { recursive: true, force: true });
}
