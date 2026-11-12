#!/usr/bin/env node
// Builds the immutable /v1/ asset tree that server/assets.ts serves (ASSETS_V1_DIR, default dist/v1):
//
//   models/<id>/<version>/model.onnx      every tier in packages/core/src/runtime/models-v1.json
//   models/<id>/<version>/manifest.json   that tier's manifest, from the registry
//   reference/ref-v2.json                 packages/core/assets/ref-v2.json
//   runtime/ort/<version>/ort-wasm-*      the onnxruntime-web files @tilawa/core/web loads, for the
//                                         onnxruntime-web version in this package's lock file
//   manifest.json                         every file above: path, bytes, sha256
//
// Model sources (one of):
//   --models-dir <dir>   local exports laid out as <dir>/<id>-<suffix>/model.onnx, e.g.
//                        spliced-clean-int8/ for spliced-clean@1-int8. Only model.onnx is read.
//   --models-url <url>   downloads. With {id} / {version} placeholders the URL is a template
//                        (e.g. a GitHub release: .../download/models-v1/{id}-{version}.onnx);
//                        otherwise it is a mirror of the /v1/ layout (<url>/models/<id>/<version>/model.onnx).
//
// Every model and the reference are checked against the sha256 in the registry; a mismatch fails the run.
// Files already in --out are never overwritten: the same bytes are kept, different bytes fail (they're immutable).
//
// Run after `vite build` (which empties dist/). Usage:
//   node scripts/prepare-v1-assets.mjs --models-dir ../../.scratch/models [--out dist/v1] [--tiers full,lite]
import { createHash } from "node:crypto";
import { createReadStream, createWriteStream, existsSync } from "node:fs";
import { copyFile, mkdir, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { dirname, join, resolve, sep } from "node:path";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const frontend = resolve(here, "..");
const core = resolve(frontend, "../../packages/core");

export class AssetError extends Error {}

export async function sha256File(path) {
  const h = createHash("sha256");
  for await (const chunk of createReadStream(path)) h.update(chunk);
  return h.digest("hex");
}

/** The onnxruntime-web version this package locks, checked against what's installed. */
export async function lockedOrtVersion(dir = frontend) {
  const lock = JSON.parse(await readFile(join(dir, "package-lock.json"), "utf8"));
  const version = lock.packages?.["node_modules/onnxruntime-web"]?.version;
  if (!version) throw new AssetError("onnxruntime-web is not in package-lock.json");
  const installed = JSON.parse(await readFile(join(dir, "node_modules/onnxruntime-web/package.json"), "utf8")).version;
  if (installed !== version)
    throw new AssetError(`node_modules has onnxruntime-web ${installed} but the lock says ${version}: run npm ci`);
  return { version, dist: join(dir, "node_modules/onnxruntime-web/dist") };
}

/** The .wasm/.mjs files onnxruntime-web/all (what @tilawa/core/web imports) fetches at runtime. */
export async function ortRuntimeFiles(dist) {
  const names = new Set();
  for (const bundle of ["ort.all.bundle.min.mjs", "ort.all.min.mjs"]) {
    const p = join(dist, bundle);
    if (!existsSync(p)) continue;
    for (const m of (await readFile(p, "utf8")).matchAll(/ort-wasm-simd-threaded[\w.]*?\.(?:mjs|wasm)/g)) names.add(m[0]);
  }
  const files = [...names].filter((n) => existsSync(join(dist, n))).sort();
  if (!files.some((f) => f.endsWith(".wasm"))) throw new AssetError(`No onnxruntime-web wasm files found in ${dist}`);
  return files;
}

function modelCandidates(dir, tier) {
  const short = tier.version.replace(/^\d+-/, "");
  return [`${tier.id}-${short}`, `${tier.id}-${tier.version}`, join(tier.id, tier.version)].map((d) => join(dir, d, tier.manifest.file));
}

function modelUrl(base, tier) {
  if (/\{(id|version|file)\}/.test(base))
    return base.replaceAll("{id}", tier.id).replaceAll("{version}", tier.version).replaceAll("{file}", tier.manifest.file);
  return `${base.replace(/\/?$/, "/")}models/${tier.id}/${tier.version}/${tier.manifest.file}`;
}

async function download(url, dest, log) {
  log(`  downloading ${url}`);
  let res;
  try {
    res = await fetch(url);
  } catch (e) {
    throw new AssetError(`Could not download ${url}: ${e?.cause?.message ?? e?.message ?? e}`);
  }
  if (!res.ok || !res.body) throw new AssetError(`${res.status} ${res.statusText} downloading ${url}`);
  await mkdir(dirname(dest), { recursive: true });
  const part = `${dest}.part`;
  try {
    await pipeline(Readable.fromWeb(res.body), createWriteStream(part));
  } catch (e) {
    await rm(part, { force: true });
    throw new AssetError(`Download of ${url} failed: ${e?.message ?? e}`);
  }
  await rename(part, dest);
}

/** Puts `src` at `out/rel` unless the identical file is already there; different bytes there fail. */
async function place(out, rel, src, expected, written) {
  const dest = join(out, rel);
  const bytes = (await stat(src)).size;
  const sha256 = expected?.sha256 ?? (await sha256File(src));
  if (existsSync(dest)) {
    const have = await sha256File(dest);
    if (have !== sha256) throw new AssetError(`${rel} already exists with different bytes (${have}); /v1/ files are immutable`);
  } else {
    await mkdir(dirname(dest), { recursive: true });
    await copyFile(src, `${dest}.part`);
    await rename(`${dest}.part`, dest);
  }
  written.push({ path: rel.split(sep).join("/"), bytes, sha256 });
}

async function placeText(out, rel, text, written) {
  const tmp = join(out, `.${rel.replaceAll("/", "_")}.tmp`);
  await mkdir(out, { recursive: true });
  await writeFile(tmp, text);
  try {
    await place(out, rel, tmp, null, written);
  } finally {
    await rm(tmp, { force: true });
  }
}

async function verify(path, want, what) {
  const { size } = await stat(path);
  if (size !== want.bytes) throw new AssetError(`${what}: ${path} is ${size} bytes, the registry says ${want.bytes}`);
  const got = await sha256File(path);
  if (got !== want.sha256) throw new AssetError(`${what}: sha256 mismatch for ${path}\n  expected ${want.sha256}\n  got      ${got}`);
}

export async function defaults() {
  const registry = JSON.parse(await readFile(join(core, "src/runtime/models-v1.json"), "utf8"));
  // The reference's expected size and digest, from the built package (packages/core/src/defaults.ts).
  const { REFERENCE, MODEL_REGISTRY } = await import("@tilawa/core");
  for (const t of registry.tiers) {
    const built = MODEL_REGISTRY.tiers.find((x) => x.tier === t.tier);
    if (built?.sha256 !== t.sha256) throw new AssetError(`packages/core/dist is stale: rebuild it (tier ${t.tier})`);
  }
  return {
    registry,
    reference: { path: join(core, "assets", REFERENCE.path.split("/").pop()), rel: REFERENCE.path, bytes: REFERENCE.bytes, sha256: REFERENCE.sha256 },
    ort: await lockedOrtVersion(),
  };
}

/**
 * @param {{ out: string, modelsDir?: string, modelsUrl?: string, tiers?: string[], registry?: any,
 *           reference?: { path: string, rel: string, bytes: number, sha256: string },
 *           ort?: { version: string, dist: string }, log?: (s: string) => void }} o
 */
export async function prepare(o) {
  const log = o.log ?? ((s) => console.log(s));
  if (!o.modelsDir === !o.modelsUrl) throw new AssetError("Pass exactly one of --models-dir or --models-url");
  const d = o.registry && o.reference && o.ort ? o : { ...(await defaults()), ...o };
  const out = resolve(o.out);
  const written = [];

  const tiers = d.registry.tiers.filter((t) => !o.tiers || o.tiers.includes(t.tier));
  if (!tiers.length) throw new AssetError(`No tiers match ${o.tiers}`);
  for (const t of tiers) {
    if (t.manifest.sha256 !== t.sha256 || t.manifest.bytes !== t.bytes) throw new AssetError(`Registry tier ${t.tier} disagrees with its manifest`);
    const rel = join("models", t.id, t.version);
    log(`model ${t.id}@${t.version} (${t.tier})`);
    let src;
    let cleanup = null;
    if (o.modelsDir) {
      src = modelCandidates(resolve(o.modelsDir), t).find((p) => existsSync(p));
      if (!src) throw new AssetError(`No model for ${t.id}@${t.version} in ${o.modelsDir} (tried ${modelCandidates(o.modelsDir, t).join(", ")})`);
    } else {
      src = cleanup = join(out, `.download-${t.id}-${t.version}.onnx`);
    }
    try {
      if (o.modelsUrl) await download(modelUrl(o.modelsUrl, t), src, log);
      await verify(src, t, `${t.id}@${t.version}`);
      await place(out, join(rel, t.manifest.file), src, t, written);
    } finally {
      if (cleanup) await rm(cleanup, { force: true });
    }
    await placeText(out, join(rel, "manifest.json"), `${JSON.stringify(t.manifest, null, 1)}\n`, written);
  }

  log(`reference ${d.reference.rel}`);
  await verify(d.reference.path, d.reference, "reference");
  await place(out, d.reference.rel, d.reference.path, d.reference, written);

  log(`runtime onnxruntime-web ${d.ort.version}`);
  for (const f of await ortRuntimeFiles(d.ort.dist)) await place(out, join("runtime/ort", d.ort.version, f), join(d.ort.dist, f), null, written);

  written.sort((a, b) => a.path.localeCompare(b.path));
  const manifest = { schema: 1, ort: d.ort.version, files: written };
  // The listing changes when files are added, so it's rewritten (it isn't one of the immutable files).
  await writeFile(join(out, "manifest.json"), `${JSON.stringify(manifest, null, 1)}\n`);
  log(`wrote ${written.length} files to ${out}`);
  return manifest;
}

function parseArgs(argv) {
  const o = { out: join(frontend, "dist/v1") };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const v = () => {
      if (i + 1 >= argv.length) throw new AssetError(`${a} needs a value`);
      return argv[++i];
    };
    if (a === "--models-dir") o.modelsDir = v();
    else if (a === "--models-url") o.modelsUrl = v();
    else if (a === "--out") o.out = v();
    else if (a === "--tiers") o.tiers = v().split(",");
    else throw new AssetError(`Unknown argument ${a}`);
  }
  return o;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    await prepare(parseArgs(process.argv.slice(2)));
  } catch (e) {
    console.error(`prepare-v1-assets: ${e instanceof AssetError ? e.message : e?.stack ?? e}`);
    process.exit(1);
  }
}
