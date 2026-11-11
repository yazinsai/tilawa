#!/usr/bin/env node
/**
 * Vendors the research engine into this package from one pinned commit.
 *
 *   node scripts/sync-engine.mjs --repo <path to research checkout> [--commit <sha>] [--check]
 *
 * Research `packages/core/src/*.ts` -> `src/engine/`, research `packages/sdk/src/*.ts` (+ models-v1.json) ->
 * `src/runtime/`. Research-only dirs (bench, relabel, review, e2e, buckets, fixtures, sdk/bench) are not copied.
 * The only edit is import specifiers: `@tilawa/core` in the runtime becomes `../engine/index.js`, and the synthetic
 * evidence fixture (copied to `test/fixtures/synthetic.ts`) points at `src/engine/`. Everything is read with
 * `git show <commit>:<path>`, so the research checkout's working tree and branch don't matter.
 *
 * `src/engine/SOURCE.json` records {repo, commit, files: {path: sha256}}: every file this script writes, keyed by
 * its path in this package, hashed as written. `--check` rewrites nothing and exits 1 if any file differs.
 */
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

import { COMMIT, REPO } from "./pin.mjs";

const pkg = join(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const opt = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
};
const repo = opt("--repo") ?? process.env.TILAWA_RESEARCH_REPO;
const commit = opt("--commit") ?? COMMIT;
const check = args.includes("--check");
if (!repo) {
  console.error("usage: node scripts/sync-engine.mjs --repo <research checkout> [--commit <sha>] [--check]");
  process.exit(2);
}

const git = (...a) => execFileSync("git", ["-C", repo, ...a], { maxBuffer: 64 << 20 });
const ls = (dir) =>
  git("ls-tree", "--name-only", commit, `${dir}/`).toString().split("\n").filter(Boolean).map((p) => p.slice(dir.length + 1));
const show = (path) => git("show", `${commit}:${path}`);

/** [source path in research, destination in this package, rewrite] */
const plan = [];
const toEngine = (s) => s.replaceAll(/from "@tilawa\/core"/g, 'from "../engine/index.js"');
for (const f of ls("packages/core/src")) if (f.endsWith(".ts")) plan.push([`packages/core/src/${f}`, `src/engine/${f}`, null]);
for (const f of ls("packages/sdk/src")) {
  if (f.endsWith(".ts")) plan.push([`packages/sdk/src/${f}`, `src/runtime/${f}`, toEngine]);
  else if (f.endsWith(".json")) plan.push([`packages/sdk/src/${f}`, `src/runtime/${f}`, null]);
}
// The shipped config passes the phrase asset to every engine; the wrappers bundle it from here.
plan.push(["packages/core/assets/phrases-v1.json", "src/engine/phrases-v1.json", null]);
plan.push(["packages/core/assets/ref-v2.json", "assets/ref-v2.json", null]);
plan.push(["packages/core/assets/phrases-v1.json", "assets/phrases-v1.json", null]);
// Deterministic test inputs (not shipped).
plan.push(["packages/core/src/fixtures/index.ts", "test/fixtures/synthetic.ts",
  (s) => s.replaceAll(/from "\.\.\/([a-z-]+\.js)"/g, 'from "../../src/engine/$1"')]);
for (const f of ["tiny/manifest.json", "tiny/model.onnx", "tiny/reference.json", "tlog-clean-00010-630.wav"])
  plan.push([`packages/sdk/test/fixtures/${f}`, `test/fixtures/${f}`, null]);


const sha = (b) => createHash("sha256").update(b).digest("hex");
const files = {};
const outputs = new Map();
for (const [src, dst, rewrite] of plan) {
  let bytes = show(src);
  if (rewrite) bytes = Buffer.from(rewrite(bytes.toString("utf8")), "utf8");
  if (dst.startsWith("src/runtime/") && dst.endsWith(".ts") && bytes.includes("@tilawa/")) throw new Error(`${src}: unrewritten @tilawa import`);
  outputs.set(dst, bytes);
  files[dst] = sha(bytes);
}
const source = Buffer.from(JSON.stringify({ repo: REPO, commit, files: Object.fromEntries(Object.entries(files).sort()) }, null, 2) + "\n");
outputs.set("src/engine/SOURCE.json", source);

const managed = (dir) => (existsSync(join(pkg, dir)) ? readdirSync(join(pkg, dir), { recursive: true, withFileTypes: true })
  .filter((e) => e.isFile()).map((e) => relative(pkg, join(e.parentPath, e.name))) : []);
const stale = [...managed("src/engine"), ...managed("src/runtime")].filter((p) => !outputs.has(p));

if (check) {
  const bad = [...outputs].filter(([p, b]) => !existsSync(join(pkg, p)) || !readFileSync(join(pkg, p)).equals(b)).map(([p]) => p);
  for (const p of [...bad, ...stale]) console.error(`differs: ${p}`);
  if (bad.length || stale.length) process.exit(1);
  console.log(`${outputs.size} files match ${commit.slice(0, 8)}`);
} else {
  for (const p of stale) rmSync(join(pkg, p));
  for (const [p, b] of outputs) {
    mkdirSync(dirname(join(pkg, p)), { recursive: true });
    writeFileSync(join(pkg, p), b);
  }
  console.log(`wrote ${outputs.size} files from ${commit.slice(0, 8)}${stale.length ? `, removed ${stale.length}` : ""}`);
}
