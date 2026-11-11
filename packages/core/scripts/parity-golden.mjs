#!/usr/bin/env node
/**
 * Writes test/fixtures/parity/golden.json: the research session's events, at the pinned commit, for the parity
 * scenarios (test/fixtures/parity/scenarios.ts).
 *
 *   node scripts/parity-golden.mjs --repo <research checkout> [--commit <sha>]
 *
 * The research tree is `git archive`d into a temp dir; its workspace isn't installed. Its sources are bundled with
 * esbuild (`@tilawa/core` aliased to the archived core) and run on this package's onnxruntime-node, so the only
 * difference from the public session is the code under test.
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";
import { COMMIT } from "./pin.mjs";

const pkg = join(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const opt = (n) => (args.includes(n) ? args[args.indexOf(n) + 1] : undefined);
const repo = opt("--repo") ?? process.env.TILAWA_RESEARCH_REPO;
const commit = opt("--commit") ?? COMMIT;
if (!repo) {
  console.error("usage: node scripts/parity-golden.mjs --repo <research checkout> [--commit <sha>]");
  process.exit(2);
}

const tmp = mkdtempSync(join(tmpdir(), "tilawa-parity-"));
try {
  const tar = execFileSync("git", ["-C", repo, "archive", commit, "packages/core/src", "packages/core/assets", "packages/sdk/src", "packages/sdk/test/fixtures"], { maxBuffer: 256 << 20 });
  execFileSync("tar", ["-x", "-C", tmp], { input: tar });
  const r = (p) => join(tmp, p).replaceAll("\\", "/");
  const driver = join(tmp, "driver.ts");
  writeFileSync(driver, `
import { readFileSync, writeFileSync } from "node:fs";
import { loadReferenceIndex } from "@tilawa/core";
import { syntheticFixture } from "${r("packages/core/src/fixtures/index.ts")}";
import { createEngine } from "${r("packages/sdk/src/session.ts")}";
import { loadNodeModel, readWav } from "${r("packages/sdk/src/node.ts")}";
import { runScenarios } from "${join(pkg, "test/fixtures/parity/scenarios.ts").replaceAll("\\", "/")}";
const asset = (p: string) => JSON.parse(readFileSync("${r("packages/core/assets")}/" + p, "utf8"));
const index = loadReferenceIndex(asset("ref-v2.json"));
const engine = createEngine({ reference: index, phrases: asset("phrases-v1.json"), model: await loadNodeModel("${r("packages/sdk/test/fixtures/tiny")}"), recordEvidence: true });
const wav = await readWav("${r("packages/sdk/test/fixtures/tlog-clean-00010-630.wav")}");
const results = await runScenarios({
  start: (p, policy) => (p ? engine.startMemorization(p, policy) : engine.startFreeRecitation(policy)) as never,
  wav, index,
  synthetic: (spec) => syntheticFixture(index, spec as never).evidence,
});
writeFileSync(process.argv[2], JSON.stringify({ commit: "${commit}", results }) + "\\n");
`);
  const outDir = join(pkg, "node_modules/.cache/tilawa-parity");
  mkdirSync(outDir, { recursive: true });
  const bundle = join(outDir, "research-driver.mjs");
  await build({
    entryPoints: [driver], bundle: true, platform: "node", format: "esm", target: "node20", outfile: bundle,
    alias: { "@tilawa/core": r("packages/core/src/index.ts") }, external: ["onnxruntime-node"], logLevel: "warning",
  });
  const out = join(pkg, "test/fixtures/parity/golden.json");
  execFileSync(process.execPath, [bundle, out], { stdio: "inherit", cwd: pkg });
  console.log(`wrote ${out} from ${commit.slice(0, 8)}`);
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
