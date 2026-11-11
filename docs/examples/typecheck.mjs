#!/usr/bin/env node
// Typechecks docs/examples/*.ts against the built @tilawa/core (packages/core/dist), resolved through the package's
// "exports" map exactly as an app would. Build the package first:
//   (cd packages/core && npm ci && npm run build)
//   node docs/examples/typecheck.mjs
import { spawnSync } from "node:child_process";
import { existsSync, lstatSync, mkdirSync, rmSync, symlinkSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const core = resolve(here, "../../packages/core");
if (!existsSync(join(core, "dist/index.d.ts"))) {
  console.error("packages/core/dist is missing: run `npm ci && npm run build` in packages/core first.");
  process.exit(1);
}
const tsc = join(core, "node_modules/.bin/tsc");
if (!existsSync(tsc)) {
  console.error("TypeScript is missing: run `npm ci` in packages/core first.");
  process.exit(1);
}

// node_modules/@tilawa/core -> packages/core, so imports go through its package.json "exports".
const link = join(here, "node_modules/@tilawa/core");
mkdirSync(dirname(link), { recursive: true });
try {
  if (lstatSync(link)) rmSync(link, { recursive: true, force: true });
} catch {}
symlinkSync(relative(dirname(link), core), link, "dir");

const r = spawnSync(tsc, ["-p", join(here, "tsconfig.json")], { stdio: "inherit" });
if (r.status === 0) console.log("docs/examples: tsc --noEmit passed");
process.exit(r.status ?? 1);
