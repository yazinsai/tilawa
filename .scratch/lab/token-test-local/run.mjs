import { stripTypeScriptTypes } from "node:module";
import { readFileSync, writeFileSync, mkdirSync, rmSync } from "node:fs";
import { resolve, dirname } from "node:path";
const root = resolve(".scratch/lab/token-test-local");
const files = ["core/src/fuzzy.ts", "core/src/local.ts", "core/src/heldout-token.ts", "core/test/heldout-token.test.ts",
  "core/test/local-token-validation.test.ts", "sdk/src/heldout-token-training.ts", "sdk/test/heldout-token-training.test.ts"];
try {
  for (const file of files) {
    const dest = resolve(root, file.replace(/\.ts$/, ".js"));
    mkdirSync(dirname(dest), { recursive: true });
    const source = readFileSync(resolve("packages", file), "utf8").replace('from "vitest"', `from "${root}/assert.mjs"`);
    writeFileSync(dest, stripTypeScriptTypes(source));
  }
  for (const file of files.filter(f => f.includes("/test/"))) await import(resolve(root, file.replace(/\.ts$/, ".js")));
} finally {
  for (const pkg of ["core", "sdk"]) rmSync(resolve(root, pkg), { recursive: true, force: true });
}
