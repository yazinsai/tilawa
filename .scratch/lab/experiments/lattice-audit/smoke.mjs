// Lightweight algorithm checks when the supervisor cannot execute. This is not a substitute for Vitest/build.
import { registerHooks } from "node:module";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
const expect = (actual) => ({
  toBe: (expected) => assert.equal(actual, expected),
  toEqual: (expected) => assert.deepEqual(actual, expected),
  toHaveLength: (length) => assert.equal(actual.length, length),
  toBeCloseTo: (expected, places = 2) => assert.ok(Math.abs(actual - expected) < 0.5 * 10 ** -places, `${actual} != ${expected}`),
});
let passed = 0;
globalThis.smokeVitest = { expect, describe: (_name, fn) => fn(), it: (name, fn) => { fn(); passed++; console.log(`PASS ${name}`); } };
const stub = "data:text/javascript,export const {expect,describe,it}=globalThis.smokeVitest;";
registerHooks({ resolve(specifier, context, next) {
  if (specifier === "vitest") return { url: stub, shortCircuit: true };
  if (specifier.endsWith(".js") && context.parentURL?.startsWith("file:")) {
    const url = new URL(specifier, context.parentURL);
    if (!existsSync(fileURLToPath(url))) return next(url.href.slice(0, -3) + ".ts", context);
  }
  return next(specifier, context);
} });
await import("../../../../packages/core/test/ctc-prefix.test.ts");
await import("../../../../packages/core/test/local-acoustic.test.ts");
console.log(`${passed} lightweight algorithm checks passed; full validation remains pending.`);
