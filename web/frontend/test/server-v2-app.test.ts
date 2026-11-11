// Whole-app wiring: legacy /help/api untouched, v2 mounted, headers and /v1 assets.
import { afterAll, beforeAll, expect, test } from "vitest";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { makeReport, makeWav, reportForm } from "./help-v2-fixture";

const root = await mkdtemp(join(tmpdir(), "tilawa-app-"));
const helpDir = join(root, "help");
const assets = join(root, "v1");
const legacyRows = [
  '{"id":"old1","file":"audio/old1.wav","consent":"2026-09-30","speaker":"spk_legacy1","seed":1}',
  '{"id":"old2","file":"audio/old2.webm","consent":"2026-09-30","speaker":"spk_legacy2","seed":2,"note":"ünïcode"}',
].join("\n") + "\n";
const legacySubscribers = '{"email":"a@example.com","lang":"en","ts":"2026-10-01T00:00:00.000Z"}\n';

Object.assign(process.env, {
  HELP_DATA_DIR: helpDir,
  STORAGE_DIR: join(root, "reports"),
  HELP_ADMIN_TOKEN: "app-admin-token",
  HELP_PROMPT_SECRET: "app-test-secret",
  HELP_REPORT_CONSENT_VERSIONS: "2026-11-report-1",
  QURAN_JSON: fileURLToPath(new URL("../public/quran.json", import.meta.url)),
});
await mkdir(join(helpDir, "audio"), { recursive: true });
await writeFile(join(helpDir, "submissions.jsonl"), legacyRows);
await writeFile(join(helpDir, "subscribers.jsonl"), legacySubscribers);
await mkdir(join(assets, "models"), { recursive: true });
await writeFile(join(assets, "models", "m.onnx"), Buffer.from("0123456789abcdef"));
await writeFile(join(assets, "ort.wasm"), Buffer.from([0, 0x61, 0x73, 0x6d]));
await writeFile(join(assets, "ort.mjs"), "export {}");
await writeFile(join(assets, "ref-v1.json"), "{}");

const { createApp } = await import("../server/app");
const { app } = createApp({ ...process.env, ASSETS_V1_DIR: assets });
const isolated = createApp({ ...process.env, ASSETS_V1_DIR: assets, HELP_ISOLATION: "1" }).app;
const admin = { authorization: "Bearer app-admin-token" };

beforeAll(() => {});
afterAll(() => rm(root, { recursive: true, force: true }));

test("legacy v1 submit still works and earlier rows stay byte-for-byte", async () => {
  const prompt = await (await app.request("/help/api/prompt")).json();
  const params = new URLSearchParams({
    seed: String(prompt.seed), speaker: "spk_new_one", consent: "2026-09-30", followed: "1", duration: "4",
    device: "phone", gender: "", level: "", v: "4", token: prompt.token,
  });
  const res = await app.request(`/help/api/submit?${params}`, { method: "POST", headers: { "content-type": "audio/wav" }, body: new Uint8Array(5000) });
  expect(res.status).toBe(200);
  const after = await readFile(join(helpDir, "submissions.jsonl"));
  expect(after.subarray(0, Buffer.byteLength(legacyRows)).equals(Buffer.from(legacyRows))).toBe(true);
  const added = after.subarray(Buffer.byteLength(legacyRows)).toString().trim().split("\n");
  expect(added).toHaveLength(1);
  expect(JSON.parse(added[0])).toMatchObject({ speaker: "spk_new_one", consent: "2026-09-30" });
  expect(await readFile(join(helpDir, "subscribers.jsonl"), "utf8")).toBe(legacySubscribers);
  expect((await app.request("/help/api/config")).status).toBe(200);
  expect((await app.request("/help/api/export", { headers: admin })).status).toBe(200);
});

test("read-only legacy export through v2 keeps each row's consent", async () => {
  expect((await app.request("/help/api/v2/legacy/submissions")).status).toBe(401);
  const res = await app.request("/help/api/v2/legacy/submissions", { headers: admin });
  expect(res.status).toBe(200);
  expect(res.headers.get("content-type")).toBe("application/x-ndjson");
  const body = Buffer.from(await res.arrayBuffer());
  expect(body.subarray(0, Buffer.byteLength(legacyRows)).equals(Buffer.from(legacyRows))).toBe(true);
  expect(body.toString().trim().split("\n").map((l) => JSON.parse(l).consent)).toEqual(["2026-09-30", "2026-09-30", "2026-09-30"]);
  expect((await app.request("/help/api/v2/legacy/submissions", { method: "POST", headers: admin })).status).toBe(404);
});

test("v2 is mounted under /help/api/v2 with data headers; legacy and v2 do not collide", async () => {
  const wav = makeWav(700);
  const report = makeReport(wav);
  const res = await app.request("/help/api/v2/reports", { method: "POST", body: reportForm(report, wav), headers: { origin: "https://tilawa.dev" } });
  expect(res.status).toBe(201);
  for (const r of [res, await app.request("/help/api/v2/reports", { headers: admin })]) {
    expect(r.headers.get("cache-control")).toBe("no-store");
    expect(r.headers.get("x-content-type-options")).toBe("nosniff");
    expect(r.headers.get("cross-origin-resource-policy")).toBe("same-origin");
  }
  expect((await app.request("/help/api/v2/reports")).status).toBe(401);
  expect((await readFile(join(helpDir, "v2", "index.jsonl"), "utf8"))).toContain(report.reportId);
});

test("admin and data routes are no-store; /help isolation is opt-in", async () => {
  for (const path of ["/api/reports", "/api/diagnostics", "/help/api/export"]) {
    const r = await app.request(path);
    expect(r.status).toBe(401);
    expect(r.headers.get("cache-control")).toBe("no-store");
    expect(r.headers.get("cross-origin-resource-policy")).toBe("same-origin");
  }
  const login = await app.request("/admin/login");
  expect(login.headers.get("cache-control")).toBe("no-store");
  const help = await app.request("/help");
  expect(help.headers.get("cross-origin-embedder-policy")).toBeNull();
  expect(help.headers.get("cache-control")).toBe("no-cache");
  const helpIso = await isolated.request("/help");
  expect(helpIso.headers.get("cross-origin-opener-policy")).toBe("same-origin");
  expect(helpIso.headers.get("cross-origin-embedder-policy")).toBe("require-corp");
  const home = await app.request("/api/health");
  expect(home.headers.get("cross-origin-embedder-policy")).toBe("require-corp");
});

test("/v1 assets: immutable, CORS *, CORP cross-origin, MIME, Range", async () => {
  const r = await app.request("/v1/models/m.onnx");
  expect(r.status).toBe(200);
  expect(r.headers.get("cache-control")).toBe("public, max-age=31536000, immutable");
  expect(r.headers.get("access-control-allow-origin")).toBe("*");
  expect(r.headers.get("cross-origin-resource-policy")).toBe("cross-origin");
  expect(r.headers.get("content-type")).toBe("application/octet-stream");
  expect(r.headers.get("accept-ranges")).toBe("bytes");
  expect(await r.text()).toBe("0123456789abcdef");
  expect((await app.request("/v1/ort.wasm")).headers.get("content-type")).toBe("application/wasm");
  expect((await app.request("/v1/ort.mjs")).headers.get("content-type")).toMatch(/^text\/javascript/);
  expect((await app.request("/v1/ref-v1.json")).headers.get("content-type")).toMatch(/^application\/json/);

  const part = await app.request("/v1/models/m.onnx", { headers: { range: "bytes=2-5" } });
  expect(part.status).toBe(206);
  expect(part.headers.get("content-range")).toBe("bytes 2-5/16");
  expect(await part.text()).toBe("2345");
  const suffix = await app.request("/v1/models/m.onnx", { headers: { range: "bytes=-3" } });
  expect(await suffix.text()).toBe("def");
  expect((await app.request("/v1/models/m.onnx", { headers: { range: "bytes=99-" } })).status).toBe(416);
  const head = await app.request("/v1/models/m.onnx", { method: "HEAD" });
  expect(head.headers.get("content-length")).toBe("16");
  const pre = await app.request("/v1/models/m.onnx", { method: "OPTIONS", headers: { origin: "https://x.example" } });
  expect(pre.headers.get("access-control-allow-origin")).toBe("*");

  // "/v1/%2e%2e/..." is normalized by URL parsing to a non-/v1 path; it must still not leak data.
  expect((await app.request("/v1/%2e%2e/help/submissions.jsonl")).status).toBe(404);
  for (const miss of ["/v1/nope.onnx", "/v1/models/", "/v1/..%2Fhelp%2Fsubmissions.jsonl", "/v1/models%2F..%2F..%2Fhelp%2Fsubmissions.jsonl"]) {
    const m = await app.request(miss);
    expect(m.status, miss).toBe(404);
    expect(m.headers.get("cache-control")).toBe("no-store");
  }
  expect((await app.request("/v1/models/m.onnx", { method: "PUT" })).status).toBe(405);
});
