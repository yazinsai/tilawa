import { afterAll, beforeAll, expect, test } from "vitest";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Hono } from "hono";

const root = await mkdtemp(join(tmpdir(), "tilawa-auth-"));
const id = "0b0c6a3e-2f4b-4f43-9d8c-3b7e8b1f2a10";
process.env.STORAGE_DIR = join(root, "reports");
process.env.ADMIN_PASSWORD = "test-password";
process.env.HELP_ADMIN_TOKEN = "test-token";

const { reportsApp } = await import("../server/reports");
const { diagnosticsApp } = await import("../server/diagnostics");
const { adminApp } = await import("../server/admin");
const app = new Hono();
app.route("/api/reports", reportsApp);
app.route("/api/diagnostics", diagnosticsApp);
app.route("/admin", adminApp);

beforeAll(async () => {
  for (const kind of ["reports", "diagnostics"]) {
    await mkdir(join(root, kind, id), { recursive: true });
    await writeFile(join(root, kind, id, "audio.wav"), new Uint8Array(64));
    await writeFile(join(root, kind, id, "meta.json"),
      JSON.stringify({ id, timestamp: new Date().toISOString(), notes: "<script>x</script>", trigger: "surah_jump" }));
  }
});
afterAll(() => rm(root, { recursive: true, force: true }));

const bearer = { authorization: "Bearer test-token" };

for (const base of ["/api/reports", "/api/diagnostics"]) {
  test(`${base} list and audio are closed to anonymous requests`, async () => {
    expect((await app.request(base)).status).toBe(401);
    expect((await app.request(`${base}/${id}/audio`)).status).toBe(401);
    expect((await app.request(base, { headers: { authorization: "Bearer wrong" } })).status).toBe(401);
  });
  test(`${base} list and audio open with the admin token`, async () => {
    expect((await app.request(base, { headers: bearer })).status).toBe(200);
    expect((await app.request(`${base}/${id}/audio`, { headers: bearer })).status).toBe(200);
    expect((await app.request(`${base}/..%2F..%2Fetc/audio`, { headers: bearer })).status).toBe(404);
  });
}

test("forged legacy cookie is rejected; password login yields a working signed session", async () => {
  expect((await app.request("/api/reports", { headers: { cookie: "admin_auth=1" } })).status).toBe(401);
  expect((await app.request("/admin", { headers: { cookie: "admin_auth=1" } })).status).toBe(302);

  const bad = new FormData(); bad.set("password", "nope");
  const badRes = await app.request("/admin/login", { method: "POST", body: bad });
  expect(badRes.headers.get("set-cookie")).toBeNull();

  const form = new FormData(); form.set("password", "test-password");
  const res = await app.request("/admin/login", { method: "POST", body: form });
  const cookie = res.headers.get("set-cookie")!.split(";")[0];
  expect(cookie).toMatch(/^admin_session=\d+\.[0-9a-f]{64}$/);
  expect((await app.request("/api/reports", { headers: { cookie } })).status).toBe(200);
  const page = await app.request("/admin", { headers: { cookie } });
  expect(page.status).toBe(200);
  const html = await page.text();
  expect(html).not.toContain("<script>x</script>");
  expect(html).toContain("&lt;script&gt;");

  const tampered = cookie.replace(/.$/, (ch) => (ch === "0" ? "1" : "0"));
  expect((await app.request("/api/reports", { headers: { cookie: tampered } })).status).toBe(401);
});

test("submissions stay open and cannot choose their stored id", async () => {
  const form = new FormData();
  form.set("audio", new Blob([new Uint8Array(32)], { type: "audio/wav" }), "a.wav");
  form.set("metadata", JSON.stringify({ id: "../../evil", surah: 1, ayah: 1 }));
  const res = await app.request("/api/reports", { method: "POST", body: form });
  expect(res.status).toBe(201);
  expect((await res.json()).id).toMatch(/^[0-9a-f-]{36}$/);
});
