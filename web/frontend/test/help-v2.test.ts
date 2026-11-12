import { afterAll, describe, expect, test, vi } from "vitest";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { makeReport, makeWav, reportForm, sha256 } from "./help-v2-fixture";

process.env.HELP_ADMIN_TOKEN = "admin-test-token";
const { createHelpV2 } = await import("../server/help/v2");
type Config = Parameters<typeof createHelpV2>[0];

const roots: string[] = [];
afterAll(() => Promise.all(roots.map((r) => rm(r, { recursive: true, force: true }))));

const ORIGIN = "https://tilawa.dev";
const INGEST_KEY = "app-secret-key";
async function setup(patch: Partial<NonNullable<Config>> = {}) {
  const root = await mkdtemp(join(tmpdir(), "tilawa-v2-"));
  roots.push(root);
  const clock = { t: Date.parse("2026-11-11T10:00:00Z") };
  const config = {
    dir: join(root, "v2"),
    legacyDir: root,
    consentVersions: ["2026-11-report-1"],
    allowedOrigins: [ORIGIN],
    ingestKeys: new Map([[sha256(INGEST_KEY), "quran-app"]]),
    reportsPerHour: 20,
    deleteRetentionMs: 30 * 86_400_000,
    minFreeBytes: 0,
    now: () => clock.t,
    ...patch,
  };
  const v2 = createHelpV2(config);
  await v2.ready;
  return { ...v2, config, clock, root };
}

const admin = { authorization: "Bearer admin-test-token" };
const post = (app: any, form: FormData, headers: Record<string, string> = { origin: ORIGIN }) =>
  app.request("/reports", { method: "POST", body: form, headers });

describe("POST /reports", () => {
  test("a valid report gets a receipt; files, index and hashed token are stored", async () => {
    const { app, config } = await setup();
    const wav = makeWav(1500);
    const report = makeReport(wav);
    const res = await post(app, reportForm(report, wav));
    expect(res.status).toBe(201);
    expect(res.headers.get("access-control-allow-origin")).toBe(ORIGIN);
    expect(res.headers.get("cache-control")).toBe("no-store");
    const receipt = await res.json();
    expect(Object.keys(receipt).sort()).toEqual(["deleteToken", "receivedAt", "reportId"]);
    expect(receipt.reportId).toBe(report.reportId);
    expect(Buffer.from(receipt.deleteToken, "base64url")).toHaveLength(32);

    const dir = join(config.dir, "reports", report.reportId);
    expect((await readdir(dir)).sort()).toEqual(["audio.wav", "receipt.json", "report.json"]);
    expect(new Uint8Array(await readFile(join(dir, "audio.wav")))).toEqual(wav);
    const stored = await readFile(join(dir, "receipt.json"), "utf8");
    expect(stored).not.toContain(receipt.deleteToken);
    expect(JSON.parse(stored).deleteTokenSha256).toEqual([sha256(receipt.deleteToken)]);
    expect(await readdir(join(config.dir, "tmp"))).toEqual([]);

    const rows = (await readFile(join(config.dir, "index.jsonl"), "utf8")).trim().split("\n").map((l) => JSON.parse(l));
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      event: "received", reportId: report.reportId, receivedAt: receipt.receivedAt, source: "tilawa.dev/help",
      sha256: report.audio.sha256, durationMs: 1500, incidentKinds: ["false_warning"],
      consent: { version: "2026-11-report-1", send: true },
    });
  });

  test("a retry is idempotent and a different payload for the same id is 409", async () => {
    const { app, config } = await setup();
    const wav = makeWav(1000);
    const report = makeReport(wav);
    const first = await (await post(app, reportForm(report, wav))).json();
    const retry = await post(app, reportForm(report, wav));
    expect(retry.status).toBe(200);
    const again = await retry.json();
    expect(again.reportId).toBe(first.reportId);
    expect(again.receivedAt).toBe(first.receivedAt);

    const otherWav = makeWav(1200);
    const changed = makeReport(otherWav, (r) => (r.reportId = report.reportId));
    expect((await post(app, reportForm(changed, otherWav))).status).toBe(409);
    const sameAudioNewNote = { ...report, incidents: [{ ...report.incidents[0], claim: { answer: "yes" } }] };
    expect((await post(app, reportForm(sameAudioNewNote, wav))).status).toBe(409);

    const rows = (await readFile(join(config.dir, "index.jsonl"), "utf8")).trim().split("\n");
    expect(rows).toHaveLength(1);
    // Both delete tokens work: the first one is still valid after the retry.
    const del = await app.request(`/reports/${report.reportId}`, { method: "DELETE", headers: { "x-delete-token": first.deleteToken } });
    expect(del.status).toBe(200);
    const del2 = await app.request(`/reports/${report.reportId}`, { method: "DELETE", headers: { "x-delete-token": again.deleteToken } });
    expect(del2.status).toBe(200);
    // A deleted id cannot be re-uploaded.
    expect((await post(app, reportForm(report, wav))).status).toBe(409);
  });

  test("invalid WAV, sha, consent and oversize are rejected and nothing is stored", async () => {
    const { app, config } = await setup();
    const errors = async (form: FormData) => {
      const res = await post(app, form);
      return { status: res.status, body: await res.json() };
    };
    const wav = makeWav(1000);

    const stereo = makeWav(1000, { channels: 2 });
    expect(await errors(reportForm(makeReport(stereo), stereo))).toMatchObject({ status: 400, body: { errors: expect.arrayContaining(["wav.header"]) } });
    expect(await errors(reportForm(makeReport(wav, (r) => (r.audio.sha256 = sha256("x"))), wav)))
      .toMatchObject({ status: 400, body: { errors: ["audio.sha256"] } });
    expect(await errors(reportForm(makeReport(wav, (r) => (r.consent.send = false)), wav)))
      .toMatchObject({ status: 400, body: { errors: ["consent.send"] } });
    expect(await errors(reportForm(makeReport(wav, (r) => (r.consent.version = "2026-09-30")), wav)))
      .toMatchObject({ status: 400, body: { errors: ["consent.version"] } });
    const noAudio = new FormData(); noAudio.set("report", JSON.stringify(makeReport(wav)));
    expect((await errors(noAudio)).status).toBe(400);

    const long = makeWav(300_100);
    const oversize = await post(app, reportForm(makeReport(long), long));
    expect(oversize.status).toBe(413);
    const huge = new FormData();
    huge.set("report", "x".repeat(15 * 1024 * 1024));
    huge.set("audio", new Blob([wav]));
    expect((await post(app, huge)).status).toBe(413);

    expect(existsSync(join(config.dir, "index.jsonl"))).toBe(false);
    expect(await readdir(join(config.dir, "reports"))).toEqual([]);
  });

  test("with no consent version allowlisted every report is rejected", async () => {
    const { app } = await setup({ consentVersions: [] });
    const wav = makeWav(500);
    const res = await post(app, reportForm(makeReport(wav), wav));
    expect(res.status).toBe(400);
    expect((await res.json()).errors).toEqual(["consent.version"]);
  });

  test("uploads need an allowlisted origin or a scoped ingestion key", async () => {
    const { app, config } = await setup();
    const wav = makeWav(500);
    expect((await post(app, reportForm(makeReport(wav), wav), {})).status).toBe(403);
    expect((await post(app, reportForm(makeReport(wav), wav), { origin: "https://evil.example" })).status).toBe(403);
    expect((await post(app, reportForm(makeReport(wav), wav), { authorization: "Bearer wrong" })).status).toBe(401);
    expect((await post(app, reportForm(makeReport(wav), wav), admin)).status).toBe(401);
    const report = makeReport(wav, (r) => (r.source = "claims-to-be-tilawa"));
    const res = await post(app, reportForm(report, wav), { authorization: `Bearer ${INGEST_KEY}` });
    expect(res.status).toBe(201);
    const row = JSON.parse((await readFile(join(config.dir, "index.jsonl"), "utf8")).trim());
    expect(row).toMatchObject({ source: "quran-app", claimedSource: "claims-to-be-tilawa", ingest: { via: "key", app: "quran-app" } });
  });

  test("rate limit: a token bucket per IP hash", async () => {
    const { app, clock } = await setup({ reportsPerHour: 3 });
    const wav = makeWav(300);
    const send = (ip: string) => post(app, reportForm(makeReport(wav), wav), { origin: ORIGIN, "cf-connecting-ip": ip });
    for (let i = 0; i < 3; i++) expect((await send("1.1.1.1")).status).toBe(201);
    const limited = await send("1.1.1.1");
    expect(limited.status).toBe(429);
    expect(Number(limited.headers.get("retry-after"))).toBeGreaterThan(0);
    expect((await send("2.2.2.2")).status).toBe(201);
    clock.t += 20 * 60_000; // one token refills every 20 minutes
    expect((await send("1.1.1.1")).status).toBe(201);
    expect((await send("1.1.1.1")).status).toBe(429);
  });

  test("CORS preflight allows the tilawa.dev origin for POST and DELETE only", async () => {
    const { app } = await setup();
    const pre = await app.request("/reports", { method: "OPTIONS", headers: { origin: ORIGIN, "access-control-request-method": "POST" } });
    expect(pre.status).toBe(204);
    expect(pre.headers.get("access-control-allow-origin")).toBe(ORIGIN);
    expect(pre.headers.get("access-control-allow-methods")).toBe("POST");
    expect(pre.headers.get("access-control-allow-credentials")).toBeNull();
    const id = crypto.randomUUID();
    const preDel = await app.request(`/reports/${id}`, { method: "OPTIONS", headers: { origin: ORIGIN, "access-control-request-method": "DELETE" } });
    expect(preDel.headers.get("access-control-allow-methods")).toBe("DELETE");
    expect(preDel.headers.get("access-control-allow-headers")).toContain("X-Delete-Token");
    const evil = await app.request("/reports", { method: "OPTIONS", headers: { origin: "https://evil.example" } });
    expect(evil.status).toBe(403);
    expect(evil.headers.get("access-control-allow-origin")).toBeNull();
    const get = await app.request("/reports", { headers: { origin: ORIGIN, ...admin } });
    expect(get.headers.get("access-control-allow-origin")).toBeNull();
  });
});

describe("admin reads, delete, purge", () => {
  async function withReport() {
    const ctx = await setup();
    const wav = makeWav(800);
    const report = makeReport(wav);
    const receipt = await (await post(ctx.app, reportForm(report, wav))).json();
    return { ...ctx, wav, report, receipt };
  }

  test("anonymous list/get/audio get 401; admin succeeds", async () => {
    const { app, report, wav } = await withReport();
    const id = report.reportId;
    for (const path of ["/reports", `/reports/${id}`, `/reports/${id}/audio`, "/legacy/submissions"]) {
      expect((await app.request(path)).status).toBe(401);
      expect((await app.request(path, { headers: { authorization: "Bearer nope" } })).status).toBe(401);
    }
    expect((await app.request("/admin/purge", { method: "POST" })).status).toBe(401);

    const list = await app.request("/reports", { headers: admin });
    expect(list.status).toBe(200);
    expect((await list.json()).reports.map((r: any) => r.reportId)).toEqual([id]);
    const one = await app.request(`/reports/${id}`, { headers: admin });
    expect(await one.json()).toEqual(report);
    const audio = await app.request(`/reports/${id}/audio`, { headers: admin });
    expect(audio.headers.get("content-type")).toBe("audio/wav");
    expect(audio.headers.get("cross-origin-resource-policy")).toBe("same-origin");
    expect(audio.headers.get("cache-control")).toBe("no-store");
    expect(new Uint8Array(await audio.arrayBuffer())).toEqual(wav);
    expect((await app.request("/reports/not-a-uuid", { headers: admin })).status).toBe(404);
    expect((await app.request(`/reports/${crypto.randomUUID()}/audio`, { headers: admin })).status).toBe(404);
  });

  test("delete: wrong token 403, right token soft-deletes, then GET is 404", async () => {
    const { app, report, receipt, config } = await withReport();
    const id = report.reportId;
    const del = (headers: Record<string, string>) => app.request(`/reports/${id}`, { method: "DELETE", headers });
    expect((await del({})).status).toBe(401);
    expect((await del({ "x-delete-token": "wrong" })).status).toBe(403);
    expect((await app.request("/reports/../x", { method: "DELETE", headers: { "x-delete-token": "a" } })).status).not.toBe(200);
    expect((await app.request(`/reports/${crypto.randomUUID()}`, { method: "DELETE", headers: { "x-delete-token": "a" } })).status).toBe(404);

    const ok = await del({ "x-delete-token": receipt.deleteToken, origin: ORIGIN });
    expect(ok.status).toBe(200);
    expect(ok.headers.get("access-control-allow-origin")).toBe(ORIGIN);
    expect((await del({ "x-delete-token": receipt.deleteToken })).status).toBe(200); // idempotent
    expect((await app.request(`/reports/${id}`, { headers: admin })).status).toBe(404);
    expect((await app.request(`/reports/${id}/audio`, { headers: admin })).status).toBe(404);
    expect((await (await app.request("/reports", { headers: admin })).json()).reports).toEqual([]);
    const all = (await (await app.request("/reports?include=deleted", { headers: admin })).json()).reports;
    expect(all).toMatchObject([{ reportId: id, status: "deleted" }]);
    expect(existsSync(join(config.dir, "reports", id))).toBe(false);
    expect((await readdir(join(config.dir, "deleted", id))).sort()).toEqual(["audio.wav", "deleted.json", "receipt.json", "report.json"]);
  });

  test("an admin bearer can delete without the token", async () => {
    const { app, report } = await withReport();
    const res = await app.request(`/reports/${report.reportId}`, { method: "DELETE", headers: admin });
    expect(res.status).toBe(200);
  });

  test("purge drops deleted audio and report after the retention window only", async () => {
    const { app, report, receipt, config, clock, purgeDeleted } = await withReport();
    const id = report.reportId;
    await app.request(`/reports/${id}`, { method: "DELETE", headers: { "x-delete-token": receipt.deleteToken } });
    const dir = join(config.dir, "deleted", id);
    clock.t += 29 * 86_400_000;
    expect(await purgeDeleted()).toEqual([]);
    expect(existsSync(join(dir, "audio.wav"))).toBe(true);
    clock.t += 2 * 86_400_000;
    const res = await app.request("/admin/purge", { method: "POST", headers: admin });
    expect(await res.json()).toEqual({ purged: [id] });
    expect((await readdir(dir)).sort()).toEqual(["deleted.json", "receipt.json"]);
    expect(JSON.parse(await readFile(join(dir, "deleted.json"), "utf8"))).toMatchObject({ reportId: id, by: "contributor", purgedAt: expect.any(String) });
    expect(await purgeDeleted()).toEqual([]);
    const all = (await (await app.request("/reports?include=deleted", { headers: admin })).json()).reports;
    expect(all).toMatchObject([{ reportId: id, status: "purged" }]);
    expect((await app.request("/admin/purge?retentionDays=-1", { method: "POST", headers: admin })).status).toBe(400);
  });

  test("a restart (fresh module, same storage) keeps data and idempotency, and repairs a lost index row", async () => {
    const { app, report, wav, receipt, config } = await withReport();
    // Simulate a crash between the rename and the index append for a second report.
    const wav2 = makeWav(400);
    const report2 = makeReport(wav2);
    expect((await post(app, reportForm(report2, wav2))).status).toBe(201);
    const indexPath = join(config.dir, "index.jsonl");
    const lines = (await readFile(indexPath, "utf8")).trim().split("\n");
    const { writeFile } = await import("node:fs/promises");
    await writeFile(indexPath, lines[0] + "\n");

    vi.resetModules();
    const fresh = await import("../server/help/v2");
    const v2 = fresh.createHelpV2(config);
    await v2.ready;
    const list = (await (await v2.app.request("/reports", { headers: admin })).json()).reports.map((r: any) => r.reportId);
    expect(list.sort()).toEqual([report.reportId, report2.reportId].sort());
    const retry = await post(v2.app, reportForm(report, wav));
    expect(retry.status).toBe(200);
    expect((await retry.json()).receivedAt).toBe(receipt.receivedAt);
    const other = makeWav(900);
    expect((await post(v2.app, reportForm(makeReport(other, (r) => (r.reportId = report.reportId)), other))).status).toBe(409);
    const del = await v2.app.request(`/reports/${report.reportId}`, { method: "DELETE", headers: { "x-delete-token": receipt.deleteToken } });
    expect(del.status).toBe(200);
  });

  test("concurrent identical uploads store one report", async () => {
    const { app, config } = await setup();
    const wav = makeWav(600);
    const report = makeReport(wav);
    const statuses = (await Promise.all([1, 2, 3].map(() => post(app, reportForm(report, wav))))).map((r: any) => r.status).sort();
    expect(statuses).toEqual([200, 200, 201]);
    expect((await readFile(join(config.dir, "index.jsonl"), "utf8")).trim().split("\n")).toHaveLength(1);
  });
});
