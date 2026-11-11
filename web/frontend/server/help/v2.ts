// /help/api/v2: private collection of Tilawa reports (contract §6-7).
//
// Storage (under HELP_V2_DIR, default $HELP_DATA_DIR/v2 = ./storage/help/v2):
//   reports/<reportId>/{report.json,audio.wav,receipt.json}   live reports
//   deleted/<reportId>/{report.json,audio.wav,receipt.json,deleted.json}
//                                                            soft-deleted; purgeDeleted() drops
//                                                            report.json + audio.wav after retention
//   tmp/                                                     staging for atomic writes
//   index.jsonl                                              append-only event log
// receipt.json holds sha256 hashes only; delete tokens are never stored in clear.
import { createHash, randomBytes, timingSafeEqual } from "node:crypto";
import { existsSync } from "node:fs";
import { appendFile, mkdir, open, readdir, readFile, rename, rm, statfs, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { Hono, type Context, type MiddlewareHandler } from "hono";
import { bodyLimit } from "hono/body-limit";
import { isAdmin, isStoredId, requireAdmin } from "../auth.js";
import { ipHash } from "./ip.js";
import { MAX_REPORT_JSON_BYTES, MAX_WAV_BYTES, validateReport } from "./report-validate.js";

export interface HelpV2Config {
  dir: string;
  legacyDir: string;
  consentVersions: string[];
  allowedOrigins: string[];
  /** sha256(hex) of an ingestion key -> app id recorded as the report's source. */
  ingestKeys: Map<string, string>;
  reportsPerHour: number;
  deleteRetentionMs: number;
  minFreeBytes: number;
  now: () => number;
}

const list = (v: string | undefined) => (v ?? "").split(",").map((s) => s.trim()).filter(Boolean);

/** `HELP_INGEST_KEY_HASHES="appId=<sha256hex>,<sha256hex>"`; a bare hash gets app id `key-<hash[:8]>`. */
export function parseIngestKeys(v: string | undefined): Map<string, string> {
  const keys = new Map<string, string>();
  for (const entry of list(v)) {
    const [app, hash] = entry.includes("=") ? entry.split("=", 2) : [`key-${entry.slice(0, 8)}`, entry];
    if (/^[0-9a-f]{64}$/i.test(hash) && /^[A-Za-z0-9._-]{1,64}$/.test(app)) keys.set(hash.toLowerCase(), app);
  }
  return keys;
}

export function configFromEnv(env = process.env): HelpV2Config {
  const legacyDir = resolve(env.HELP_DATA_DIR ?? "./storage/help");
  return {
    dir: resolve(env.HELP_V2_DIR ?? join(legacyDir, "v2")),
    legacyDir,
    consentVersions: list(env.HELP_REPORT_CONSENT_VERSIONS),
    allowedOrigins: list(env.HELP_ALLOWED_ORIGINS ?? "https://tilawa.dev"),
    ingestKeys: parseIngestKeys(env.HELP_INGEST_KEY_HASHES),
    reportsPerHour: Number(env.HELP_REPORTS_PER_HOUR ?? 20),
    deleteRetentionMs: Number(env.HELP_DELETE_RETENTION_DAYS ?? 30) * 86_400_000,
    minFreeBytes: Number(env.HELP_MIN_FREE_BYTES ?? 1024 ** 3),
    now: Date.now,
  };
}

// Whole multipart body: the largest report.json plus the longest WAV plus form overhead.
export const MAX_BODY_BYTES = MAX_REPORT_JSON_BYTES + MAX_WAV_BYTES + 64 * 1024;

const sha256 = (data: string | Uint8Array) => createHash("sha256").update(data).digest("hex");
const safeEqualHex = (a: string, b: string) => a.length === b.length && timingSafeEqual(Buffer.from(a), Buffer.from(b));

/** In-memory token bucket per key: `capacity` tokens, refilled evenly over an hour. */
export function tokenBucket(capacity: number, now: () => number) {
  const buckets = new Map<string, { tokens: number; at: number }>();
  const rate = capacity / 3_600_000;
  return (key: string): number => {
    const t = now();
    if (buckets.size > 10_000) {
      for (const [k, b] of buckets) if (b.tokens + (t - b.at) * rate >= capacity) buckets.delete(k);
    }
    const b = buckets.get(key) ?? { tokens: capacity, at: t };
    b.tokens = Math.min(capacity, b.tokens + (t - b.at) * rate);
    b.at = t;
    buckets.set(key, b);
    if (b.tokens >= 1) {
      b.tokens -= 1;
      return 0;
    }
    return Math.ceil((1 - b.tokens) / rate / 1000); // seconds until the next token
  };
}

/** Serializes async sections within this process. */
function mutex() {
  let tail: Promise<unknown> = Promise.resolve();
  return <T>(fn: () => Promise<T>): Promise<T> => {
    const run = tail.then(fn, fn);
    tail = run.catch(() => undefined);
    return run;
  };
}

async function writeDurable(path: string, data: string | Uint8Array) {
  const fh = await open(path, "wx");
  try {
    await fh.writeFile(data);
    await fh.sync();
  } finally {
    await fh.close();
  }
}

async function fsyncDir(path: string) {
  try {
    const fh = await open(path, "r");
    try { await fh.sync(); } finally { await fh.close(); }
  } catch {
    // Some platforms refuse fsync on directories; the rename is still atomic.
  }
}

interface StoredReceipt {
  reportId: string;
  receivedAt: string;
  jsonSha256: string;
  audioSha256: string;
  deleteTokenSha256: string[];
  source: string;
  ingest: { via: "origin"; origin: string } | { via: "key"; app: string };
}

const MAX_TOKENS_PER_REPORT = 8;

export function createHelpV2(config: HelpV2Config = configFromEnv()) {
  const dirs = {
    reports: join(config.dir, "reports"),
    deleted: join(config.dir, "deleted"),
    tmp: join(config.dir, "tmp"),
  };
  const indexPath = join(config.dir, "index.jsonl");
  const legacySubmissions = join(config.legacyDir, "submissions.jsonl");
  const locked = mutex();
  const take = tokenBucket(config.reportsPerHour, config.now);
  const iso = () => new Date(config.now()).toISOString();

  const appendIndex = async (row: Record<string, unknown>) => {
    const fh = await open(indexPath, "a");
    try {
      await fh.write(JSON.stringify(row) + "\n");
      await fh.sync();
    } finally {
      await fh.close();
    }
  };

  const readIndex = async (): Promise<Record<string, any>[]> =>
    existsSync(indexPath)
      ? (await readFile(indexPath, "utf8")).split("\n").filter(Boolean).flatMap((l) => {
          try { return [JSON.parse(l)]; } catch { return []; } // a torn last line after a crash
        })
      : [];

  const readReceipt = async (dir: string): Promise<StoredReceipt | null> => {
    try { return JSON.parse(await readFile(join(dir, "receipt.json"), "utf8")); } catch { return null; }
  };

  const receivedRow = (receipt: StoredReceipt, report: any) => ({
    event: "received",
    reportId: receipt.reportId,
    receivedAt: receipt.receivedAt,
    source: receipt.source,
    claimedSource: report.source,
    ingest: receipt.ingest,
    consent: report.consent,
    sha256: receipt.audioSha256,
    durationMs: report.audio?.durationMs,
    incidentKinds: (report.incidents ?? []).map((i: any) => i.kind),
  });

  // Boot: make dirs, drop half-written staging, and re-add index rows for reports
  // that were renamed into place but whose index append was lost.
  const ready = (async () => {
    for (const d of Object.values(dirs)) await mkdir(d, { recursive: true });
    for (const name of await readdir(dirs.tmp)) await rm(join(dirs.tmp, name), { recursive: true, force: true });
    const indexed = new Set((await readIndex()).filter((r) => r.event === "received").map((r) => r.reportId));
    for (const id of await readdir(dirs.reports)) {
      if (indexed.has(id) || !isStoredId(id)) continue;
      const receipt = await readReceipt(join(dirs.reports, id));
      if (!receipt) continue;
      const report = JSON.parse(await readFile(join(dirs.reports, id, "report.json"), "utf8"));
      await appendIndex(receivedRow(receipt, report));
    }
  })();

  const newToken = () => {
    const token = randomBytes(32).toString("base64url");
    return { token, hash: sha256(token) };
  };

  const freeBytes = async () => {
    try {
      const s = await statfs(config.dir);
      return s.bavail * s.bsize;
    } catch {
      return Infinity;
    }
  };

  const app = new Hono();

  // Data responses: never cached, never embeddable cross-origin, never sniffed.
  app.use("*", async (c, next) => {
    await next();
    c.header("Cache-Control", "no-store");
    c.header("X-Content-Type-Options", "nosniff");
    c.header("Cross-Origin-Resource-Policy", "same-origin");
  });

  // CORS for POST/DELETE from allowlisted origins only; no credentials.
  const originAllowed = (c: Context) => {
    const origin = c.req.header("origin");
    return origin && config.allowedOrigins.includes(origin) ? origin : null;
  };
  const preflight = (methods: string) => (c: Context) => {
    const origin = originAllowed(c);
    if (!origin) return c.body(null, 403);
    return c.body(null, 204, {
      "Access-Control-Allow-Origin": origin,
      "Access-Control-Allow-Methods": methods,
      "Access-Control-Allow-Headers": "Content-Type, X-Delete-Token",
      "Access-Control-Max-Age": "600",
      Vary: "Origin",
    });
  };
  const corsResponse: MiddlewareHandler = async (c, next) => {
    await next();
    const origin = originAllowed(c);
    if (origin) {
      c.header("Access-Control-Allow-Origin", origin);
      c.header("Vary", "Origin");
    }
  };
  app.options("/reports", preflight("POST"));
  app.options("/reports/:id", preflight("DELETE"));

  app.post(
    "/reports",
    corsResponse,
    async (c, next) => {
      const wait = take(ipHash(c));
      if (wait) return c.json({ error: "rate_limited" }, 429, { "Retry-After": String(wait) });
      await next();
    },
    bodyLimit({ maxSize: MAX_BODY_BYTES, onError: (c) => c.json({ error: "too_large" }, 413) }),
    async (c) => {
      // Who is sending: a scoped ingestion key, or a browser on an allowlisted origin.
      let ingest: StoredReceipt["ingest"];
      const auth = c.req.header("authorization");
      if (auth !== undefined) {
        const app = auth.startsWith("Bearer ") ? config.ingestKeys.get(sha256(auth.slice(7))) : undefined;
        if (!app) return c.json({ error: "unauthorized" }, 401);
        ingest = { via: "key", app };
      } else {
        const origin = originAllowed(c);
        if (!origin) return c.json({ error: "origin_not_allowed" }, 403);
        ingest = { via: "origin", origin };
      }

      let form: FormData;
      try {
        form = await c.req.formData();
      } catch (err) {
        // A streamed body over the limit surfaces here as the parse failure.
        const e = err as { name?: string; cause?: { name?: string } };
        if (e?.name === "BodyLimitError" || e?.cause?.name === "BodyLimitError") return c.json({ error: "too_large" }, 413);
        return c.json({ error: "invalid", errors: ["multipart"] }, 400);
      }
      const reportPart = form.get("report");
      const audioPart = form.get("audio");
      if (reportPart === null || !(audioPart instanceof Blob)) return c.json({ error: "invalid", errors: ["multipart"] }, 400);
      if (audioPart.size > MAX_WAV_BYTES) return c.json({ error: "invalid", errors: ["audio.max"] }, 413);
      const json = typeof reportPart === "string"
        ? new TextEncoder().encode(reportPart)
        : new Uint8Array(await reportPart.arrayBuffer());
      const wav = new Uint8Array(await audioPart.arrayBuffer());

      const result = validateReport(json, wav, { consentVersions: config.consentVersions });
      if (!result.ok) return c.json({ error: "invalid", errors: result.errors }, 400);
      const report = result.report;
      const id = report.reportId;
      const jsonSha256 = sha256(json);
      const audioSha256 = report.audio.sha256;

      await ready;
      return locked(async () => {
        const live = join(dirs.reports, id);
        const existing = await readReceipt(live);
        if (existing) {
          if (existing.jsonSha256 !== jsonSha256 || existing.audioSha256 !== audioSha256) {
            return c.json({ error: "conflict", reportId: id }, 409);
          }
          // Same payload again (the client lost our reply). Only hashes are stored,
          // so hand out a fresh delete token; earlier tokens stay valid.
          const { token, hash } = newToken();
          const updated = { ...existing, deleteTokenSha256: [...existing.deleteTokenSha256, hash].slice(-MAX_TOKENS_PER_REPORT) };
          const staged = join(dirs.tmp, `${id}.receipt.${randomBytes(4).toString("hex")}`);
          await writeDurable(staged, JSON.stringify(updated));
          await rename(staged, join(live, "receipt.json"));
          await fsyncDir(live);
          return c.json({ reportId: id, receivedAt: existing.receivedAt, deleteToken: token }, 200);
        }
        if (existsSync(join(dirs.deleted, id))) return c.json({ error: "deleted", reportId: id }, 409);
        if ((await freeBytes()) < config.minFreeBytes) return c.json({ error: "storage_full" }, 507);

        const { token, hash } = newToken();
        const receipt: StoredReceipt = {
          reportId: id,
          receivedAt: iso(),
          jsonSha256,
          audioSha256,
          deleteTokenSha256: [hash],
          source: ingest.via === "key" ? ingest.app : report.source,
          ingest,
        };
        const staging = join(dirs.tmp, `${id}.${randomBytes(4).toString("hex")}`);
        await mkdir(staging);
        try {
          await writeDurable(join(staging, "report.json"), json);
          await writeDurable(join(staging, "audio.wav"), wav);
          await writeDurable(join(staging, "receipt.json"), JSON.stringify(receipt));
          await fsyncDir(staging);
          await rename(staging, live);
          await fsyncDir(dirs.reports);
        } catch (err) {
          await rm(staging, { recursive: true, force: true });
          throw err;
        }
        await appendIndex(receivedRow(receipt, report));
        return c.json({ reportId: id, receivedAt: receipt.receivedAt, deleteToken: token }, 201);
      });
    },
  );

  const markDeleted = async (id: string, by: "contributor" | "admin") => {
    const deletedAt = iso();
    await rename(join(dirs.reports, id), join(dirs.deleted, id));
    await fsyncDir(dirs.deleted);
    await writeFile(join(dirs.deleted, id, "deleted.json"), JSON.stringify({ reportId: id, deletedAt, by }));
    await appendIndex({ event: "deleted", reportId: id, deletedAt, by });
    return deletedAt;
  };

  app.delete("/reports/:id", corsResponse, async (c) => {
    const id = c.req.param("id");
    if (!isStoredId(id)) return c.json({ error: "invalid_id" }, 400);
    const admin = isAdmin(c);
    const token = c.req.header("x-delete-token");
    if (!admin && !token) return c.json({ error: "unauthorized" }, 401);
    await ready;
    return locked(async () => {
      const liveReceipt = await readReceipt(join(dirs.reports, id));
      const goneReceipt = liveReceipt ? null : await readReceipt(join(dirs.deleted, id));
      const receipt = liveReceipt ?? goneReceipt;
      if (!receipt) return c.json({ error: "not_found" }, 404);
      const tokenOk = !!token && receipt.deleteTokenSha256.some((h) => safeEqualHex(h, sha256(token)));
      if (!admin && !tokenOk) return c.json({ error: "forbidden" }, 403);
      if (!liveReceipt) return c.json({ reportId: id, deleted: true }, 200);
      const deletedAt = await markDeleted(id, tokenOk ? "contributor" : "admin");
      return c.json({ reportId: id, deleted: true, deletedAt }, 200);
    });
  });

  /** Drops report.json and audio.wav of reports deleted more than `retentionMs` ago. Admin-invoked. */
  const purgeDeleted = async (retentionMs = config.deleteRetentionMs) => {
    await ready;
    return locked(async () => {
      const purged: string[] = [];
      for (const id of await readdir(dirs.deleted)) {
        const dir = join(dirs.deleted, id);
        let marker: { deletedAt: string; purgedAt?: string; [k: string]: unknown };
        try { marker = JSON.parse(await readFile(join(dir, "deleted.json"), "utf8")); } catch { continue; }
        if (marker.purgedAt || config.now() - Date.parse(marker.deletedAt) < retentionMs) continue;
        await rm(join(dir, "audio.wav"), { force: true });
        await rm(join(dir, "report.json"), { force: true });
        const purgedAt = iso();
        await writeFile(join(dir, "deleted.json"), JSON.stringify({ ...marker, purgedAt }));
        await appendIndex({ event: "purged", reportId: id, purgedAt });
        purged.push(id);
      }
      return purged;
    });
  };

  // Admin reads. Everything below needs the admin bearer or the /admin session.
  app.get("/reports", requireAdmin, async (c) => {
    await ready;
    const byId = new Map<string, Record<string, any>>();
    for (const row of await readIndex()) {
      if (row.event === "received") byId.set(row.reportId, { ...row, event: undefined, status: "live" });
      else if (byId.has(row.reportId)) byId.get(row.reportId)!.status = row.event; // deleted | purged
    }
    const all = c.req.query("include") === "deleted";
    const reports = [...byId.values()].filter((r) => all || r.status === "live");
    return c.json({ reports });
  });

  const liveFile = async (c: Context, name: string) => {
    const id = c.req.param("id") ?? "";
    if (!isStoredId(id)) return null;
    await ready;
    try { return await readFile(join(dirs.reports, id, name)); } catch { return null; }
  };
  app.get("/reports/:id", requireAdmin, async (c) => {
    const body = await liveFile(c, "report.json");
    return body ? c.body(body, 200, { "Content-Type": "application/json" }) : c.json({ error: "not_found" }, 404);
  });
  app.get("/reports/:id/audio", requireAdmin, async (c) => {
    const body = await liveFile(c, "audio.wav");
    return body ? c.body(body, 200, { "Content-Type": "audio/wav" }) : c.json({ error: "not_found" }, 404);
  });

  app.post("/admin/purge", requireAdmin, async (c) => {
    const days = c.req.query("retentionDays");
    if (days !== undefined && !(Number(days) >= 0)) return c.json({ error: "invalid_retention" }, 400);
    const purged = await purgeDeleted(days === undefined ? undefined : Number(days) * 86_400_000);
    return c.json({ purged });
  });

  // Historical v1 rows, byte-for-byte (each keeps its original `consent` version).
  app.get("/legacy/submissions", requireAdmin, async (c) => {
    const body = existsSync(legacySubmissions) ? await readFile(legacySubmissions) : Buffer.alloc(0);
    return c.body(body, 200, { "Content-Type": "application/x-ndjson" });
  });

  return { app, ready, purgeDeleted, config };
}
