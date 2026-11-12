import { Hono } from "hono";
import { bodyLimit } from "hono/body-limit";
import { isStoredId, requireAdmin } from "./auth.js";
import { randomUUID } from "node:crypto";
import { mkdir, writeFile, readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

const STORAGE_DIR = process.env.STORAGE_DIR || "./storage/reports";

export const reportsApp = new Hono();

// POST /api/reports — accept audio + metadata
reportsApp.post("/", bodyLimit({ maxSize: 25 * 1024 * 1024 }), async (c) => {
  const form = await c.req.formData();
  const audio = form.get("audio") as File | null;
  const metaRaw = form.get("metadata") as string | null;

  if (!audio || !metaRaw) {
    return c.json({ error: "Missing audio or metadata" }, 400);
  }

  let meta: Record<string, unknown>;
  try {
    meta = JSON.parse(metaRaw);
  } catch {
    return c.json({ error: "Invalid metadata JSON" }, 400);
  }

  const id = randomUUID();
  const dir = join(STORAGE_DIR, id);
  await mkdir(dir, { recursive: true });

  // Save audio
  const audioBuffer = Buffer.from(await audio.arrayBuffer());
  await writeFile(join(dir, "audio.wav"), audioBuffer);

  // Save metadata
  const fullMeta = {
    ...meta,
    id,
    timestamp: new Date().toISOString(),
    userAgent: c.req.header("user-agent") || "",
    audioSizeBytes: audioBuffer.length,
  };
  await writeFile(join(dir, "meta.json"), JSON.stringify(fullMeta, null, 2));

  return c.json({ id, status: "saved" }, 201);
});

// GET /api/reports — list all reports
reportsApp.get("/", requireAdmin, async (c) => {
  try {
    const entries = await readdir(STORAGE_DIR);
    const reports = [];
    for (const entry of entries) {
      try {
        const metaPath = join(STORAGE_DIR, entry, "meta.json");
        const raw = await readFile(metaPath, "utf-8");
        reports.push(JSON.parse(raw));
      } catch {
        // Skip broken entries
      }
    }
    // Sort newest first
    reports.sort(
      (a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
    return c.json(reports);
  } catch {
    return c.json([]);
  }
});

// GET /api/reports/:id/audio — stream audio file
reportsApp.get("/:id/audio", requireAdmin, async (c) => {
  const id = c.req.param("id");
  if (!isStoredId(id)) return c.json({ error: "Not found" }, 404);
  const filePath = join(STORAGE_DIR, id, "audio.wav");
  try {
    const data = await readFile(filePath);
    return new Response(data, {
      headers: {
        "Content-Type": "audio/wav",
        "Content-Length": String(data.length),
      },
    });
  } catch {
    return c.json({ error: "Not found" }, 404);
  }
});
