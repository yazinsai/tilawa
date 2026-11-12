import { Hono } from "hono";
import { bodyLimit } from "hono/body-limit";
import { isStoredId, requireAdmin } from "./auth.js";
import { randomUUID } from "node:crypto";
import { mkdir, writeFile, readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

const STORAGE_DIR = process.env.STORAGE_DIR
  ? join(process.env.STORAGE_DIR, "../diagnostics")
  : "./storage/diagnostics";

export const diagnosticsApp = new Hono();

// POST /api/diagnostics — accept auto-captured diagnostic data
diagnosticsApp.post("/", bodyLimit({ maxSize: 25 * 1024 * 1024 }), async (c) => {
  const form = await c.req.formData();
  const audio = form.get("audio") as File | null;
  const eventsRaw = form.get("events") as string | null;
  const trigger = form.get("trigger") as string | null;

  if (!eventsRaw || !trigger) {
    return c.json({ error: "Missing events or trigger" }, 400);
  }

  let events: unknown[];
  try {
    events = JSON.parse(eventsRaw);
  } catch {
    return c.json({ error: "Invalid events JSON" }, 400);
  }

  const id = randomUUID();
  const dir = join(STORAGE_DIR, id);
  await mkdir(dir, { recursive: true });

  // Save audio if provided
  if (audio) {
    const audioBuffer = Buffer.from(await audio.arrayBuffer());
    await writeFile(join(dir, "audio.wav"), audioBuffer);
  }

  // Save metadata
  const meta = {
    id,
    trigger,
    events,
    timestamp: new Date().toISOString(),
    userAgent: c.req.header("user-agent") || "",
    hasAudio: !!audio,
  };
  await writeFile(join(dir, "meta.json"), JSON.stringify(meta, null, 2));

  return c.json({ id, status: "saved" }, 201);
});

// GET /api/diagnostics — list all diagnostics
diagnosticsApp.get("/", requireAdmin, async (c) => {
  try {
    const entries = await readdir(STORAGE_DIR);
    const items = [];
    for (const entry of entries) {
      try {
        const raw = await readFile(join(STORAGE_DIR, entry, "meta.json"), "utf-8");
        const meta = JSON.parse(raw);
        // Return summary (without full events array for list view)
        items.push({
          id: meta.id,
          trigger: meta.trigger,
          timestamp: meta.timestamp,
          hasAudio: meta.hasAudio,
          eventCount: Array.isArray(meta.events) ? meta.events.length : 0,
        });
      } catch { /* skip */ }
    }
    items.sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
    );
    return c.json(items);
  } catch {
    return c.json([]);
  }
});

// GET /api/diagnostics/:id/audio — stream audio
diagnosticsApp.get("/:id/audio", requireAdmin, async (c) => {
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
