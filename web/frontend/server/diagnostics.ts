import { Hono } from "hono";
import { isStoredId, requireAdmin } from "./auth.js";
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

const STORAGE_DIR = process.env.STORAGE_DIR
  ? join(process.env.STORAGE_DIR, "../diagnostics")
  : "./storage/diagnostics";

// 0.x upload route is retired: V1 sends reports to /help/api/v2. Stored 0.x data stays readable by admins.
export const diagnosticsApp = new Hono();

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
