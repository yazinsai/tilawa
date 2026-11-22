import { Hono } from "hono";
import { isStoredId, requireAdmin } from "./auth.js";
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

const STORAGE_DIR = process.env.STORAGE_DIR || "./storage/reports";

// 0.x upload routes are retired: V1 sends reports to /help/api/v2. Stored 0.x data stays readable by admins.
export const reportsApp = new Hono();

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
