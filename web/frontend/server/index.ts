import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import { Hono } from "hono";
import { adminApp } from "./admin.js";
import { diagnosticsApp } from "./diagnostics.js";
import { helpApi } from "./help/index.js";
import { reportsApp } from "./reports.js";

const app = new Hono();

// COOP/COEP for SharedArrayBuffer + cache control
app.use("*", async (c, next) => {
  await next();
  const path = new URL(c.req.url).pathname;
  // /help loads Google Fonts cross-origin and needs no SharedArrayBuffer.
  if (path === "/help" || path.startsWith("/help/")) {
    c.header("Cache-Control", "no-cache");
    return;
  }
  c.header("Cross-Origin-Opener-Policy", "same-origin");
  c.header("Cross-Origin-Embedder-Policy", "require-corp");

  if (path === "/sw.js" || path === "/index.html" || path === "/") {
    c.header("Cache-Control", "no-cache, no-store, must-revalidate");
  } else if (path.startsWith("/assets/")) {
    c.header("Cache-Control", "public, max-age=31536000, immutable");
  }
});

// API routes
app.get("/api/health", (c) => c.json({ ok: true }));

// Cache-bust endpoint to clear stale service workers
app.get("/api/clear-cache", (c) => {
  c.header("Clear-Site-Data", '"cache", "storage"');
  return c.json({ cleared: true });
});

// Mount reports API
app.route("/api/reports", reportsApp);
app.route("/api/diagnostics", diagnosticsApp);

// Mount admin dashboard
app.route("/admin", adminApp);

// Mistake-recording site: static page in dist/help, API here.
app.route("/help/api", helpApi);
app.get("/help", (c) => c.redirect("/help/", 301));
app.get("/help/", serveStatic({ root: "./dist", path: "help/index.html" }));

// Serve static files from Vite build
app.use("/*", serveStatic({ root: "./dist" }));

// SPA fallback
app.get("/*", serveStatic({ root: "./dist", path: "index.html" }));

const port = parseInt(process.env.PORT || "5000");
console.log(`Server running on port ${port}`);
serve({ fetch: app.fetch, port });
