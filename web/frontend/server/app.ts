import { serveStatic } from "@hono/node-server/serve-static";
import { Hono } from "hono";
import { adminApp } from "./admin.js";
import { createAssetsV1 } from "./assets.js";
import { diagnosticsApp } from "./diagnostics.js";
import { helpApi } from "./help/index.js";
import { configFromEnv, createHelpV2 } from "./help/v2.js";
import { reportsApp } from "./reports.js";

// Stored user data and admin pages: never cached, never embeddable cross-origin.
const DATA_PATH = /^\/(?:api\/(?:reports|diagnostics)|admin|help\/api)(?:\/|$)/;
const isHelp = (path: string) => path === "/help" || path.startsWith("/help/");

export function createApp(env = process.env) {
  const app = new Hono();
  const helpV2 = createHelpV2(configFromEnv(env));

  app.use("*", async (c, next) => {
    await next();
    const path = c.req.path;
    c.header("X-Content-Type-Options", "nosniff");
    if (path.startsWith("/v1/")) return; // assets set their own cache/CORP/CORS headers
    const data = DATA_PATH.test(path);
    if (data) {
      c.header("Cache-Control", "no-store");
      c.header("Cross-Origin-Resource-Policy", "same-origin");
    }
    if (isHelp(path)) {
      if (!data) c.header("Cache-Control", "no-cache");
      // Opt-in cross-origin isolation for /help (the V1 SDK runs inference there).
      // /help self-hosts its fonts and bundles the SDK (dist/help + dist/assets), so
      // COEP require-corp has nothing cross-origin to block.
      if (env.HELP_ISOLATION === "1") {
        c.header("Cross-Origin-Opener-Policy", "same-origin");
        c.header("Cross-Origin-Embedder-Policy", "require-corp");
      }
      return;
    }
    // COOP/COEP for SharedArrayBuffer on the demo.
    c.header("Cross-Origin-Opener-Policy", "same-origin");
    c.header("Cross-Origin-Embedder-Policy", "require-corp");
    if (path === "/sw.js" || path === "/index.html" || path === "/") {
      c.header("Cache-Control", "no-cache, no-store, must-revalidate");
    } else if (path.startsWith("/assets/")) {
      c.header("Cache-Control", "public, max-age=31536000, immutable");
    }
  });

  app.get("/api/health", (c) => c.json({ ok: true }));

  // Cache-bust endpoint to clear stale service workers
  app.get("/api/clear-cache", (c) => {
    c.header("Clear-Site-Data", '"cache", "storage"');
    return c.json({ cleared: true });
  });

  app.route("/api/reports", reportsApp);
  app.route("/api/diagnostics", diagnosticsApp);
  app.route("/admin", adminApp);

  // Immutable V1 assets (models / reference / runtime).
  app.route("/v1", createAssetsV1(env.ASSETS_V1_DIR ?? "./dist/v1"));

  // Mistake-recording site: static page in dist/help, API here. v2 first.
  app.route("/help/api/v2", helpV2.app);
  app.route("/help/api", helpApi);
  app.get("/help", (c) => c.redirect("/help/", 301));
  app.get("/help/", serveStatic({ root: "./dist", path: "help/index.html" }));

  // Serve static files from Vite build
  app.use("/*", serveStatic({ root: "./dist" }));

  // SPA fallback
  app.get("/*", serveStatic({ root: "./dist", path: "index.html" }));

  return { app, helpV2 };
}
