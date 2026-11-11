// Immutable V1 assets (models, reference corpora, runtime) under /v1/.
// Files are never overwritten: a new version gets a new name (contract §5).
// They come from ASSETS_V1_DIR (default ./dist/v1), populated at build/deploy
// time, never from git.
import { createReadStream, statSync } from "node:fs";
import { extname, join, resolve, sep } from "node:path";
import { Readable } from "node:stream";
import { Hono, type Context } from "hono";

const MIME: Record<string, string> = {
  ".onnx": "application/octet-stream",
  ".wasm": "application/wasm",
  ".mjs": "text/javascript; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
};

const SHARED = {
  "Access-Control-Allow-Origin": "*",
  "Cross-Origin-Resource-Policy": "cross-origin",
  "X-Content-Type-Options": "nosniff",
};

/** Parses a single `bytes=` range against `size`; null = ignore, "bad" = 416. */
function parseRange(header: string, size: number): { start: number; end: number } | null | "bad" {
  const m = /^bytes=(\d*)-(\d*)$/.exec(header.trim());
  if (!m || (m[1] === "" && m[2] === "")) return null; // multi-range or junk: serve the whole file
  let start: number, end: number;
  if (m[1] === "") {
    const suffix = Number(m[2]);
    if (suffix === 0) return "bad";
    start = Math.max(0, size - suffix);
    end = size - 1;
  } else {
    start = Number(m[1]);
    end = m[2] === "" ? size - 1 : Math.min(Number(m[2]), size - 1);
  }
  return start > end || start >= size ? "bad" : { start, end };
}

export function createAssetsV1(dir = process.env.ASSETS_V1_DIR ?? "./dist/v1") {
  const root = resolve(dir);
  const app = new Hono();

  const serve = (c: Context) => {
    let rel: string;
    try {
      rel = decodeURIComponent(c.req.path.replace(/^\/v1\//, ""));
    } catch {
      return c.body(null, 404, { ...SHARED, "Cache-Control": "no-store" });
    }
    const path = resolve(join(root, rel));
    const parts = rel.split(/[\\/]/);
    const stats = (() => {
      try { return statSync(path); } catch { return null; }
    })();
    if (!path.startsWith(root + sep) || parts.some((p) => p === ".." || p.startsWith(".")) || !stats?.isFile()) {
      return c.body(null, 404, { ...SHARED, "Cache-Control": "no-store" });
    }
    const size = stats.size;
    const headers: Record<string, string> = {
      ...SHARED,
      "Content-Type": MIME[extname(path).toLowerCase()] ?? "application/octet-stream",
      // Versioned files never change; the top-level index grows as versions are added.
      "Cache-Control": rel === "manifest.json" ? "no-cache" : "public, max-age=31536000, immutable",
      "Accept-Ranges": "bytes",
      "Access-Control-Expose-Headers": "Content-Length, Content-Range, Accept-Ranges",
      "Last-Modified": stats.mtime.toUTCString(),
    };
    const rangeHeader = c.req.header("range");
    const range = rangeHeader ? parseRange(rangeHeader, size) : null;
    if (range === "bad") return c.body(null, 416, { ...headers, "Content-Range": `bytes */${size}` });
    const { start, end } = range ?? { start: 0, end: size - 1 };
    headers["Content-Length"] = String(size === 0 ? 0 : end - start + 1);
    if (range) headers["Content-Range"] = `bytes ${start}-${end}/${size}`;
    const status = range ? 206 : 200;
    if (c.req.method === "HEAD" || size === 0) return c.body(null, status, headers);
    const stream = Readable.toWeb(createReadStream(path, { start, end })) as ReadableStream;
    return c.body(stream, status, headers);
  };

  app.on(["GET", "HEAD"], "/*", serve);
  app.options("/*", (c) =>
    c.body(null, 204, {
      ...SHARED,
      "Access-Control-Allow-Methods": "GET, HEAD",
      "Access-Control-Allow-Headers": "Range",
      "Access-Control-Max-Age": "86400",
    }),
  );
  app.all("/*", (c) => c.body(null, 405, { ...SHARED, Allow: "GET, HEAD, OPTIONS" }));
  return app;
}
