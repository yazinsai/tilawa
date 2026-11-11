import { createReadStream, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defaultClientConditions, defineConfig, type Plugin } from "vite";

const root = path.dirname(fileURLToPath(import.meta.url));

/**
 * Dev server only: serve the prepared V1 assets (`npm run assets:v1` → dist/v1) at /v1/, as server/assets.ts does
 * in production. `vite build` empties dist/, so prepare the assets after building.
 */
function v1Assets(): Plugin {
  const dir = path.resolve(root, process.env.ASSETS_V1_DIR ?? "dist/v1");
  return {
    name: "tilawa-v1-assets",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/v1", (req, res, next) => {
        const file = path.resolve(dir, decodeURIComponent((req.url ?? "/").split("?")[0]).replace(/^\/+/, ""));
        if (!file.startsWith(dir + path.sep)) return next();
        let size: number;
        try {
          const s = statSync(file);
          if (!s.isFile()) return next();
          size = s.size;
        } catch {
          return next();
        }
        const type = file.endsWith(".wasm") ? "application/wasm" : file.endsWith(".mjs") ? "text/javascript" : file.endsWith(".json") ? "application/json" : "application/octet-stream";
        res.setHeader("Content-Type", type);
        res.setHeader("Content-Length", String(size));
        res.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
        createReadStream(file).pipe(res);
      });
    },
  };
}

export default defineConfig({
  plugins: [v1Assets()],
  resolve: {
    // No alias: @tilawa/core and its subpaths resolve to the built package (file:../../packages/core → dist).
    // packages/core has its own dev copy of onnxruntime-web: use this app's.
    dedupe: ["onnxruntime-web"],
    // onnxruntime-web's non-bundled build: its .mjs/.wasm load from /v1/runtime/ort/<version>/ (the SDK's
    // wasmPaths) instead of being inlined into our bundle and copied into dist/assets.
    conditions: ["onnxruntime-web-use-extern-wasm", ...defaultClientConditions],
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  build: {
    rollupOptions: {
      // BUILD_ONLY=help builds just /help.
      input:
        process.env.BUILD_ONLY === "help"
          ? { help: path.join(root, "help/index.html") }
          : { main: path.join(root, "index.html"), help: path.join(root, "help/index.html") },
    },
  },
});
