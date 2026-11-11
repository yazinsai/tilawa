import { realpathSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const root = path.dirname(fileURLToPath(import.meta.url));
let nodeModulesReal: string | undefined;
try {
  nodeModulesReal = realpathSync(path.join(root, "node_modules"));
} catch {
  nodeModulesReal = undefined;
}

export default defineConfig({
  define: {
    // Cache-buster for /audio-processor.js: the worklet is fetched by URL from
    // public/, so CDN edges (Cloudflare, max-age=14400) can pin a stale copy
    // across deploys. A per-build id in the query string sidesteps that.
    __BUILD_ID__: JSON.stringify(Date.now().toString(36)),
  },
  resolve: {
    // Exact match only, so subpaths (@tilawa/core/web, /report) resolve to the built package
    // (node_modules/@tilawa/core -> packages/core, dist) that /help bundles.
    alias: [
      {
        find: /^@tilawa\/core$/,
        replacement: fileURLToPath(
          new URL("../../packages/core/src/index.ts", import.meta.url),
        ),
      },
    ],
    // packages/core's dist imports onnxruntime-web; use this app's copy, not the core devDependency.
    dedupe: ["onnxruntime-web"],
  },
  build: {
    rollupOptions: {
      // BUILD_ONLY=help builds just /help (e.g. while the demo page is being rewritten).
      input:
        process.env.BUILD_ONLY === "help"
          ? { help: path.join(root, "help/index.html") }
          : { main: path.join(root, "index.html"), help: path.join(root, "help/index.html") },
    },
  },
  worker: {
    format: "es",
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
    // Worktree `node_modules` is a symlink into the main checkout; Vite's
    // default fs.allow is the worktree root and 403s the ORT wasm fetch.
    fs: {
      allow: [root, ...(nodeModulesReal ? [nodeModulesReal] : [])],
    },
  },
});
