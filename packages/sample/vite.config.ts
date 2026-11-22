import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import basicSsl from "@vitejs/plugin-basic-ssl";
import { defineConfig } from "vite";

const isolation = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };
const src = (p: string) => fileURLToPath(new URL(p, import.meta.url));
const git = (...args: string[]) => execFileSync("git", args, { encoding: "utf8" }).trim();

// HTTPS (self-signed) so a phone on the LAN gets a secure context: getUserMedia and WebCrypto need one.
export default defineConfig({
  base: "./",
  define: { __SDK_BUILD__: JSON.stringify({ revision: git("rev-parse", "HEAD"), branch: git("branch", "--show-current"), builtAt: new Date().toISOString(), engineDirty: Boolean(git("status", "--porcelain", "--untracked-files=no", "--", "../core/src", "../sdk/src")) }) },
  plugins: [basicSsl()],
  resolve: {
    alias: {
      "@tilawa/core": src("../core/src/index.ts"),
      "@tilawa/sdk/web": src("../sdk/src/web.ts"),
    },
  },
  server: { headers: isolation, host: true, allowedHosts: [".trycloudflare.com"], fs: { allow: ["../.."] } },
  preview: { headers: isolation, host: true, allowedHosts: [".trycloudflare.com"] },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: { rollupOptions: { input: { main: src("index.html"), device: src("device.html") } } },
});
