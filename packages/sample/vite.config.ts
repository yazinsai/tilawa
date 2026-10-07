import { fileURLToPath } from "node:url";
import basicSsl from "@vitejs/plugin-basic-ssl";
import { defineConfig } from "vite";

const isolation = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };
const src = (p: string) => fileURLToPath(new URL(p, import.meta.url));

// HTTPS (self-signed) so a phone on the LAN gets a secure context: getUserMedia and WebCrypto need one.
export default defineConfig({
  base: "./",
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
});
