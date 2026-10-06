import { defineConfig } from "vite";

const isolation = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };

export default defineConfig({
  base: "./",
  server: { headers: isolation, host: true },
  preview: { headers: isolation, host: true },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
