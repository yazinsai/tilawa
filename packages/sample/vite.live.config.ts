import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
const src = (p: string) => fileURLToPath(new URL(p, import.meta.url));
const git = (...args: string[]) => execFileSync("git", args, { encoding: "utf8" }).trim();
export default defineConfig({
  root: src("live"), base: "/live/", publicDir: false,
  define: { __LIVE_BUILD__: JSON.stringify({ revision: git("rev-parse", "HEAD"), branch: git("branch", "--show-current"), builtAt: new Date().toISOString(), engineDirty: Boolean(git("status", "--porcelain", "--untracked-files=no", "--", "../core/src", "../sdk/src")) }) },
  resolve: { alias: { "@tilawa/core": src("../core/src/index.ts"), "@tilawa/sdk/web": src("../sdk/src/web.ts") } },
  build: { outDir: src("../../modal/live-dist"), emptyOutDir: true },
});
