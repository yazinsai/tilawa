// End-to-end check of the built demo against real models (opt-in: needs the V1 model files).
//
//   TILAWA_MODELS_DIR=<dir with spliced-clean-int8/, spliced-clean-lite-int8/> npx playwright test
//   TILAWA_MODELS_URL=<mirror or template, see scripts/prepare-v1-assets.mjs>   npx playwright test
//
// It builds the demo and server, prepares dist/v1 from those models, starts the built server and recites a
// real recording into Chromium's fake microphone. Without a models source every test is skipped.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { defineConfig, devices } from "@playwright/test";

const PORT = Number(process.env.E2E_PORT ?? 4317);
const source = process.env.TILAWA_MODELS_DIR
  ? `--models-dir "${resolve(process.env.TILAWA_MODELS_DIR)}"`
  : process.env.TILAWA_MODELS_URL
    ? `--models-url "${process.env.TILAWA_MODELS_URL}"`
    : null;

/** Yā-Sīn 36:1–5, one reciter, 28.6 s (lab/benchmark/test_corpus), as 16 kHz mono PCM for the fake microphone. */
export const CLIP = resolve(process.env.E2E_CLIP ?? "../../lab/benchmark/test_corpus/multi_036_001_005.wav");
const work = mkdtempSync(join(tmpdir(), "tilawa-e2e-"));
const wav = join(work, "clip-16k.wav");
if (source && existsSync(CLIP))
  execFileSync("ffmpeg", ["-v", "error", "-y", "-i", CLIP, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav]);
mkdirSync(join(work, "storage"), { recursive: true });
process.env.E2E_ENABLED = source ? "1" : "";

export default defineConfig({
  testDir: "e2e",
  timeout: 180_000,
  expect: { timeout: 30_000 },
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: `http://localhost:${PORT}`,
    permissions: ["microphone"],
    launchOptions: {
      args: [
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        `--use-file-for-fake-audio-capture=${wav}%noloop`,
        "--autoplay-policy=no-user-gesture-required",
      ],
    },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: source
    ? {
        command: [
          "npm run build",
          "npm run build:server",
          `node scripts/prepare-v1-assets.mjs ${source}`,
          "node dist-server/index.mjs",
        ].join(" && "),
        url: `http://localhost:${PORT}/api/health`,
        timeout: 600_000,
        reuseExistingServer: false,
        stdout: "pipe",
        env: {
          PORT: String(PORT),
          STORAGE_DIR: join(work, "storage/reports"),
          HELP_DATA_DIR: join(work, "storage/help"),
        },
      }
    : undefined,
});
