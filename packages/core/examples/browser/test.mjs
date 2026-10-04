// Browser regression for the published `dist`: serves the repo root, loads
// examples/browser/index.html in headless Chromium, feeds test_corpus clips
// through the file input, and checks `final_sequence` against the manifest.
//
//   npm run test:browser            # default sample set
//   npm run test:browser -- --all   # every mp3/wav sample
//   PW_CHANNEL=chrome npm run test:browser   # use installed Chrome
//
// Needs the model + corpus in web/frontend/public
// (web/frontend/scripts/fetch-zipformer-assets.sh).
import { createServer } from "node:http";
import { existsSync, readFileSync, createReadStream, statSync } from "node:fs";
import { extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const ROOT = resolve(fileURLToPath(import.meta.url), "../../../../..");
const CORPUS = join(ROOT, "lab/benchmark/test_corpus");
const PUBLIC = join(ROOT, "web/frontend/public");
const DEFAULT_IDS = [
  "ref_001001",
  "ref_112001",
  "retasy_000",
  "ref_023115",
  "multi_113_001_005",
  "multi_002_285_286",
];

for (const f of ["models/zipformer_a0w_ep1_a05.int8.onnx", "zipformer_quran.json", "quran.json"]) {
  if (!existsSync(join(PUBLIC, f))) {
    console.error(`missing ${join(PUBLIC, f)} — run web/frontend/scripts/fetch-zipformer-assets.sh`);
    process.exit(1);
  }
}
if (!existsSync(join(ROOT, "packages/core/dist/index.js"))) {
  console.error("missing packages/core/dist — run npm run build");
  process.exit(1);
}

const manifest = JSON.parse(readFileSync(join(CORPUS, "manifest.json"), "utf8"));
const all = (Array.isArray(manifest) ? manifest : manifest.samples).filter((s) => !s.file.endsWith(".m4a"));
const samples = process.argv.includes("--all") ? all : all.filter((s) => DEFAULT_IDS.includes(s.id));

const MIME = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".json": "application/json",
  ".wasm": "application/wasm",
};
const server = createServer((req, res) => {
  let path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname));
  if (path.endsWith("/")) path += "index.html";
  const file = join(ROOT, path);
  if (!file.startsWith(ROOT) || !existsSync(file) || !statSync(file).isFile()) {
    res.writeHead(404).end();
    return;
  }
  res.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream" });
  createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const url = `http://127.0.0.1:${server.address().port}/packages/core/examples/browser/`;

const browser = await chromium.launch({ channel: process.env.PW_CHANNEL || undefined });
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("[pageerror]", e.message));
page.on("console", (m) => m.type() === "error" && console.error("[console]", m.text()));

let failures = 0;
let passed = 0;
try {
  await page.goto(url);
  await page.waitForFunction(
    () => /^(Ready|Error)/.test(document.getElementById("status").textContent),
    null,
    { timeout: 300_000 },
  );
  const status = await page.textContent("#status");
  console.log(status);
  if (!status.startsWith("Ready")) throw new Error(status);

  for (const s of samples) {
    await page.evaluate(() => delete document.body.dataset.done);
    await page.setInputFiles("#file", join(CORPUS, s.file));
    await page.waitForFunction(() => document.body.dataset.done === "1", null, { timeout: 300_000 });
    const got = (await page.textContent("#final")).trim();
    const want = s.expected_verses.map((v) => `${v.surah}:${v.ayah}`).join(", ");
    const ok = got === want;
    if (ok) passed++;
    else failures++;
    console.log(`${ok ? "PASS" : "FAIL"} ${s.id.padEnd(20)} want [${want}] got [${got}]`);
  }
} catch (e) {
  console.error(e);
  failures++;
} finally {
  await browser.close();
  server.close();
}

console.log(`${passed}/${samples.length} passed`);
process.exit(failures ? 1 : 0);
