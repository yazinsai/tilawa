// End-to-end /help in `?mock=1` mode against the real server (/help/api/v2 with a temp store), in headless Chromium
// with fake media. Covers every screen in English and Arabic on iPhone and Pixel viewports, axe checks on each
// screen, a forced network failure with keep-on-device + refresh + resend, an in-memory retry, the discard
// confirmation, and the microphone-denied recovery screen. Asserts the reports landed in storage.
//
//   npm run test:help-e2e                 # builds /help, then runs this
//   PW_CHANNEL=chrome node test/help-e2e/flow.mjs   # use installed Chrome instead of Playwright's Chromium
//
// Screenshots: test/help-screens/<lang>-<device>-<nn>-<screen>.png. Axe results: test/help-screens/axe.json.
import { spawn } from "node:child_process";
import { mkdtemp, mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import AxeBuilder from "@axe-core/playwright";
import { chromium, devices } from "playwright";

const FRONTEND = fileURLToPath(new URL("../..", import.meta.url));
const SHOTS = join(FRONTEND, "test/help-screens");
const CONSENT = "2026-11-report-1";
const failures = [];
const axeLog = [];
const check = (ok, msg) => {
  if (!ok) failures.push(msg);
  console.log(`${ok ? "ok  " : "FAIL"} ${msg}`);
};

const freePort = () =>
  new Promise((resolve) => {
    const s = createServer();
    s.listen(0, "127.0.0.1", () => {
      const { port } = s.address();
      s.close(() => resolve(port));
    });
  });

const port = await freePort();
const origin = `http://127.0.0.1:${port}`;
const store = await mkdtemp(join(tmpdir(), "tilawa-help-e2e-"));
const server = spawn(join(FRONTEND, "node_modules/.bin/tsx"), ["server/index.ts"], {
  cwd: FRONTEND,
  env: {
    ...process.env,
    PORT: String(port),
    HELP_DATA_DIR: join(store, "help"),
    STORAGE_DIR: join(store, "reports"),
    HELP_REPORT_CONSENT_VERSIONS: CONSENT,
    HELP_ALLOWED_ORIGINS: origin,
    HELP_ISOLATION: "1",
    HELP_MIN_FREE_BYTES: "0",
    QURAN_JSON: join(FRONTEND, "dist/quran.json"),
  },
  stdio: ["ignore", "pipe", "pipe"],
});
server.stderr.on("data", (d) => process.stderr.write(`[server] ${d}`));
for (let i = 0; ; i++) {
  try {
    if ((await fetch(`${origin}/api/health`)).ok) break;
  } catch {}
  if (i > 100) throw new Error("server did not start");
  await new Promise((r) => setTimeout(r, 100));
}

const browser = await chromium.launch({
  channel: process.env.PW_CHANNEL || undefined,
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required"],
});
await rm(SHOTS, { recursive: true, force: true });
await mkdir(SHOTS, { recursive: true });

const DEVICES = { iphone: devices["iPhone 13"], pixel: devices["Pixel 7"] };

async function snap(page, tag, n, screen) {
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: join(SHOTS, `${tag}-${String(n).padStart(2, "0")}-${screen}.png`), fullPage: true });
}

async function axe(page, tag, screen) {
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]).analyze();
  axeLog.push({ run: tag, screen, violations: r.violations.map((v) => ({ id: v.id, impact: v.impact, help: v.help, nodes: v.nodes.map((n) => n.target.join(" ")) })) });
  check(r.violations.length === 0, `${tag} axe ${screen}${r.violations.length ? `: ${r.violations.map((v) => `${v.id}(${v.nodes.length})`).join(", ")}` : ""}`);
}

/** Every interactive control is at least 48×48 CSS px (inline links in running text excepted). */
async function targets(page, tag, screen) {
  const small = await page.evaluate(() =>
    [...document.querySelectorAll("button, select, summary, input[type=checkbox], textarea, .mission-links a, .wordmark")]
      .filter((el) => el.offsetParent !== null)
      .map((el) => {
        const box = (el.matches("input[type=checkbox]") ? el.closest("label") ?? el : el).getBoundingClientRect();
        return { el: el.id || el.className || el.tagName, w: Math.round(box.width), h: Math.round(box.height) };
      })
      .filter((b) => b.w < 48 || b.h < 48),
  );
  check(small.length === 0, `${tag} targets ≥48px on ${screen}${small.length ? `: ${JSON.stringify(small)}` : ""}`);
}

async function screen(page, tag, n, name) {
  await snap(page, tag, n, name);
  await axe(page, tag, name);
  await targets(page, tag, name);
}

const visible = (page, sel) => page.locator(sel).waitFor({ state: "visible", timeout: 20_000 });

async function run(lang, deviceName, mode) {
  const tag = `${lang}-${deviceName}`;
  const context = await browser.newContext({ ...DEVICES[deviceName], locale: lang === "ar" ? "ar" : "en-US", permissions: ["microphone"] });
  const page = await context.newPage();
  page.on("pageerror", (e) => check(false, `${tag} page error: ${e.message}`));
  page.on("dialog", (d) => d.accept());
  let n = 0;
  await page.goto(`${origin}/help/?mock=1&speed=4&lang=${lang}`);
  check((await page.getAttribute("html", "dir")) === (lang === "ar" ? "rtl" : "ltr"), `${tag} direction`);

  // 1. consent: the required box gates Continue.
  await visible(page, "#screen-consent");
  await page.click("#consent-continue");
  check(await page.locator("#consent-error").isVisible(), `${tag} consent is required`);
  await screen(page, tag, ++n, "consent");
  await page.check("#consent-send");
  if (mode === "keep") await page.check("#consent-training");
  await page.click("#consent-continue");

  // 2. ready
  await visible(page, "#screen-ready");
  await screen(page, tag, ++n, "ready");
  await page.click("#start");

  // 3. reading
  await visible(page, "#screen-reading");
  await page.waitForFunction(() => document.querySelector("#ayah .w.now"), null, { timeout: 20_000 });
  await page.waitForTimeout(1500);
  await page.click("#stumble");
  check((await page.textContent("#stumble-count")).length > 0, `${tag} bookmark acknowledged`);
  await page.waitForFunction(() => document.querySelector("#ayah .w.flag") || document.querySelector("#ayah-ref")?.textContent, null, { timeout: 20_000 });
  await screen(page, tag, ++n, "reading");
  await page.waitForTimeout(3500); // ≈ 14 s more of take at speed=4
  await page.click("#stumble");
  await page.waitForTimeout(1500);
  check((await page.textContent("#live")).length > 0, `${tag} polite live region announced the ayah`);
  await page.click("#stop");

  // 4. review
  await visible(page, "#screen-review");
  const total = async () => (await page.textContent("#moment-count")).match(/[\d٠-٩]+/g).length;
  check((await total()) === 2, `${tag} review counter`);
  let first = true;
  const answers = ["no", "yes", "unsure", "skipped"];
  let i = 0;
  while (await page.locator("#screen-review").isVisible()) {
    await page.click("#moment-play");
    const dur = await page.evaluate(
      () =>
        new Promise((r) => {
          const a = document.querySelector("#moment-audio");
          if (a.readyState >= 1) r(a.duration);
          else a.addEventListener("loadedmetadata", () => r(a.duration), { once: true });
        }),
    );
    if (first) {
      check(dur > 1 && dur <= 8.01, `${tag} clip is playable (${dur.toFixed(2)} s)`);
      await page.waitForTimeout(300);
      await visible(page, "#moment-mark");
      await page.click("#moment-mark"); // mark another moment from playback
      await page.click("#moment-wrong-ayah");
      await page.click("#moment-details summary");
      await page.fill("#moment-text", "بسم الله");
      await screen(page, tag, ++n, "review");
      first = false;
    }
    await page.click(`.answer[data-answer="${answers[i++ % answers.length]}"]`);
  }

  // 5. send
  await visible(page, "#screen-send");
  await screen(page, tag, ++n, "send");

  // Discard asks first; "Keep it" keeps the take.
  await page.click("#discard");
  await visible(page, "#confirm");
  await snap(page, tag, ++n, "discard-confirm");
  await axe(page, tag, "discard-confirm");
  await page.click("#confirm-no");
  check(await page.locator("#screen-send").isVisible(), `${tag} discard can be cancelled`);

  if (mode === "keep" || mode === "retry") {
    await page.route("**/help/api/v2/reports", (r) => r.abort("internetdisconnected"));
    await page.click("#send");
    await visible(page, "#send-error");
    check((await page.textContent("#send-label")).length > 0, `${tag} send failure shows retry`);
    await screen(page, tag, ++n, "send-failed");
    await page.unroute("**/help/api/v2/reports");
  }
  if (mode === "keep") {
    await page.check("#keep");
    await page.waitForFunction(() => document.querySelector("#keep-status")?.textContent);
    await page.reload();
    await visible(page, "#kept");
    await screen(page, tag, ++n, "kept-after-refresh");
    await page.click("#kept-send");
    await visible(page, "#screen-send");
  }

  await page.click("#send");
  await visible(page, "#screen-done");
  const code = await page.inputValue("#code");
  check(/^[0-9a-f-]{36}\.[A-Za-z0-9_-]{20,}$/.test(code), `${tag} delete code shown`);
  await page.click("#copy");
  await screen(page, tag, ++n, "done");
  const kept = await page.evaluate(
    () =>
      new Promise((r) => {
        const q = indexedDB.open("tilawa-help", 1);
        q.onsuccess = () => {
          const g = q.result.transaction("unsent").objectStore("unsent").count();
          g.onsuccess = () => r(g.result);
        };
        q.onerror = () => r(-1);
      }),
  );
  check(kept === 0, `${tag} nothing left on the device after sending (${kept})`);
  await page.click("#again");
  // After a refresh nothing of the old consent is remembered, so a new take asks again.
  await visible(page, mode === "keep" ? "#screen-consent" : "#screen-ready");
  await context.close();
  return code.split(".")[0];
}

async function micDenied(lang, deviceName) {
  const tag = `${lang}-${deviceName}`;
  const context = await browser.newContext({ ...DEVICES[deviceName], locale: lang });
  const page = await context.newPage();
  await page.goto(`${origin}/help/?mock=1&speed=4&mic=denied&lang=${lang}`);
  await page.check("#consent-send");
  await page.click("#consent-continue");
  await page.click("#start");
  await visible(page, "#ready-error h2");
  await screen(page, tag, 90, "mic-denied");
  await context.close();
}

const sent = [];
try {
  sent.push(await run("en", "iphone", "keep"));
  sent.push(await run("ar", "pixel", "retry"));
  sent.push(await run("ar", "iphone", "plain"));
  sent.push(await run("en", "pixel", "plain"));
  await micDenied("en", "iphone");
  await micDenied("ar", "pixel");

  // The reports are in the store, whole and as answered.
  const dir = join(store, "help/v2/reports");
  const ids = await readdir(dir);
  check(sent.every((id) => ids.includes(id)) && ids.length === sent.length, `storage holds the ${sent.length} sent reports (${ids.length})`);
  for (const id of ids) {
    const report = JSON.parse(await readFile(join(dir, id, "report.json"), "utf8"));
    const wav = await readFile(join(dir, id, "audio.wav"));
    check(report.consent.version === CONSENT && report.consent.send === true && report.source === "tilawa.dev/help", `${id} consent + source`);
    check(report.audio.durationMs > 15_000 && wav.length === 44 + report.audio.durationMs * 32, `${id} whole take (${report.audio.durationMs} ms)`);
    const kinds = report.incidents.map((x) => x.kind);
    check(kinds.includes("stumble") && kinds.includes("warning") && kinds.includes("missed_mistake") && kinds.includes("wrong_position"), `${id} incidents ${kinds.join(",")}`);
    check(report.incidents.every((x) => x.claim.answer === null || ["yes", "no", "unsure", "skipped"].includes(x.claim.answer)), `${id} claims are answers only`);
    check(report.incidents.some((x) => x.claim.text === "بسم الله"), `${id} advanced text kept`);
    check(report.observations.events.some((e) => e.type === "checker"), `${id} carries the engine's observations`);
  }
  const firstReport = JSON.parse(await readFile(join(dir, sent[0], "report.json"), "utf8"));
  check(firstReport.consent.training === true && firstReport.consent.publicDataset === false, "optional consents recorded as ticked");
} finally {
  await writeFile(join(SHOTS, "axe.json"), JSON.stringify(axeLog, null, 2));
  await browser.close();
  server.kill();
  await rm(store, { recursive: true, force: true });
}

console.log(failures.length ? `\n${failures.length} failure(s)` : "\nall passed");
process.exit(failures.length ? 1 : 0);
