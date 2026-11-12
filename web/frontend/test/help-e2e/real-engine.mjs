// Opt-in: /help with the REAL engine (no ?mock) against a running server whose /v1 holds the model.
// Usage: HELP_ORIGIN=http://localhost:5077 HELP_WAV=/path/16k-mono.wav [PW_CHANNEL=chrome] node test/help-e2e/real-engine.mjs
import { chromium } from "playwright";

const origin = process.env.HELP_ORIGIN ?? "http://localhost:5077";
const wav = process.env.HELP_WAV;
if (!wav) throw new Error("HELP_WAV is required");
const browser = await chromium.launch({
  channel: process.env.PW_CHANNEL || undefined,
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-audio-capture=${wav}`, "--autoplay-policy=no-user-gesture-required"],
});
const page = await (await browser.newContext({ permissions: ["microphone"], viewport: { width: 390, height: 844 } })).newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("dialog", (d) => d.accept());
await page.goto(`${origin}/help/?lang=en`);
await page.check("#consent-send");
await page.click("#consent-continue");
await page.click("#start");
await page.waitForSelector("#screen-reading:not([hidden])", { timeout: 120_000 });
const isolated = await page.evaluate(() => crossOriginIsolated);
await page.waitForFunction(() => /Yaseen|يس/.test(document.querySelector("#ayah-ref")?.textContent ?? ""), null, { timeout: 60_000 });
const first = await page.textContent("#ayah-ref");
await page.waitForTimeout(8000);
const later = await page.textContent("#ayah-ref");
const live = await page.textContent("#live");
if (!/Yaseen/.test(live ?? "")) errors.push(`live region announced ${live}`);
await page.click("#stop");
await page.waitForSelector("#screen-review:not([hidden]), #screen-send:not([hidden])", { timeout: 30_000 });
for (let i = 0; i < 8 && (await page.locator("#screen-review:not([hidden])").count()); i++) {
  await page.click('.answer[data-answer="skipped"]').catch(() => {});
  await page.waitForTimeout(200);
}
await page.click("#send");
await page.waitForSelector("#screen-done:not([hidden])", { timeout: 60_000 });
console.log(JSON.stringify({ isolated, first, later, live, deleteCode: !!(await page.textContent("#screen-done")), errors }));
await browser.close();
if (errors.length) process.exit(1);
