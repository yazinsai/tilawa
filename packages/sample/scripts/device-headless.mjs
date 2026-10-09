// Runs the device page's clip measurement in a desktop browser, headless, and prints the results JSON.
//   pnpm build && pnpm preview --port 4318 &   then
//   npx -y -p playwright node scripts/device-headless.mjs [--tier full|lite] [--browser chrome|webkit]
import { parseArgs } from "node:util";
import { chromium, webkit } from "playwright";

const { values } = parseArgs({ options: {
  url: { type: "string", default: "https://localhost:4318/device.html" },
  tier: { type: "string", default: "full" },
  browser: { type: "string", default: "chrome" },
} });
const browser = values.browser === "webkit" ? await webkit.launch() : await chromium.launch({ channel: "chrome", headless: true });
const page = await (await browser.newContext({ ignoreHTTPSErrors: true })).newPage();
page.on("pageerror", (e) => console.error("page error:", e.message));
await page.goto(`${values.url}?tiers=${values.tier}&battery=0`);
await page.click("#start");
await page.waitForFunction(() => /^Done|broke/.test(document.getElementById("status").textContent), null, { timeout: 1_800_000 });
console.log(await page.textContent("#json"));
await browser.close();
