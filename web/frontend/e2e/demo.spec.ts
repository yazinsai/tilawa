import { mkdirSync } from "node:fs";
import { expect, test, type Page } from "@playwright/test";

const SCREENS = "test/demo-screens";
mkdirSync(SCREENS, { recursive: true });

test.skip(!process.env.E2E_ENABLED, "set TILAWA_MODELS_DIR or TILAWA_MODELS_URL to run the end-to-end demo test");

/** Every ayah the position display shows, in order. */
async function watchAyahs(page: Page) {
  await page.evaluate(() => {
    const seen: string[] = [];
    (window as unknown as { __seen: string[] }).__seen = seen;
    const el = document.getElementById("ayah")!;
    new MutationObserver(() => {
      const ref = el.dataset.ref;
      if (ref && seen.at(-1) !== ref) seen.push(ref);
    }).observe(el, { attributes: true, attributeFilter: ["data-ref"] });
  });
  return () => page.evaluate(() => (window as unknown as { __seen: string[] }).__seen);
}

async function recite(page: Page, label: string) {
  const ortRequests: { url: string; status: number }[] = [];
  page.on("response", (r) => {
    if (r.url().includes("/v1/runtime/ort/")) ortRequests.push({ url: new URL(r.url()).pathname, status: r.status() });
  });
  await page.goto("/");
  await expect(page.locator("#btn-start")).toBeVisible();
  await page.screenshot({ path: `${SCREENS}/${label}-idle.png`, fullPage: true });

  expect(await page.evaluate(() => crossOriginIsolated)).toBe(true);
  const seen = await watchAyahs(page);
  const started = Date.now();
  await page.click("#btn-start");
  await expect(page.locator("body")).toHaveAttribute("data-view", "listening", { timeout: 120_000 });
  const readyMs = Date.now() - started;

  // The clip ends on 36:5.
  await expect(page.locator("#ayah")).toHaveAttribute("data-ref", "36:5", { timeout: 90_000 });
  await page.screenshot({ path: `${SCREENS}/${label}-listening.png`, fullPage: true });
  const backend = await page.locator("#tech-list").getAttribute("data-backend");

  await page.click("#btn-stop");
  await expect(page.locator("body")).toHaveAttribute("data-view", "stopped");
  const summary = await page.locator("#summary").textContent();
  await page.locator("#tech").evaluate((d) => ((d as HTMLDetailsElement).open = true));
  await page.screenshot({ path: `${SCREENS}/${label}-stopped.png`, fullPage: true });

  const result = { label, readyMs, ayahs: await seen(), summary, backend, ortRequests };
  console.log(JSON.stringify(result));
  expect(ortRequests.length).toBeGreaterThan(0);
  for (const r of ortRequests) expect(r.status, r.url).toBe(200);
  expect(summary).toContain("36:");
  return result;
}

test("desktop: free recitation of Yā-Sīn 36:1–5 follows to 36:5", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await recite(page, "desktop");
});

test("mobile: same clip on a phone-sized viewport", async ({ browser, baseURL }) => {
  const context = await browser.newContext({
    baseURL,
    viewport: { width: 390, height: 844 },
    deviceScaleFactor: 2,
    isMobile: true,
    hasTouch: true,
    permissions: ["microphone"],
  });
  const page = await context.newPage();
  await recite(page, "mobile");
  await context.close();
});
