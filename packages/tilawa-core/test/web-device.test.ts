import { describe, expect, it } from "vitest";
import { autoWebTier, defaultWebProviders, isAppleMobile } from "../src/web-device.js";

const IPHONE_CHROME = "Mozilla/5.0 (iPhone; CPU iPhone OS 26_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/142.0 Mobile/15E148 Safari/604.1";
const IPAD = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.1 Safari/605.1.15";
const ANDROID = "Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0 Mobile Safari/537.36";
const gpu = {};

describe("web device defaults", () => {
  it("iOS gets the lite tier on WASM even with WebGPU (the tab is killed building the full one)", () => {
    for (const nav of [{ userAgent: IPHONE_CHROME, gpu }, { userAgent: IPAD, maxTouchPoints: 5, gpu }]) {
      expect(isAppleMobile(nav)).toBe(true);
      expect(autoWebTier(nav)).toBe("lite");
      expect(defaultWebProviders(nav)).toEqual(["wasm"]);
    }
  });

  it("a desktop Mac stays on full, WebGPU first", () => {
    const nav = { userAgent: IPAD, maxTouchPoints: 0, gpu };
    expect(autoWebTier(nav)).toBe("full");
    expect(defaultWebProviders(nav)).toEqual(["webgpu", "wasm"]);
  });

  it("Android keeps the old rule: full with WebGPU, lite without or on low memory", () => {
    expect(autoWebTier({ userAgent: ANDROID, gpu })).toBe("full");
    expect(autoWebTier({ userAgent: ANDROID })).toBe("lite");
    expect(autoWebTier({ userAgent: ANDROID, gpu, deviceMemory: 2 })).toBe("lite");
    expect(defaultWebProviders({ userAgent: ANDROID, gpu })).toEqual(["webgpu", "wasm"]);
  });
});
