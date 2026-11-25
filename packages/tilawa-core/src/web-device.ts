import type { TierName } from "./runtime/models.js";

type Nav = Partial<Navigator> & { gpu?: unknown; deviceMemory?: number; userAgentData?: { mobile?: boolean } };

/**
 * iPhone/iPad, where every browser is WebKit and a tab is killed past a tight memory cap. iPadOS reports a
 * desktop Mac user agent, so a touch "Mac" counts too.
 */
export function isAppleMobile(nav: Nav): boolean {
  const ua = nav.userAgent ?? "";
  return /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && (nav.maxTouchPoints ?? 0) > 1);
}

/** `tier: "auto"` on the web: lite on low-memory devices, phones without WebGPU and all iOS devices. */
export function autoWebTier(nav: Nav): TierName {
  if (typeof nav.deviceMemory === "number" && nav.deviceMemory < 4) return "lite";
  if (isAppleMobile(nav)) return "lite";
  const mobile = nav.userAgentData?.mobile ?? /Android|Mobile/i.test(nav.userAgent ?? "");
  return mobile && !("gpu" in nav) ? "lite" : "full";
}
