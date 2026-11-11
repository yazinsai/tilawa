import { createHash, randomBytes } from "node:crypto";
import type { Context } from "hono";

// One salt per process for every /help rate limiter (legacy v1 and v2).
// Set IP_SALT to keep hashes comparable across restarts.
export const IP_SALT = process.env.IP_SALT ?? randomBytes(16).toString("hex");

// Behind Cloudflare -> Dokku nginx, X-Forwarded-For carries Cloudflare's edge IP.
export function ipHash(c: Context) {
  const ip = c.req.header("cf-connecting-ip") ?? c.req.header("x-forwarded-for")?.split(",")[0] ?? "";
  return createHash("sha256").update(IP_SALT + ip.trim()).digest("hex").slice(0, 16);
}
