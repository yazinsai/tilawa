import { createHmac, timingSafeEqual } from "node:crypto";
import type { Context, MiddlewareHandler } from "hono";
import { getCookie } from "hono/cookie";

// Server-side admin auth for stored recordings, reports and diagnostics.
// Accepts either the /admin session cookie (HMAC-signed, derived from
// ADMIN_PASSWORD) or `Authorization: Bearer $HELP_ADMIN_TOKEN`.
// With neither variable set, every protected route is closed.

export const ADMIN_COOKIE = "admin_session";
export const ADMIN_SESSION_SECONDS = 24 * 3600;

const password = () => process.env.ADMIN_PASSWORD ?? "";
const bearerToken = () => process.env.HELP_ADMIN_TOKEN ?? "";

function safeEqual(a: string, b: string): boolean {
  const ab = Buffer.from(a);
  const bb = Buffer.from(b);
  return ab.length === bb.length && timingSafeEqual(ab, bb);
}

function sign(expiresAt: number): string {
  return createHmac("sha256", `tilawa-admin-session\0${password()}`)
    .update(String(expiresAt))
    .digest("hex");
}

export function passwordMatches(candidate: unknown): boolean {
  return password() !== "" && typeof candidate === "string" && safeEqual(candidate, password());
}

export function newSessionValue(now = Date.now()): string {
  const expiresAt = now + ADMIN_SESSION_SECONDS * 1000;
  return `${expiresAt}.${sign(expiresAt)}`;
}

export function isAdmin(c: Context, now = Date.now()): boolean {
  const token = bearerToken();
  const header = c.req.header("authorization") ?? "";
  if (token && header.startsWith("Bearer ") && safeEqual(header.slice(7), token)) return true;
  if (!password()) return false;
  const value = getCookie(c, ADMIN_COOKIE);
  if (!value) return false;
  const [exp, mac] = value.split(".");
  const expiresAt = Number(exp);
  if (!Number.isFinite(expiresAt) || expiresAt < now || !mac) return false;
  return safeEqual(mac, sign(expiresAt));
}

export const requireAdmin: MiddlewareHandler = async (c, next) => {
  if (!isAdmin(c)) return c.json({ error: "Unauthorized" }, 401);
  await next();
};

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
export const isStoredId = (id: string) => UUID.test(id);

export function escapeHtml(value: unknown): string {
  return String(value ?? "").replace(/[&<>"']/g, (ch) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]!,
  );
}
