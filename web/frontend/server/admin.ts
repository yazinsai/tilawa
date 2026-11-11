import { Hono } from "hono";
import { setCookie } from "hono/cookie";
import { ADMIN_COOKIE, ADMIN_SESSION_SECONDS, escapeHtml, isAdmin, newSessionValue, passwordMatches } from "./auth.js";
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

const STORAGE_DIR = process.env.STORAGE_DIR || "./storage/reports";
const DIAGNOSTICS_DIR = process.env.STORAGE_DIR
  ? join(process.env.STORAGE_DIR, "../diagnostics")
  : "./storage/diagnostics";

export const adminApp = new Hono();

const isAuthed = isAdmin;

// Login page
adminApp.get("/login", (c) => {
  const error = c.req.query("error") ? "<p style='color:#c0564b'>Wrong password</p>" : "";
  return c.html(`<!DOCTYPE html>
<html><head><title>Admin Login</title>
<style>body{font-family:system-ui;background:#faf8f3;display:flex;justify-content:center;align-items:center;height:100vh}
form{background:#fff;padding:2rem;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);text-align:center}
input{display:block;margin:0.5rem auto;padding:0.5rem;border:1px solid #ddd;border-radius:6px;font-size:1rem}
button{margin-top:0.5rem;padding:0.5rem 1.5rem;background:#b8986a;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:1rem}</style>
</head><body>
<form method="POST" action="/admin/login">
  <h2>Admin</h2>${error}
  <input type="password" name="password" placeholder="Password" autofocus>
  <button type="submit">Login</button>
</form></body></html>`);
});

adminApp.post("/login", async (c) => {
  const form = await c.req.formData();
  const pw = form.get("password") as string;
  if (passwordMatches(pw)) {
    setCookie(c, ADMIN_COOKIE, newSessionValue(), {
      path: "/",
      httpOnly: true,
      secure: new URL(c.req.url).protocol === "https:" || c.req.header("x-forwarded-proto") === "https",
      sameSite: "Strict",
      maxAge: ADMIN_SESSION_SECONDS,
    });
    return c.redirect("/admin");
  }
  return c.redirect("/admin/login?error=1");
});

// Admin dashboard
adminApp.get("/", async (c) => {
  if (!isAuthed(c)) return c.redirect("/admin/login");

  let reports: any[] = [];
  try {
    const entries = await readdir(STORAGE_DIR);
    for (const entry of entries) {
      try {
        const raw = await readFile(join(STORAGE_DIR, entry, "meta.json"), "utf-8");
        reports.push(JSON.parse(raw));
      } catch { /* skip */ }
    }
    reports.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  } catch { /* empty */ }

  let diagnostics: any[] = [];
  try {
    const entries = await readdir(DIAGNOSTICS_DIR);
    for (const entry of entries) {
      try {
        const raw = await readFile(join(DIAGNOSTICS_DIR, entry, "meta.json"), "utf-8");
        diagnostics.push(JSON.parse(raw));
      } catch { /* skip */ }
    }
    diagnostics.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  } catch { /* empty */ }

  const rows = reports.map(r => `
    <tr>
      <td>${escapeHtml(new Date(r.timestamp).toLocaleString())}</td>
      <td>Surah ${escapeHtml(r.surah)}, Ayah ${escapeHtml(r.ayah)}</td>
      <td>${escapeHtml(r.modelPrediction || "—")}</td>
      <td>${escapeHtml(r.debugBundle?.mode || "—")}</td>
      <td>${escapeHtml(r.notes ? String(r.notes).slice(0, 80) : "—")}</td>
      <td><audio controls src="/api/reports/${encodeURIComponent(r.id)}/audio" preload="none"></audio></td>
    </tr>`).join("");

  const triggerLabel = (t: string) =>
    t === "surah_jump" ? "Surah Jump" : t === "rapid_switching" ? "Rapid Switching" : t;

  const diagRows = diagnostics.map(d => `
    <tr>
      <td>${escapeHtml(new Date(d.timestamp).toLocaleString())}</td>
      <td><span class="trigger-badge trigger-${escapeHtml(d.trigger)}">${escapeHtml(triggerLabel(d.trigger))}</span></td>
      <td>${Array.isArray(d.events) ? d.events.length : 0}</td>
      <td>${d.hasAudio ? `<audio controls src="/api/diagnostics/${encodeURIComponent(d.id)}/audio" preload="none"></audio>` : "—"}</td>
    </tr>`).join("");

  return c.html(`<!DOCTYPE html>
<html><head><title>Admin Dashboard</title>
<style>
body{font-family:system-ui;background:#faf8f3;padding:2rem;max-width:1100px;margin:0 auto}
h1,h2{color:#2c2416;margin-bottom:1rem}
h1{font-size:1.4rem}
h2{font-size:1.2rem;margin-top:2.5rem}
.count{color:#8a7e6b;font-size:0.9rem;margin-bottom:1.5rem}
table{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 6px rgba(0,0,0,0.06);margin-bottom:2rem}
th{background:#f5f0e8;color:#2c2416;padding:0.75rem;text-align:left;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.04em}
td{padding:0.75rem;border-top:1px solid #f0ebe3;font-size:0.9rem;color:#2c2416;vertical-align:middle}
audio{height:32px;width:200px}
tr:hover td{background:#faf8f3}
.empty{text-align:center;padding:3rem;color:#8a7e6b}
.trigger-badge{display:inline-block;padding:0.2rem 0.6rem;border-radius:4px;font-size:0.8rem;font-weight:500}
.trigger-surah_jump{background:#fde8e8;color:#9b2c2c}
.trigger-rapid_switching{background:#fef3cd;color:#856404}
</style></head><body>
<h1>Error Reports</h1>
<p class="count">${reports.length} report${reports.length !== 1 ? "s" : ""}</p>
${reports.length ? `<table>
<thead><tr><th>Time</th><th>Expected Verse</th><th>Model Predicted</th><th>Mode</th><th>Notes</th><th>Audio</th></tr></thead>
<tbody>${rows}</tbody>
</table>` : "<p class='empty'>No reports yet.</p>"}

<h2>Auto-Diagnostics</h2>
<p class="count">${diagnostics.length} diagnostic${diagnostics.length !== 1 ? "s" : ""}</p>
${diagnostics.length ? `<table>
<thead><tr><th>Time</th><th>Trigger</th><th>Events</th><th>Audio</th></tr></thead>
<tbody>${diagRows}</tbody>
</table>` : "<p class='empty'>No diagnostics captured yet.</p>"}
</body></html>`);
});
