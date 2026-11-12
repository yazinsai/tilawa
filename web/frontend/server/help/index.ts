// tilawa.dev/help — volunteers record correct recitations. Only v4 (correct
// recitation) prompts are issued or accepted: deliberately altering the Quran
// is not permissible, so mistake prompts from older versions are retired.
import { randomBytes, randomInt } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { appendFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { Hono, type Context } from "hono";
import { bodyLimit } from "hono/body-limit";
import { loadQuran } from "./prompts.mjs";
import { generatePrompt } from "./prompts-v4.mjs";
import { signPrompt, verifyPrompt } from "./prompt-token.mjs";
import { ipHash } from "./ip.js";

const DATA_DIR = resolve(process.env.HELP_DATA_DIR ?? "./storage/help");
const QURAN_JSON = resolve(process.env.QURAN_JSON ?? "./dist/quran.json");
const ADMIN_TOKEN = process.env.HELP_ADMIN_TOKEN ?? "";
// Configure a stable key to let pending prompts survive server restarts.
const PROMPT_SECRET = process.env.HELP_PROMPT_SECRET ?? randomBytes(32).toString("hex");
const MAX_BYTES = 8 * 1024 * 1024;
const MIN_BYTES = 4 * 1024;
const MAX_PER_HOUR = 60;
const CONSENT_VERSION = "2026-09-30";
const SUBMISSIONS = join(DATA_DIR, "submissions.jsonl");
const SUBSCRIBERS = join(DATA_DIR, "subscribers.jsonl");

const AUDIO_EXT: Record<string, string> = {
  "audio/webm": "webm",
  "audio/ogg": "ogg",
  "audio/mp4": "m4a",
  "audio/mpeg": "mp3",
  "audio/wav": "wav",
  "audio/x-wav": "wav",
};
const ID = /^[A-Za-z0-9_-]{8,40}$/;
const EMAIL = /^[^\s@]{1,64}@[^\s@]{1,190}\.[^\s@]{2,}$/;
const ENUMS: Record<string, Set<string>> = {
  device: new Set(["phone", "laptop", "desktop", "tablet", "headset", "other"]),
  gender: new Set(["male", "female", ""]),
  level: new Set(["beginner", "intermediate", "hafiz", ""]),
};

let quran: ReturnType<typeof loadQuran> | null = null;
const getQuran = () => (quran ??= loadQuran(JSON.parse(readFileSync(QURAN_JSON, "utf8"))));

const readJsonl = (path: string) =>
  existsSync(path)
    ? readFileSync(path, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l))
    : [];
const used = new Set<string>(readJsonl(SUBMISSIONS).map((r) => `${r.speaker}:${r.seed}`));
const subscribed = new Set<string>(readJsonl(SUBSCRIBERS).map((r) => r.email));
const ready = mkdir(join(DATA_DIR, "audio"), { recursive: true });

const hits = new Map<string, number[]>();
function rateLimited(key: string) {
  const now = Date.now();
  const recent = (hits.get(key) ?? []).filter((t) => now - t < 3_600_000);
  recent.push(now);
  hits.set(key, recent);
  return recent.length > MAX_PER_HOUR;
}

const isAdmin = (c: Context) => !!ADMIN_TOKEN && c.req.header("authorization") === `Bearer ${ADMIN_TOKEN}`;

export const helpApi = new Hono();

helpApi.get("/config", (c) => c.json({ consent: CONSENT_VERSION }));

helpApi.get("/prompt", (c) => {
  const p = generatePrompt(getQuran(), randomInt(0, 2 ** 32 - 1));
  return c.json({ v: p.v, token: signPrompt(p, PROMPT_SECRET), seed: p.seed, kind: p.label.kind, display: p.display, instruction: p.instruction });
});

helpApi.post(
  "/submit",
  bodyLimit({ maxSize: MAX_BYTES, onError: (c) => c.json({ error: "too large" }, 413) }),
  async (c) => {
    const ip = ipHash(c);
    if (rateLimited(ip)) return c.json({ error: "Too many submissions, try again later." }, 429);

    const q = c.req.query();
    const seed = Number(q.seed);
    const duration = Number(q.duration);
    const mime = (c.req.header("content-type") ?? "").split(";")[0].trim();
    const problems = [
      !Number.isInteger(seed) || seed < 0 || seed > 2 ** 32 ? "seed" : "",
      !ID.test(q.speaker ?? "") ? "speaker" : "",
      q.consent !== CONSENT_VERSION ? "consent" : "",
      q.followed !== "1" ? "followed" : "",
      !AUDIO_EXT[mime] ? "content-type" : "",
      !(duration >= 1 && duration <= 90) ? "duration" : "",
      ...Object.entries(ENUMS).map(([k, set]) => (set.has(q[k] ?? "") ? "" : k)),
    ].filter(Boolean);
    if (problems.length) return c.json({ error: `Invalid: ${problems.join(", ")}` }, 400);
    if (used.has(`${q.speaker}:${seed}`)) return c.json({ error: "Already submitted." }, 409);

    // Tabs opened before v4 may still hold mistake prompts; never store those.
    const prompt = q.v === "4" ? verifyPrompt(q.token, seed, PROMPT_SECRET) : null;
    if (!prompt || prompt.v !== 4 || prompt.label.kind !== "none") return c.json({ error: "Invalid prompt. Load another ayah and try again." }, 400);

    const audio = Buffer.from(await c.req.arrayBuffer());
    if (audio.length < MIN_BYTES) return c.json({ error: "Recording is empty or too short." }, 400);

    await ready;
    const id = `${Date.now().toString(36)}_${randomBytes(4).toString("hex")}`;
    const file = `audio/${id}.${AUDIO_EXT[mime]}`;
    await writeFile(join(DATA_DIR, file), audio);
    used.add(`${q.speaker}:${seed}`);

    const record = {
      id,
      file,
      ts: new Date().toISOString(),
      seed,
      prompt_version: prompt.v,
      scenario: prompt.label.scenario ?? null,
      mechanism: prompt.label.mechanism ?? null,
      passage_family: prompt.label.passage_family ?? null,
      elicitation: prompt.label.elicitation,
      ayahs: prompt.label.ayahs,
      passage: prompt.label.passage,
      fluency_control: prompt.label.fluency_control ?? false,
      speaker: q.speaker,
      device: q.device,
      gender: q.gender || null,
      level: q.level || null,
      ref: prompt.label.ref,
      kind: prompt.label.kind,
      mistakes: prompt.label.mistakes,
      extra_mistake: q.extra === "1",
      duration_s: duration,
      mime,
      bytes: audio.length,
      user_agent: (c.req.header("user-agent") ?? "").slice(0, 200),
      ip_hash: ip,
      consent: CONSENT_VERSION,
    };
    await appendFile(SUBMISSIONS, JSON.stringify(record) + "\n");
    return c.json({ ok: true, id });
  },
);

// Deliberately no speaker id: emails must not be linkable to recordings.
helpApi.post("/subscribe", bodyLimit({ maxSize: 4096 }), async (c) => {
  if (rateLimited(ipHash(c))) return c.json({ error: "rate" }, 429);
  const body = await c.req.json().catch(() => null);
  const email = String(body?.email ?? "").trim().toLowerCase();
  if (!EMAIL.test(email)) return c.json({ error: "email" }, 400);
  if (!subscribed.has(email)) {
    await ready;
    subscribed.add(email);
    const lang = body.lang === "ar" ? "ar" : "en";
    await appendFile(SUBSCRIBERS, JSON.stringify({ email, lang, ts: new Date().toISOString() }) + "\n");
  }
  return c.json({ ok: true });
});

const ndjson = async (c: Context, path: string) => {
  if (!isAdmin(c)) return c.json({ error: "unauthorized" }, 401);
  return c.body(existsSync(path) ? await readFile(path, "utf8") : "", 200, { "content-type": "application/x-ndjson" });
};
helpApi.get("/export", (c) => ndjson(c, SUBMISSIONS));
helpApi.get("/subscribers", (c) => ndjson(c, SUBSCRIBERS));

helpApi.get("/audio/:name", async (c) => {
  if (!isAdmin(c)) return c.json({ error: "unauthorized" }, 401);
  const name = c.req.param("name");
  const path = join(DATA_DIR, "audio", name);
  if (!/^[a-z0-9_]+\.(webm|ogg|m4a|mp3|wav)$/.test(name) || !existsSync(path)) return c.json({ error: "not found" }, 404);
  return c.body(await readFile(path), 200, { "content-type": "application/octet-stream" });
});
