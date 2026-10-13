import { existsSync, mkdirSync, readFileSync, writeFileSync, readdirSync } from "node:fs";
import { execSync } from "node:child_process";
import { join, relative, resolve } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "../reference.js";
import { qudSession, runSession, scoreSessionEvents, summarizeSessions, validateSession, weakerContexts } from "./sessions.js";
import type { BenchSession, SessionScore } from "./sessions.js";
import { SESSION_DUMP_VERSION, contextKey, quranText, sessionDumpContext, sessionDumpRecord, sessionEvents } from "./session-dump.js";
import type { SessionDumpRecord } from "./session-dump.js";
const { values } = parseArgs({ options: { manifest: { type: "string" }, qud: { type: "string" }, listener: { type: "string", default: "teacher" },
  only: { type: "string" }, out: { type: "string", default: "../../docs/benchmarks/session-baseline-v1.json" },
  /** Review-only session dump for the scoreboard page (`sessions-<sha>-<listener>.json` + `index.json`); `none` skips it. */
  "clip-dump": { type: "string", default: process.env.CLIP_DUMP_DIR ?? "../../.scratch/scoreboard/clips" },
  /** Served by `tools/scoreboard_serve.py` as `/audio/sessions/`; session audio must live under it. */
  "audio-root": { type: "string", default: "../../.scratch/session-bench" },
  sha: { type: "string" } } });
const asset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const text = quranText(asset);
const audioRoot = resolve(values["audio-root"]!);
const dumped: SessionDumpRecord[] = [];
const phrases = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8"));
const rows: SessionScore[] = [];
const failures: { id: string; reason: string }[] = [];
const items: { session: BenchSession; evidence: any; audio: string }[] = [];
if (values.qud) for (const f of readdirSync(values.qud).filter(f => f.endsWith(".json") && (!values.only || f === `${values.only}.json`)).sort()) {
  const rec = JSON.parse(readFileSync(resolve(values.qud, f), "utf8"));
  try { items.push({ session: qudSession(rec, `${rec.id}.wav`), evidence: rec, audio: resolve(values.qud, `${rec.id}.wav`) }); }
  catch (e) { failures.push({ id: rec.id, reason: String(e) }); }
}
if (values.manifest) {
  const m = JSON.parse(readFileSync(values.manifest, "utf8"));
  if (m.version !== 1 || !Array.isArray(m.sessions)) throw new Error("Invalid manifest");
  const base = resolve(values.manifest, "..");
  for (const x of m.sessions) items.push({ session: x.session, evidence: JSON.parse(readFileSync(resolve(base, x.evidence), "utf8")), audio: resolve(base, x.session.audio) });
}
if (!items.length) throw new Error("Pass --manifest or --qud with session data");
const dumping = values["clip-dump"] !== "none";
for (const { session, evidence, audio } of items) {
  try {
    validateSession(session, index);
    const rel = relative(audioRoot, audio);
    const record = sessionDumpRecord(session, rel.startsWith("..") ? audio : `sessions/${rel}`, text);
    for (const context of weakerContexts(session.context)) {
      const row = runSession(session, evidence, index, context, phrases);
      rows.push(row);
      if (!dumping) continue;
      const events = sessionEvents(session, evidence, index, context, phrases);
      const detail = sessionDumpContext(session, events, scoreSessionEvents(session, events, index, context), index);
      if (["hits", "falseWarnings", "nonErrorWarnings", "wrongPositionEvents"].some(k => (detail.score as any)[k] !== (row as any)[k]))
        throw new Error("review dump replay disagrees with runSession; update session-dump.ts sessionEvents");
      record.contexts[contextKey(context)] = detail;
    }
    if (dumping) { if (rel.startsWith("..")) console.log(`warning: ${audio} is outside --audio-root; the review page cannot play it`); dumped.push(record); }
    console.log(`scored ${session.id}`);
  } catch (e) { failures.push({ id: session.id, reason: String(e) }); console.log(`excluded ${session.id}: ${e}`); }
}
writeFileSync(resolve(values.out!), JSON.stringify({ version: 1, gating: false, listener: values.listener, reference: index.version,
  note: "Reported only; recorded-session agreement and coverage are not established. QUD is the professional easy end. No release gates replaced. Delays cover matched warnings only; censoredByMiss counts errors without that state.",
  summary: summarizeSessions(rows), rows, failures }, null, 2) + "\n");
for (const g of summarizeSessions(rows)) console.log(`${g.key}: recall ${g.hits}/${g.errors}, confirmed delay p50/p90 ${g.confirmedDelay.p50 ?? "-"}/${g.confirmedDelay.p90 ?? "-"} s (n=${g.confirmedDelay.n}, censored ${g.confirmedDelay.censoredByMiss})`);
console.log(`${rows.length} context rows; ${failures.length} exclusions -> ${values.out}`);
if (!rows.length) process.exitCode = 1;
if (dumping && dumped.length) {
  const dir = resolve(values["clip-dump"]!); mkdirSync(dir, { recursive: true });
  const sha = values.sha ?? execSync("git rev-parse --short HEAD", { encoding: "utf8" }).trim(), recordedAt = new Date().toISOString();
  const file = `sessions-${sha}-${values.listener}.json`;
  writeFileSync(join(dir, file), JSON.stringify({ version: SESSION_DUMP_VERSION, kind: "sessions", sha, listener: values.listener, recordedAt, reference: index.version, sessions: dumped }));
  const indexPath = join(dir, "index.json");
  const runs: { file: string }[] = existsSync(indexPath) ? JSON.parse(readFileSync(indexPath, "utf8")) : [];
  writeFileSync(indexPath, JSON.stringify([...runs.filter(x => x.file !== file), { sha, listener: values.listener, recordedAt, file, kind: "sessions" }], null, 1) + "\n");
  console.log(`session dump: ${join(dir, file)}`);
}
