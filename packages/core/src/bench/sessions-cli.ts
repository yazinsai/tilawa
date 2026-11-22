import { existsSync, mkdirSync, readFileSync, writeFileSync, readdirSync } from "node:fs";
import { execSync } from "node:child_process";
import { join, relative, resolve } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "../reference.js";
import { qudAudit, qudContexts, runSession, summarizeSessions, validateSession, weakerContexts } from "./sessions.js";
import type { BenchSession, QudAudit, SessionEvidence, SessionScore } from "./sessions.js";
import { SESSION_DUMP_VERSION, contextKey, quranText, sessionDumpContext, sessionDumpRecord } from "./session-dump.js";
import type { SessionDumpRecord } from "./session-dump.js";
const { values } = parseArgs({ options: { manifest: { type: "string" }, qud: { type: "string" }, listener: { type: "string", default: "teacher" },
  only: { type: "string" }, audit: { type: "boolean", default: false }, out: { type: "string", default: "../../docs/benchmarks/session-baseline-v1.json" },
  /** Review-only session dump for the scoreboard page (`sessions-<sha>-<listener>.json` + `index.json`); `none` skips it. */
  "clip-dump": { type: "string", default: process.env.CLIP_DUMP_DIR ?? "../../.scratch/scoreboard/clips" },
  /** Served by `tools/scoreboard_serve.py` as `/audio/sessions/`; session audio must live under it. */
  "audio-root": { type: "string", default: "../../.scratch/session-bench" },
  /** With --qud: score `<dir>/<id>.evidence.json` (a fresh student decode of `<dir>/../<id>.wav`) instead of the QUD teacher evidence; recordings without one are listed as missing. */
  "student-evidence": { type: "string" },
  /** `qud` (default) scores QUD recordings at assignment, page, surah and none; `own` uses only the session's context chain. */
  "qud-contexts": { type: "string", default: "qud" },
  /** Comma-separated QUD `meta.style`s left out of scoring; `none` scores every style. */
  "unsupported-styles": { type: "string", default: "hadr" },
  sha: { type: "string" } } });
const UNSUPPORTED_REASON = "fast recitation unsupported pending fast-speech sample collection and listener fine-tuning";
const unsupportedStyles = new Set(values["unsupported-styles"] === "none" ? [] : values["unsupported-styles"]!.split(","));
const unsupported: { id: string; style: string; reason: string }[] = [];
const omittedContexts: { id: string; context: "assignment"; reason: string }[] = [];
const contexts = (s: BenchSession) => {
  if (s.source !== "qud" || s.context.kind !== "none" || values["qud-contexts"] !== "qud") return weakerContexts(s.context);
  const { contexts, omitted } = qudContexts(s);
  if (omitted) { omittedContexts.push({ id: s.id, context: "assignment", reason: omitted }); console.log(`no assignment for ${s.id}: ${omitted}`); }
  return contexts;
};
const missing: string[] = [];
const asset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const text = quranText(asset);
const audioRoot = resolve(values["audio-root"]!);
const dumped: SessionDumpRecord[] = [];
const phrases = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8"));
const rows: SessionScore[] = [];
const failures: { id: string; reason: string }[] = [];
const audits: QudAudit[] = [];
const items: { session: BenchSession; evidence: SessionEvidence; audio: string }[] = [];
if (values.qud) for (const f of readdirSync(values.qud).filter(f => f.endsWith(".json") && (!values.only || f === `${values.only}.json`)).sort()) {
  const rec = JSON.parse(readFileSync(resolve(values.qud, f), "utf8"));
  const { session, audit } = qudAudit(rec, index);
  audits.push(audit);
  if (!session) { failures.push({ id: rec.id, reason: audit.reason! }); continue; }
  if (unsupportedStyles.has(rec.meta?.style)) { unsupported.push({ id: rec.id, style: rec.meta.style, reason: UNSUPPORTED_REASON }); continue; }
  const student = values["student-evidence"] && resolve(values["student-evidence"], `${rec.id}.evidence.json`);
  if (!student) items.push({ session, evidence: rec, audio: resolve(values.qud, `${rec.id}.wav`) });
  else if (existsSync(student)) items.push({ session, evidence: JSON.parse(readFileSync(student, "utf8")), audio: resolve(values["student-evidence"]!, "..", `${rec.id}.wav`) });
  else missing.push(rec.id);
}
if (missing.length) console.log(`no student decode for ${missing.length}: ${missing.join(", ")}`);
if (values.audit) {
  console.table(audits);
  writeFileSync(resolve(values.out!), JSON.stringify({ version: 1, reference: index.version, audit: audits }, null, 2) + "\n");
  console.log(`${audits.filter(a => a.status === "converted").length}/${audits.length} converted -> ${values.out}`);
  // Every QUD recording is expected to convert; any exclusion (or an empty audit) fails.
  process.exit(audits.length && audits.every(a => a.status === "converted") ? 0 : 1);
}
if (values.manifest) {
  const m = JSON.parse(readFileSync(values.manifest, "utf8"));
  if (m.version !== 1 || !Array.isArray(m.sessions)) throw new Error("Invalid manifest");
  const base = resolve(values.manifest, "..");
  for (const x of m.sessions) items.push({ session: x.session, evidence: JSON.parse(readFileSync(resolve(base, x.evidence.replace("{listener}", values.listener!)), "utf8")), audio: resolve(base, x.session.audio) });
}
if (!items.length) throw new Error("Pass --manifest or --qud with session data");
const dumping = values["clip-dump"] !== "none";
for (const { session, evidence, audio } of items) {
  try {
    const t0 = performance.now();
    validateSession(session, index);
    const rel = relative(audioRoot, audio);
    const record = sessionDumpRecord(session, rel.startsWith("..") ? audio : `sessions/${rel}`, text);
    for (const context of contexts(session)) {
      const { score, events } = runSession(session, evidence, index, context, phrases);
      rows.push(score);
      if (dumping) record.contexts[contextKey(context)] = sessionDumpContext(session, events, score, index);
    }
    if (dumping) { if (rel.startsWith("..")) console.log(`warning: ${audio} is outside --audio-root; the review page cannot play it`); dumped.push(record); }
    console.log(`scored ${session.id} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  } catch (e) { failures.push({ id: session.id, reason: String(e) }); console.log(`excluded ${session.id}: ${e}`); }
}
writeFileSync(resolve(values.out!), JSON.stringify({ version: 1, gating: false, listener: values.listener, reference: index.version,
  note: "Reported only; recorded-session agreement and coverage are not established. QUD is the professional easy end. No release gates replaced. Delays cover matched warnings only; censoredByMiss counts errors without that state.",
  summary: summarizeSessions(rows), rows, failures, ...(unsupported.length ? { unsupported } : {}), ...(omittedContexts.length ? { omittedContexts } : {}), ...(missing.length ? { missingEvidence: missing } : {}), ...(audits.length ? { audit: audits } : {}) }, null, 2) + "\n");
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
