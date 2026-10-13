import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "../reference.js";
import { qudAudit, runSession, validateSession, weakerContexts } from "./sessions.js";
import type { BenchSession, QudAudit, SessionScore } from "./sessions.js";
const { values } = parseArgs({ options: { manifest: { type: "string" }, qud: { type: "string" }, listener: { type: "string", default: "teacher" },
  only: { type: "string" }, audit: { type: "boolean", default: false }, out: { type: "string", default: "../../docs/benchmarks/session-baseline-v1.json" } } });
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8"));
const rows: SessionScore[] = [];
const failures: { id: string; reason: string }[] = [];
const audits: QudAudit[] = [];
const items: { session: BenchSession; evidence: any }[] = [];
if (values.qud) for (const f of readdirSync(values.qud).filter(f => f.endsWith(".json") && (!values.only || f === `${values.only}.json`)).sort()) {
  const rec = JSON.parse(readFileSync(resolve(values.qud, f), "utf8"));
  const { session, audit } = qudAudit(rec, index);
  audits.push(audit);
  if (session) items.push({ session, evidence: rec }); else failures.push({ id: rec.id, reason: audit.reason! });
}
if (values.audit) {
  console.table(audits);
  writeFileSync(resolve(values.out!), JSON.stringify({ version: 1, reference: index.version, audit: audits }, null, 2) + "\n");
  console.log(`${audits.filter(a => a.status === "converted").length}/${audits.length} converted -> ${values.out}`);
  process.exit(audits.length ? 0 : 1);
}
if (values.manifest) {
  const m = JSON.parse(readFileSync(values.manifest, "utf8"));
  if (m.version !== 1 || !Array.isArray(m.sessions)) throw new Error("Invalid manifest");
  const base = resolve(values.manifest, "..");
  for (const x of m.sessions) items.push({ session: x.session, evidence: JSON.parse(readFileSync(resolve(base, x.evidence), "utf8")) });
}
if (!items.length) throw new Error("Pass --manifest or --qud with session data");
for (const { session, evidence } of items) {
  try {
    const t0 = performance.now();
    validateSession(session, index);
    for (const context of weakerContexts(session.context)) rows.push(runSession(session, evidence, index, context, phrases));
    console.log(`scored ${session.id} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  } catch (e) { failures.push({ id: session.id, reason: String(e) }); console.log(`excluded ${session.id}: ${e}`); }
}
writeFileSync(resolve(values.out!), JSON.stringify({ version: 1, gating: false, listener: values.listener, reference: index.version,
  note: "Reported only; recorded-session agreement and coverage are not established. QUD is the professional easy end. No release gates replaced.", rows, failures, ...(audits.length ? { audit: audits } : {}) }, null, 2) + "\n");
console.log(`${rows.length} context rows; ${failures.length} exclusions -> ${values.out}`);
if (!rows.length) process.exitCode = 1;
