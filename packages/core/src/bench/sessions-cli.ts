import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "../reference.js";
import { qudSession, runSession, summarizeSessions, validateSession, weakerContexts } from "./sessions.js";
import type { BenchSession, SessionScore } from "./sessions.js";
const { values } = parseArgs({ options: { manifest: { type: "string" }, qud: { type: "string" }, listener: { type: "string", default: "teacher" },
  only: { type: "string" }, out: { type: "string", default: "../../docs/benchmarks/session-baseline-v1.json" } } });
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8"));
const rows: SessionScore[] = [];
const failures: { id: string; reason: string }[] = [];
const items: { session: BenchSession; evidence: any }[] = [];
if (values.qud) for (const f of readdirSync(values.qud).filter(f => f.endsWith(".json") && (!values.only || f === `${values.only}.json`)).sort()) {
  const rec = JSON.parse(readFileSync(resolve(values.qud, f), "utf8"));
  try { items.push({ session: qudSession(rec, `${rec.id}.wav`), evidence: rec }); }
  catch (e) { failures.push({ id: rec.id, reason: String(e) }); }
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
    validateSession(session, index);
    for (const context of weakerContexts(session.context)) rows.push(runSession(session, evidence, index, context, phrases));
    console.log(`scored ${session.id}`);
  } catch (e) { failures.push({ id: session.id, reason: String(e) }); console.log(`excluded ${session.id}: ${e}`); }
}
writeFileSync(resolve(values.out!), JSON.stringify({ version: 1, gating: false, listener: values.listener, reference: index.version,
  note: "Reported only; recorded-session agreement and coverage are not established. QUD is the professional easy end. No release gates replaced. Delays cover matched warnings only; censoredByMiss counts errors without that state.",
  summary: summarizeSessions(rows), rows, failures }, null, 2) + "\n");
for (const g of summarizeSessions(rows)) console.log(`${g.key}: recall ${g.hits}/${g.errors}, confirmed delay p50/p90 ${g.confirmedDelay.p50 ?? "-"}/${g.confirmedDelay.p90 ?? "-"} s (n=${g.confirmedDelay.n}, censored ${g.confirmedDelay.censoredByMiss})`);
console.log(`${rows.length} context rows; ${failures.length} exclusions -> ${values.out}`);
if (!rows.length) process.exitCode = 1;
