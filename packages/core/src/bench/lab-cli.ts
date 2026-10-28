/**
 * Lab scorer CLI (ticket 08). Scores this shard's sessions (manifest index % LAB_SHARDS == LAB_SHARD) to LAB_OUT/score.json:
 *
 *   pnpm --filter @tilawa/core lab                                 # on Modal via tools/lab.py run --shards N
 *   pnpm --filter @tilawa/core lab --merge /lab/runs/<run>         # shards -> /lab/runs/<run>/score.json
 *
 * Manifest (`--manifest`, default $LAB_DATA/lab-v1/manifest-v1.json), paths relative to its directory:
 *   { "version": 1, "testSet": "lab-v1", "sessions": [{ "id", "split": "practice" | "exam", "session": "<BenchSession path>" | {...},
 *     "tags"?: ["strict", ...], "audio"?: { "phone"?: "<wav>", "studio"?: "<wav>" },
 *     "evidence"?: { "<listener>": { "phone"?: "<path>", "studio"?: "<path>" } } }] }
 * Evidence not listed is looked up at the manifest's "evidencePattern" ({id}, {listener}, {variant}), else evidence/<listener>/<id>.<variant>.json.
 * "slices": { "strict" } (the session builder's tag) counts as "strict".
 * Cases on data that failed the automated checks (`--checks`, default checks-v1.json beside the manifest) carry `dataCheck`.
 */
import { execSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { parseArgs } from "node:util";
import { disambiguationWords } from "../disambiguation.js";
import { loadReferenceIndex } from "../reference.js";
import { DEFAULT_CANDIDATE } from "./scoreboard.js";
import { VARIANTS, labReport, labSession, markDataChecks, mergeReports } from "./lab.js";
import type { DataChecks, LabReport, SessionResult, Split, Variant } from "./lab.js";
import type { BenchSession, SessionEvidence } from "./sessions.js";

const { values } = parseArgs({ args: process.argv.slice(2).filter(a => a !== "--"), options: {
  manifest: { type: "string", default: process.env.LAB_MANIFEST ?? `${process.env.LAB_DATA ?? "/lab/data"}/lab-v1/manifest-v1.json` },
  listener: { type: "string", default: DEFAULT_CANDIDATE },
  /** Teacher reference listener; `none` skips it. Scored only where its evidence exists. */
  teacher: { type: "string", default: "teacher" },
  variants: { type: "string", default: VARIANTS.join(",") },
  shard: { type: "string", default: process.env.LAB_SHARD ?? "0" },
  shards: { type: "string", default: process.env.LAB_SHARDS ?? "1" },
  out: { type: "string" },
  merge: { type: "string" },
  only: { type: "string" },
  sha: { type: "string" },
  "exam-cases": { type: "boolean", default: false },
  "no-tracker-oracle": { type: "boolean", default: false },
  /** Automated test set checks (ticket 04); default `checks-v1.json` beside the manifest when present, `none` skips. */
  checks: { type: "string" },
} });

const summary = (r: LabReport) => {
  const o = r.overall, a = r.attribution, c = r.components;
  console.log(`${r.testSet} ${r.listener} @ ${r.sha}: ${r.sessions.scored} sessions (${r.sessions.practice} practice, ${r.sessions.exam} exam), ${r.sessions.excluded.length} excluded`);
  console.log(`overall (${r.headline.split}, ${r.headline.variant}): ${o.per5Min ?? "-"} wrong calls / 5 min = ${o.missed} missed + ${o.falseWarnings} false warnings (${o.falseWarningsOnNonErrors} on repeats/restarts), ${o.caught}/${o.mistakes} caught (${o.misclassified} misclassified)`);
  console.log(`attribution: listener ${a.listener}, tracker ${a.tracker}, checker ${a.checker} (total ${a.total})`);
  console.log(`listener PER ${c.listener.per ?? "-"}; tracker on place ${c.tracker.onPlaceShare ?? "-"}, ${c.tracker.falseJumps} false jumps, lock delay p50/p90 ${c.tracker.lockDelayS.p50 ?? "-"}/${c.tracker.lockDelayS.p90 ?? "-"} s; checker ${c.checker.per5Min ?? "-"} / 5 min (${c.checker.missed} + ${c.checker.falseWarnings})`);
  for (const s of ["practice", "exam"] as const) console.log(`${s}: ${r.splits[s].overall.per5Min ?? "-"} / 5 min (${r.splits[s].overall.missed} + ${r.splits[s].overall.falseWarnings}) over ${r.splits[s].recitedMinutes} min`);
  const t = r.splits[r.headline.split].teacher;
  if (t) console.log(`teacher reference: ${t.per5Min ?? "-"} / 5 min (${t.missed} + ${t.falseWarnings}), PER ${t.per ?? "-"}`);
};
const write = (path: string, report: LabReport) => {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(report, null, 1) + "\n");
  summary(report);
  console.log(`-> ${path}`);
};

if (values.merge) {
  const dir = resolve(values.merge);
  const parts = readdirSync(dir).filter(d => /^\d+$/.test(d) && existsSync(join(dir, d, "score.json")))
    .map(d => JSON.parse(readFileSync(join(dir, d, "score.json"), "utf8")) as LabReport);
  write(resolve(values.out ?? join(dir, "score.json")), mergeReports(parts, { examCases: values["exam-cases"] }));
  process.exit(0);
}

const sha = values.sha ?? /-([0-9a-f]{7,40}(?:-dirty)?)-\d{8}T\d{6}Z$/.exec(process.env.LAB_RUN ?? "")?.[1] ?? (() => {
  try { return execSync("git rev-parse --short HEAD", { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim(); } catch { return "unknown"; }
})();
const shard = Number(values.shard), shards = Number(values.shards);
if (!(Number.isInteger(shard) && Number.isInteger(shards) && shards > 0 && shard >= 0 && shard < shards)) throw new Error(`bad shard ${values.shard}/${values.shards}`);
const variants = values.variants!.split(",") as Variant[];
if (variants.some(v => !VARIANTS.includes(v))) throw new Error(`unknown variant in ${values.variants}`);
const teacherName = values.teacher === "none" ? null : values.teacher!;

const asset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const phrases = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8"));
const disamb = disambiguationWords(index);

interface Entry { id: string; split: Split; session: string | BenchSession; tags?: string[]; strict?: boolean; slices?: { strict?: boolean };
  audio?: Partial<Record<Variant, string>>; evidence?: string | Record<string, Partial<Record<Variant, string>>> }
const manifestPath = resolve(values.manifest!);
const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as { version: number; testSet: string; evidencePattern?: string; sessions: Entry[] };
if (manifest.version !== 1 || !manifest.testSet || !Array.isArray(manifest.sessions)) throw new Error(`Invalid lab manifest ${manifestPath}`);
const base = dirname(manifestPath);
const evidencePath = (e: Entry, listener: string, v: Variant) => {
  const listed = typeof e.evidence === "object" ? e.evidence[listener]?.[v] : undefined;
  const fallback = manifest.evidencePattern?.replace("{id}", e.id).replace("{listener}", listener).replace("{variant}", v) ?? join("evidence", listener, `${e.id}.${v}.json`);
  const path = resolve(base, listed ?? fallback);
  if (listed && !existsSync(path)) throw new Error(`${e.id}: listed ${listener} ${v} evidence missing at ${path}`);
  return existsSync(path) ? path : null;
};
const dataRoot = resolve(process.env.LAB_DATA ?? "/lab/data");
/** Audio as the review page addresses it: relative to /lab/data, else absolute. */
const audioOf = (e: Entry) => Object.fromEntries(Object.entries(e.audio ?? {}).map(([v, p]) => {
  const abs = resolve(base, p!), rel = relative(dataRoot, abs);
  return [v, rel.startsWith("..") ? abs : rel];
}));
const checksPath = values.checks === "none" ? null : resolve(base, values.checks ?? "checks-v1.json");
if (values.checks && checksPath && !existsSync(checksPath)) throw new Error(`checks missing at ${checksPath}`);
const checks = checksPath && existsSync(checksPath) ? JSON.parse(readFileSync(checksPath, "utf8")) as DataChecks : null;
if (checks) console.log(`data checks ${checksPath}: ${checks.failed.length} failed edits, ${checks.failedClean.length} failed clean stretches marked on cases`);
const load = (path: string | null) => path ? JSON.parse(readFileSync(path, "utf8")) as SessionEvidence : undefined;

const results: SessionResult[] = [], excluded: { id: string; reason: string }[] = [];
manifest.sessions.forEach((e, i) => {
  if (i % shards !== shard || (values.only && e.id !== values.only)) return;
  try {
    const t0 = performance.now();
    if (e.split !== "practice" && e.split !== "exam") throw new Error(`invalid split ${e.split}`);
    const session = (typeof e.session === "string" ? JSON.parse(readFileSync(resolve(base, e.session), "utf8")) : e.session) as BenchSession;
    if (session.id !== e.id) throw new Error(`manifest id ${e.id} != session id ${session.id}`);
    const pick = (listener: string) => Object.fromEntries(variants.flatMap(v => { const x = load(evidencePath(e, listener, v)); return x ? [[v, x]] : []; }));
    const evidence = pick(values.listener!);
    if (!Object.keys(evidence).length) throw new Error(`no ${values.listener} evidence`);
    const strict = e.strict ?? e.slices?.strict ?? (e.tags ? e.tags.includes("strict") : undefined);
    const r = labSession({ session, split: e.split, strict, evidence, audio: audioOf(e), teacher: teacherName ? pick(teacherName) : undefined,
      trackerOracle: !values["no-tracker-oracle"] }, { index, disamb, phrases });
    results.push(checks ? markDataChecks(r, checks) : r);
    const v = r.variants.phone ?? r.variants.studio!;
    console.log(`scored ${e.id} (${e.split}${r.strict ? ", strict" : ""}) in ${((performance.now() - t0) / 1000).toFixed(1)} s: ${v.missed} missed, ${v.falseWarnings} false warnings`);
  } catch (err) {
    excluded.push({ id: e.id, reason: err instanceof Error ? err.message : String(err) });
    console.log(`excluded ${e.id}: ${err}`);
  }
});
const out = resolve(values.out ?? (process.env.LAB_OUT ? join(process.env.LAB_OUT, "score.json") : "lab-score.json"));
write(out, labReport(results, { testSet: manifest.testSet, listener: values.listener!, teacher: teacherName, sha, run: process.env.LAB_RUN ?? null,
  reference: index.version, manifest: manifestPath }, { excluded, shard: { index: shard, count: shards }, examCases: values["exam-cases"] }));
if (!results.length && !excluded.length) console.log("no sessions in this shard");
if (excluded.length) console.log(`warning: ${excluded.length} excluded (listed in the report)`);
