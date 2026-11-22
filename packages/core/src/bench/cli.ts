import { execSync } from "node:child_process";
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "../reference.js";
import type { PhraseAsset, ReferenceAsset } from "../reference.js";
import { BENCH_REPORT_VERSION, benchRecording, summarise } from "./index.js";
import type { QudRecording, RecordingResult } from "./index.js";

const { values } = parseArgs({
  options: {
    dir: { type: "string", default: "../../.scratch/bench/qud" },
    out: { type: "string", default: "../../docs/benchmarks" },
    posteriors: { type: "boolean", default: false },
    only: { type: "string" },
    "no-phrases": { type: "boolean", default: false },
  },
});

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const phrases: PhraseAsset | undefined = values["no-phrases"]
  ? undefined
  : JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8"));
const config = values.posteriors ? "posteriors" : "decoded";
const files = readdirSync(values.dir!)
  .filter((f) => f.endsWith(".json"))
  .filter((f) => !values.only || f.startsWith(values.only))
  .sort();

const results: RecordingResult[] = [];
for (const f of files) {
  const rec: QudRecording = JSON.parse(readFileSync(join(values.dir!, f), "utf8"));
  const r = benchRecording(index, rec, { posteriors: values.posteriors!, phrases });
  results.push(r);
  const ok = r.starts.filter((s) => s.latencyS !== null && s.latencyS <= 5).length;
  console.log(`${r.id}: ${ok}/${r.starts.length} starts in 5 s, false jumps ${r.falseJumps}/${r.jumps}, rtf ${r.rtf.toFixed(4)}`);
}

const summary = summarise(results);
const commit = execSync("git rev-parse --short HEAD").toString().trim();
const report = {
  version: BENCH_REPORT_VERSION,
  dataset: "QUD-Technologies/quran-alignment-benchmark v1",
  referenceVersion: index.version,
  inventory: index.inventoryId,
  commit,
  configuration: { reranking: values.posteriors, phrases: phrases?.version ?? null, chunkSeconds: 1 },
  summary,
  recordings: results,
};
mkdirSync(values.out!, { recursive: true });
const base = join(values.out!, `qud-v1-${config}${phrases ? "-phrases" : ""}`);
writeFileSync(`${base}.json`, JSON.stringify(report, null, 2) + "\n");

const pct = (n: number, d: number) => (d ? `${((100 * n) / d).toFixed(1)}%` : "–");
const s = summary;
const md = [
  `# QUD alignment benchmark v1: ${config}`,
  "",
  `Report v${BENCH_REPORT_VERSION}, reference ${index.version}, commit ${commit}. Posterior reranking: ${values.posteriors ? "on" : "off"}. Phrases: ${phrases?.version ?? "off"}. Streamed in 1 s chunks.`,
  "",
  `${s.recordings} recordings, ${s.audioMinutes.toFixed(0)} min, ${s.starts} passage starts.`,
  "",
  "| metric | value |",
  "|---|---|",
  `| correct ayah within 5 s | ${s.within5s}/${s.starts} ${pct(s.within5s, s.starts)} |`,
  `| starts with a false lock first | ${s.falseLockStarts}/${s.starts} ${pct(s.falseLockStarts, s.starts)} |`,
  `| unresolved starts | ${s.unresolved}/${s.starts} ${pct(s.unresolved, s.starts)} |`,
  `| locked time on a wrong ayah | ${(100 * s.falseLockTimeShare).toFixed(2)}% |`,
  `| false jumps | ${s.falseJumps}/${s.jumps} ${pct(s.falseJumps, s.jumps)} |`,
  `| shortlist recall | ${(100 * s.shortlistRecall).toFixed(1)}% |`,
  `| median time to first lock | ${s.medianTimeToFirstLockS?.toFixed(2) ?? "–"} s |`,
  `| shared-opening starts | ${s.sharedOpenings} |`,
  `| median lock delay after the distinguishing word | ${s.medianDistinguishLatencyS?.toFixed(2) ?? "–"} s |`,
  `| real-time factor (mean / max) | ${s.meanRtf.toFixed(4)} / ${s.maxRtf.toFixed(4)} |`,
  "",
  "| recording | style | min | starts in 5 s | false-lock starts | unresolved | wrong locked time | false jumps | rtf |",
  "|---|---|---|---|---|---|---|---|---|",
  ...results.map((r) => {
    const n = r.starts.length;
    const ok = r.starts.filter((x) => x.latencyS !== null && x.latencyS <= 5).length;
    return `| ${r.id} | ${r.style}${r.noisy ? ", noisy" : ""} | ${(r.audioS / 60).toFixed(0)} | ${ok}/${n} | ${r.starts.filter((x) => x.falseLock).length} | ${r.starts.filter((x) => x.latencyS === null).length} | ${pct(r.wrongLockedSamples, r.lockedSamples)} | ${r.falseJumps}/${r.jumps} | ${r.rtf.toFixed(4)} |`;
  }),
  "",
].join("\n");
writeFileSync(`${base}.md`, md);
console.log(`\n${md}`);
