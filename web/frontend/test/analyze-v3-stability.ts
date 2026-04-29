/**
 * Analyze streaming stability JSON with exact-match-oriented failure taxonomy.
 *
 * Usage:
 *   npx tsx test/analyze-v3-stability.ts test/stab-gate-on-v3.json
 */

import { readFileSync } from "node:fs";
import { basename, resolve } from "node:path";

type RunClass =
  | "exact"
  | "missing_only"
  | "extra_after_expected"
  | "extra_before_expected"
  | "wrong_initial"
  | "wrong_surah_jump"
  | "no_emit"
  | "partial_multi";

interface SampleRunResult {
  passed: boolean;
  exactPassed?: boolean;
  discoveredVerses: string[];
  recall: number;
  precision: number;
  seqAcc: number;
}

interface SampleStability {
  id: string;
  category: string;
  expectedVerses: string[];
  runs: SampleRunResult[];
  passRate: number;
  classification: "stable-pass" | "stable-fail" | "flaky";
  exactPassRate?: number;
  exactClassification?: "exact-stable-pass" | "exact-stable-fail" | "exact-flaky";
  medianPrecision: number;
  medianRecall: number;
}

interface StabilityReport {
  corpus: string;
  repeats: number;
  samples: SampleStability[];
}

function usage(): never {
  console.error("Usage: npx tsx test/analyze-v3-stability.ts <stability-json>");
  process.exit(1);
}

function isSubsequence<T>(needle: readonly T[], haystack: readonly T[]): boolean {
  let pos = 0;
  for (const item of haystack) {
    if (needle[pos] === item) pos++;
    if (pos === needle.length) return true;
  }
  return needle.length === 0;
}

function isPrefix<T>(prefix: readonly T[], values: readonly T[]): boolean {
  if (prefix.length > values.length) return false;
  for (let i = 0; i < prefix.length; i++) {
    if (prefix[i] !== values[i]) return false;
  }
  return true;
}

function surahOf(ref: string): string {
  return ref.split(":")[0] ?? "";
}

function classifyRun(expected: readonly string[], discovered: readonly string[]): RunClass {
  if (discovered.length === 0) return "no_emit";
  if (
    expected.length === discovered.length &&
    expected.every((ref, idx) => discovered[idx] === ref)
  ) {
    return "exact";
  }

  const expectedSet = new Set(expected);
  const discoveredSet = new Set(discovered);
  const missing = expected.filter((ref) => !discoveredSet.has(ref));
  const extras = discovered.filter((ref) => !expectedSet.has(ref));

  if (expected.length > 1 && missing.length > 0) {
    return "partial_multi";
  }

  if (extras.length === 0 && missing.length > 0) {
    return "missing_only";
  }

  if (extras.length > 0 && discovered[0] && !expectedSet.has(discovered[0])) {
    return "wrong_initial";
  }

  if (extras.length > 0) {
    const expectedSurahs = new Set(expected.map(surahOf));
    const hasWrongSurahAfterExpected =
      isSubsequence(expected, discovered) &&
      discovered.some((ref, idx) => {
        const expectedPrefixSeen = isSubsequence(expected, discovered.slice(0, idx + 1));
        return expectedPrefixSeen && !expectedSurahs.has(surahOf(ref));
      });
    if (hasWrongSurahAfterExpected) return "wrong_surah_jump";
    if (isPrefix(expected, discovered) || isSubsequence(expected, discovered)) {
      return "extra_after_expected";
    }
    return "extra_before_expected";
  }

  return "missing_only";
}

function increment<K extends string>(map: Map<K, number>, key: K, by = 1): void {
  map.set(key, (map.get(key) ?? 0) + by);
}

function formatCounts<K extends string>(counts: Map<K, number>): string {
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([key, count]) => `${key}: ${count}`)
    .join(", ");
}

const input = process.argv[2];
if (!input) usage();

const reportPath = resolve(process.cwd(), input);
const report: StabilityReport = JSON.parse(readFileSync(reportPath, "utf-8"));

const runCounts = new Map<RunClass, number>();
const byCategory = new Map<string, Map<RunClass, number>>();
const offenderScores = new Map<string, { sample: SampleStability; exactFails: number; classes: Map<RunClass, number> }>();

for (const sample of report.samples) {
  const classes = new Map<RunClass, number>();
  let exactFails = 0;
  for (const run of sample.runs) {
    const klass = classifyRun(sample.expectedVerses, run.discoveredVerses);
    increment(runCounts, klass);
    increment(classes, klass);
    if (klass !== "exact") exactFails++;

    const categoryCounts = byCategory.get(sample.category) ?? new Map<RunClass, number>();
    increment(categoryCounts, klass);
    byCategory.set(sample.category, categoryCounts);
  }

  if (exactFails > 0) {
    offenderScores.set(sample.id, { sample, exactFails, classes });
  }
}

const totalRuns = report.samples.reduce((sum, sample) => sum + sample.runs.length, 0);
const exactRuns = runCounts.get("exact") ?? 0;

console.log(`Stability taxonomy: ${basename(reportPath)}`);
console.log(`Corpus: ${report.corpus} | Samples: ${report.samples.length} | Runs: ${totalRuns}`);
console.log(`Exact runs: ${exactRuns}/${totalRuns} (${((exactRuns / totalRuns) * 100).toFixed(1)}%)`);
console.log(`Run classes: ${formatCounts(runCounts)}`);
console.log();

console.log("By category:");
for (const [category, counts] of [...byCategory.entries()].sort()) {
  console.log(`  ${category}: ${formatCounts(counts)}`);
}

console.log();
console.log("Top exact-match offenders:");
const offenders = [...offenderScores.values()]
  .sort((a, b) => {
    if (b.exactFails !== a.exactFails) return b.exactFails - a.exactFails;
    return a.sample.medianPrecision - b.sample.medianPrecision;
  })
  .slice(0, 40);

for (const { sample, exactFails, classes } of offenders) {
  const lastRun = sample.runs[sample.runs.length - 1];
  console.log(
    `  ${sample.id} (${sample.category}) exact-fails=${exactFails}/${sample.runs.length} ` +
      `P=${(sample.medianPrecision * 100).toFixed(1)} R=${(sample.medianRecall * 100).toFixed(1)} ` +
      `classes=[${formatCounts(classes)}] expected=[${sample.expectedVerses.join(", ")}] ` +
      `last=[${lastRun.discoveredVerses.join(", ")}]`,
  );
}
