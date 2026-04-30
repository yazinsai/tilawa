/**
 * Oracle upper bound for exact SeqAcc recovery from conservative cascade pruning.
 *
 * Usage:
 *   npx tsx test/oracle-prune-stability.ts
 *   npx tsx test/oracle-prune-stability.ts --input=test/stab-gate-on-v3.json --json=test/oracle-prune-upper-bound-v3.json
 */

import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

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
  discoveredVerses: string[];
  seqAcc: number;
}

interface SampleStability {
  id: string;
  category: string;
  expectedVerses: string[];
  runs: SampleRunResult[];
}

interface StabilityReport {
  corpus: string;
  repeats: number;
  samples: SampleStability[];
}

interface RecoveryExample {
  sampleId: string;
  runIndex: number;
  originalClass: RunClass;
  expected: string[];
  discovered: string[];
  pruned: string[];
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

function refsEqual(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((ref, idx) => b[idx] === ref);
}

function surahOf(ref: string): string {
  return ref.split(":")[0] ?? "";
}

function ayahOf(ref: string): number {
  return Number(ref.split(":")[1] ?? Number.NaN);
}

function isSameSurahContinuation(extra: string, previous: string): boolean {
  return surahOf(extra) === surahOf(previous) && ayahOf(extra) > ayahOf(previous);
}

function classifyRun(expected: readonly string[], discovered: readonly string[]): RunClass {
  if (discovered.length === 0) return "no_emit";
  if (refsEqual(expected, discovered)) return "exact";

  const expectedSet = new Set(expected);
  const discoveredSet = new Set(discovered);
  const missing = expected.filter((ref) => !discoveredSet.has(ref));
  const extras = discovered.filter((ref) => !expectedSet.has(ref));

  if (expected.length > 1 && missing.length > 0) return "partial_multi";
  if (extras.length === 0 && missing.length > 0) return "missing_only";
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

function conservativePrune(expected: readonly string[], discovered: readonly string[]): string[] {
  if (refsEqual(expected, discovered)) return discovered.slice();
  if (!expected.length || !discovered.length) return discovered.slice();

  if (isPrefix(expected, discovered)) {
    let previous = expected[expected.length - 1];
    let allContinuation = true;
    for (const extra of discovered.slice(expected.length)) {
      if (!isSameSurahContinuation(extra, previous)) {
        allContinuation = false;
        break;
      }
      previous = extra;
    }
    if (allContinuation) return expected.slice();
  }

  if (isSubsequence(expected, discovered)) {
    const expectedSet = new Set(expected);
    const filtered = discovered.filter((ref) => expectedSet.has(ref));
    if (refsEqual(filtered, expected)) {
      const firstExpected = discovered.findIndex((ref) => expectedSet.has(ref));
      const hasWrongPrefix = firstExpected > 0;
      if (!hasWrongPrefix) return filtered;
    }
  }

  return discovered.slice();
}

function aggressivePrune(expected: readonly string[], discovered: readonly string[]): string[] {
  const conservative = conservativePrune(expected, discovered);
  if (refsEqual(expected, conservative)) return conservative;
  if (isSubsequence(expected, discovered)) return expected.slice();
  return conservative;
}

function increment(map: Record<string, number>, key: string): void {
  map[key] = (map[key] ?? 0) + 1;
}

function addExample(
  examples: Record<string, RecoveryExample[]>,
  klass: RunClass,
  example: RecoveryExample,
): void {
  const bucket = examples[klass] ?? [];
  if (bucket.length < 8) bucket.push(example);
  examples[klass] = bucket;
}

const args = process.argv.slice(2);
const inputArg = args.find((arg) => arg.startsWith("--input="));
const jsonArg = args.find((arg) => arg.startsWith("--json="));
const inputPath = inputArg?.split("=")[1] ?? "test/stab-gate-on-v3.json";
const jsonPath = jsonArg?.split("=")[1] ?? "test/oracle-prune-upper-bound-v3.json";

const report: StabilityReport = JSON.parse(readFileSync(resolve(process.cwd(), inputPath), "utf-8"));

let totalRuns = 0;
let baselineExact = 0;
let conservativeExact = 0;
let aggressiveExact = 0;
const baselineByClass: Record<string, number> = {};
const conservativeRecoveredByClass: Record<string, number> = {};
const aggressiveRecoveredByClass: Record<string, number> = {};
const conservativeExamples: Record<string, RecoveryExample[]> = {};
const aggressiveExamples: Record<string, RecoveryExample[]> = {};

for (const sample of report.samples) {
  for (let runIndex = 0; runIndex < sample.runs.length; runIndex++) {
    const run = sample.runs[runIndex];
    const expected = sample.expectedVerses;
    const discovered = run.discoveredVerses;
    const klass = classifyRun(expected, discovered);
    totalRuns++;
    increment(baselineByClass, klass);
    if (refsEqual(expected, discovered)) baselineExact++;

    const conservative = conservativePrune(expected, discovered);
    if (refsEqual(expected, conservative)) {
      conservativeExact++;
      if (klass !== "exact") {
        increment(conservativeRecoveredByClass, klass);
        addExample(conservativeExamples, klass, {
          sampleId: sample.id,
          runIndex,
          originalClass: klass,
          expected,
          discovered,
          pruned: conservative,
        });
      }
    }

    const aggressive = aggressivePrune(expected, discovered);
    if (refsEqual(expected, aggressive)) {
      aggressiveExact++;
      if (klass !== "exact") {
        increment(aggressiveRecoveredByClass, klass);
        addExample(aggressiveExamples, klass, {
          sampleId: sample.id,
          runIndex,
          originalClass: klass,
          expected,
          discovered,
          pruned: aggressive,
        });
      }
    }
  }
}

const output = {
  input: inputPath,
  corpus: report.corpus,
  repeats: report.repeats,
  totalRuns,
  baseline: {
    exact: baselineExact,
    seqAcc: baselineExact / totalRuns,
    byClass: baselineByClass,
  },
  conservativePrune: {
    exact: conservativeExact,
    seqAcc: conservativeExact / totalRuns,
    recovered: conservativeExact - baselineExact,
    recoveredByClass: conservativeRecoveredByClass,
    examples: conservativeExamples,
  },
  aggressivePrune: {
    exact: aggressiveExact,
    seqAcc: aggressiveExact / totalRuns,
    recovered: aggressiveExact - baselineExact,
    recoveredByClass: aggressiveRecoveredByClass,
    examples: aggressiveExamples,
  },
};

writeFileSync(resolve(process.cwd(), jsonPath), JSON.stringify(output, null, 2));
console.log(`Oracle prune upper bound: ${report.corpus}, ${totalRuns} runs`);
console.log(`Baseline exact:      ${baselineExact}/${totalRuns} (${((baselineExact / totalRuns) * 100).toFixed(1)}%)`);
console.log(`Conservative exact:  ${conservativeExact}/${totalRuns} (${((conservativeExact / totalRuns) * 100).toFixed(1)}%)`);
console.log(`Aggressive exact:    ${aggressiveExact}/${totalRuns} (${((aggressiveExact / totalRuns) * 100).toFixed(1)}%)`);
console.log(`JSON saved to ${resolve(process.cwd(), jsonPath)}`);
