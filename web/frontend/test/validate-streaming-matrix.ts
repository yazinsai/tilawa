/**
 * Run the frozen validator multiple times and summarize medians/minima.
 *
 * Usage:
 *   tsx test/validate-streaming-matrix.ts
 *   tsx test/validate-streaming-matrix.ts --repeats=3 --fail-on-gate
 */

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");

interface Sample {
  id: string;
  category: string;
}

interface RunSummary {
  correct: number;
  total: number;
  passes: string[];
  failures: string[];
  categories: Record<string, { correct: number; total: number }>;
}

const args = process.argv.slice(2);
const repeatsArg = args.find((arg) => arg.startsWith("--repeats="));
const repeats = repeatsArg ? parseInt(repeatsArg.split("=")[1], 10) : 3;
const failOnGate = args.includes("--fail-on-gate");
const tsxPath = resolve(ROOT, "node_modules/.bin/tsx");

function median(values: number[]): number {
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

function parseRun(stdout: string, samples: Sample[]): RunSummary {
  const categories = Object.fromEntries(
    [...new Set(samples.map((sample) => sample.category))].map((category) => [
      category,
      { correct: 0, total: 0 },
    ]),
  ) as Record<string, { correct: number; total: number }>;
  const sampleById = new Map(samples.map((sample) => [sample.id, sample]));

  const passes: string[] = [];
  const failures: string[] = [];

  for (const line of stdout.split("\n")) {
    const match = line.match(/^\s+(PASS|FAIL)\s+(\S+)/);
    if (!match) continue;

    const [, status, id] = match;
    const sample = sampleById.get(id);
    if (!sample) continue;

    categories[sample.category].total += 1;
    if (status === "PASS") {
      passes.push(id);
      categories[sample.category].correct += 1;
    } else {
      failures.push(id);
    }
  }

  const resultMatch = stdout.match(/Result:\s+(\d+)\/(\d+)/);
  if (!resultMatch) {
    throw new Error(`Unable to parse validator output:\n${stdout}`);
  }

  return {
    correct: parseInt(resultMatch[1], 10),
    total: parseInt(resultMatch[2], 10),
    passes,
    failures,
    categories,
  };
}

function runValidator(corpus: string, noStreaming: boolean): RunSummary {
  const manifest: { samples: Sample[] } = JSON.parse(
    readFileSync(resolve(ROOT, `../../benchmark/${corpus}/manifest.json`), "utf-8"),
  );
  const commandArgs = [
    "test/validate-streaming.ts",
    `--corpus=${corpus}`,
    ...(noStreaming ? ["--no-streaming"] : []),
  ];

  try {
    const stdout = execFileSync(tsxPath, commandArgs, {
      cwd: ROOT,
      encoding: "utf-8",
      stdio: ["ignore", "pipe", "pipe"],
    });
    return parseRun(stdout, manifest.samples);
  } catch (error) {
    const stdout =
      typeof error === "object" &&
      error !== null &&
      "stdout" in error &&
      typeof error.stdout === "string"
        ? error.stdout
        : "";
    if (!stdout) throw error;
    return parseRun(stdout, manifest.samples);
  }
}

function summarize(label: string, runs: RunSummary[]): {
  medianCorrect: number;
  minCorrect: number;
  total: number;
  categoryPct: Record<string, number>;
} {
  const totals = runs.map((run) => run.correct);
  const total = runs[0]?.total ?? 0;
  const categories: Record<string, number[]> = {};
  for (const run of runs) {
    for (const [category, stats] of Object.entries(run.categories)) {
      categories[category] ??= [];
      categories[category].push(stats.total ? stats.correct / stats.total : 0);
    }
  }

  const categoryPct = Object.fromEntries(
    Object.entries(categories).map(([category, values]) => [
      category,
      Math.round(median(values) * 1000) / 10,
    ]),
  );

  console.log(`\n${label}`);
  for (const [index, run] of runs.entries()) {
    console.log(`  run ${index + 1}: ${run.correct}/${run.total}`);
  }
  console.log(`  median: ${median(totals)}/${total}`);
  console.log(`  min:    ${Math.min(...totals)}/${total}`);
  for (const [category, pct] of Object.entries(categoryPct)) {
    console.log(`  ${category}: ${pct}%`);
  }

  return {
    medianCorrect: median(totals),
    minCorrect: Math.min(...totals),
    total,
    categoryPct,
  };
}

async function main() {
  const matrix = [
    { corpus: "test_corpus", noStreaming: false },
    { corpus: "test_corpus", noStreaming: true },
    { corpus: "test_corpus_v2", noStreaming: false },
    { corpus: "test_corpus_v2", noStreaming: true },
  ];

  let gateFailed = false;

  for (const entry of matrix) {
    const runs = Array.from({ length: repeats }, () =>
      runValidator(entry.corpus, entry.noStreaming),
    );
    const label = `${entry.noStreaming ? "non-streaming" : "streaming"} ${entry.corpus}`;
    const summary = summarize(label, runs);

    if (!entry.noStreaming && entry.corpus === "test_corpus") {
      const minCorrect = Math.min(...runs.map((r) => r.correct));
      const v1Pass = minCorrect >= 45;
      console.log(`  gate_v1_streaming: ${v1Pass ? "PASS" : "FAIL"} (min=${minCorrect}/53, need ≥45)`);
      gateFailed ||= !v1Pass;
    }

    if (!entry.noStreaming && entry.corpus === "test_corpus_v2") {
      const categoriesOk = Object.values(summary.categoryPct).every((pct) => pct >= 90);
      const gateOk =
        summary.medianCorrect >= 41 &&
        summary.minCorrect >= 40 &&
        categoriesOk;
      console.log(`  gate_v2_streaming: ${gateOk ? "PASS" : "FAIL"}`);
      gateFailed ||= !gateOk;
    }
  }

  if (gateFailed && failOnGate) {
    process.exit(1);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
