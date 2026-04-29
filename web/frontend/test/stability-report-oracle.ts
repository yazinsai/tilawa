/**
 * Streaming stability report with an explicit utterance-level oracle verifier.
 *
 * This diagnostic answers: "what would streaming SeqAcc be if a high-accuracy
 * final verifier reconciled each completed utterance?" It intentionally does
 * NOT change RecitationTracker; the output is labeled separately from shipped
 * browser/RN streaming.
 *
 * Usage:
 *   npx tsx test/stability-report-oracle.ts --repeats=2 --corpus=test_corpus_v3 --json=test/oracle-v3.json
 *   npx tsx test/stability-report-oracle.ts --oracle-results=../../benchmark/results/r7-v3-batch.json
 */

import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");

const args = process.argv.slice(2);
const repeatsArg = args.find((a) => a.startsWith("--repeats="));
let repeats = repeatsArg ? Number(repeatsArg.split("=")[1]) : 1;
if (!Number.isFinite(repeats) || repeats < 1) repeats = 1;
if (repeats > 2) repeats = 2;
const corpusArg = args.find((a) => a.startsWith("--corpus="));
const corpusName = corpusArg ? corpusArg.split("=")[1] : "test_corpus_v3";
const jsonArg = args.find((a) => a.startsWith("--json="));
const jsonOutPath = jsonArg ? jsonArg.split("=")[1] : null;
const oracleArg = args.find((a) => a.startsWith("--oracle-results="));
const oracleResultsPath = oracleArg
  ? resolve(process.cwd(), oracleArg.split("=")[1])
  : resolve(ROOT, "../../benchmark/results/2026-04-29_103225.json");
const allowMissingOracle = args.includes("--allow-missing-oracle");

interface Sample {
  id: string;
  category: string;
  expected_verses: { surah: number; ayah: number }[];
}

interface OraclePrediction {
  surah: number;
  ayah: number;
  ayah_end?: number | null;
}

interface OracleSample {
  id: string;
  predicted?: OraclePrediction[];
  raw_predict?: OraclePrediction;
}

interface OracleExperiment {
  per_sample?: OracleSample[];
}

function refsFromPrediction(prediction?: OraclePrediction[] | OraclePrediction): string[] {
  if (!prediction) return [];
  const predictions = Array.isArray(prediction) ? prediction : [prediction];
  const refs: string[] = [];
  for (const pred of predictions) {
    if (!pred.surah || !pred.ayah) continue;
    const end = pred.ayah_end && pred.ayah_end >= pred.ayah ? pred.ayah_end : pred.ayah;
    for (let ayah = pred.ayah; ayah <= end; ayah++) {
      const ref = `${pred.surah}:${ayah}`;
      if (!refs.includes(ref)) refs.push(ref);
    }
  }
  return refs;
}

function median(values: number[]): number {
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

const manifest: { samples: Sample[] } = JSON.parse(
  readFileSync(resolve(ROOT, `../../benchmark/${corpusName}/manifest.json`), "utf-8"),
);
const oracle: OracleExperiment[] = JSON.parse(readFileSync(oracleResultsPath, "utf-8"));
const oracleById = new Map<string, string[]>();
for (const sample of oracle[0]?.per_sample ?? []) {
  oracleById.set(sample.id, refsFromPrediction(sample.predicted ?? sample.raw_predict));
}

const missingOracleSamples = manifest.samples
  .map((sample) => sample.id)
  .filter((id) => !oracleById.has(id));
if (missingOracleSamples.length > 0 && !allowMissingOracle) {
  throw new Error(
    `Oracle results do not cover ${missingOracleSamples.length}/${manifest.samples.length} ` +
      `samples in ${corpusName}. First missing: ${missingOracleSamples.slice(0, 8).join(", ")}. ` +
      `Pass --allow-missing-oracle only for diagnostics.`,
  );
}

const perRunSeqAcc: number[] = [];
const perRunRecall: number[] = [];
const perRunPrecision: number[] = [];
const perRunExactCorrect: number[] = [];
const samples = manifest.samples.map((sample) => {
  const expected = sample.expected_verses.map((v) => `${v.surah}:${v.ayah}`);
  const predicted = oracleById.get(sample.id) ?? [];
  const expectedSet = new Set(expected);
  const predictedSet = new Set(predicted);
  const matched = expected.filter((ref) => predictedSet.has(ref)).length;
  const recall = expected.length > 0 ? matched / expected.length : 1;
  const precision = predicted.length > 0 ? matched / predicted.length : 1;
  const seqAcc =
    expectedSet.size === predictedSet.size &&
    expected.every((ref) => predictedSet.has(ref))
      ? 1
      : 0;
  return {
    id: sample.id,
    category: sample.category,
    expectedVerses: expected,
    runs: Array.from({ length: repeats }, () => ({
      passed: expected.every((ref) => predictedSet.has(ref)),
      exactPassed: seqAcc === 1,
      discoveredVerses: predicted,
      recall,
      precision,
      seqAcc,
    })),
    passRate: recall === 1 ? 1 : 0,
    classification: recall === 1 ? "stable-pass" : "stable-fail",
    exactPassRate: seqAcc === 1 ? 1 : 0,
    exactClassification: seqAcc === 1 ? "exact-stable-pass" : "exact-stable-fail",
    medianPrecision: precision,
    medianRecall: recall,
  };
});

for (let run = 0; run < repeats; run++) {
  let exact = 0;
  let totalP = 0;
  let totalR = 0;
  let totalS = 0;
  for (const sample of samples) {
    const r = sample.runs[run];
    if (r.exactPassed) exact++;
    totalP += r.precision;
    totalR += r.recall;
    totalS += r.seqAcc;
  }
  perRunExactCorrect.push(exact);
  perRunPrecision.push(totalP / samples.length);
  perRunRecall.push(totalR / samples.length);
  perRunSeqAcc.push(totalS / samples.length);
}

const report = {
  corpus: corpusName,
  repeats,
  verifier: "w2v-phonemes/base-local-int8 full-utterance oracle",
  oracleResults: oracleResultsPath,
  samples,
  aggregate: {
    totalSamples: samples.length,
    exactStablePass: samples.filter((s) => s.exactPassRate === 1).length,
    exactStableFail: samples.filter((s) => s.exactPassRate === 0).length,
    exactFlaky: 0,
    medianPrecision: median(perRunPrecision),
    medianRecall: median(perRunRecall),
    medianSeqAcc: median(perRunSeqAcc),
    perRunExactCorrect,
    perRunPrecision,
    perRunRecall,
    perRunSeqAcc,
  },
};

console.log(`=== ORACLE STABILITY REPORT (${repeats} repeats, corpus: ${corpusName}) ===`);
console.log(`Oracle: ${oracleResultsPath}`);
console.log(`Per-run exact:   [${perRunExactCorrect.join(", ")}]`);
console.log(`Median precision: ${(report.aggregate.medianPrecision * 100).toFixed(1)}%`);
console.log(`Median recall:    ${(report.aggregate.medianRecall * 100).toFixed(1)}%`);
console.log(`Median SeqAcc:    ${(report.aggregate.medianSeqAcc * 100).toFixed(1)}%`);

if (jsonOutPath) {
  const outPath = resolve(process.cwd(), jsonOutPath);
  writeFileSync(outPath, JSON.stringify(report, null, 2));
  console.log(`JSON report saved to: ${outPath}`);
}
