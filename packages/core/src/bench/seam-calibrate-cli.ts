/**
 * Operating point: the highest seam score on any held-out clean gold-excluded seam.
 * A seam fires only strictly above that, so those seams stay silent. Writes the weights module.
 *
 *   pnpm --filter @tilawa/core exec tsx src/bench/seam-calibrate-cli.ts --weights <json> --feats <jsonl> --out <ts>
 */
import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { scoreSeam } from "../seam-tagger.js";
import type { SeamToken, SeamWeights } from "../seam-tagger.js";

const { values } = parseArgs({ options: {
  weights: { type: "string" },
  feats: { type: "string" },
  out: { type: "string" },
} });
if (!values.weights || !values.feats || !values.out) throw new Error("--weights, --feats and --out are required");

const weights = JSON.parse(readFileSync(values.weights, "utf8")) as SeamWeights;
const lines = readFileSync(values.feats, "utf8").split("\n").filter(Boolean);
let maxClean = 0;
let clean = 0;
let pos = 0;
let posAbove = 0;
const posScores: number[] = [];
for (const line of lines) {
  const row = JSON.parse(line) as { split: string; clean: boolean; span: [number, number]; tok: SeamToken[] };
  const score = scoreSeam(row.tok, row.span, weights);
  if (row.clean) {
    clean++;
    if (score > maxClean) maxClean = score;
  } else if (!row.clean) {
    pos++;
    posScores.push(score);
  }
}
if (!clean) maxClean = 1;
weights.threshold = maxClean;
posAbove = posScores.filter((s) => s > maxClean).length;
posScores.sort((a, b) => a - b);
const mid = posScores.length ? posScores[Math.floor(posScores.length / 2)] : 0;
console.log(JSON.stringify({ clean, maxClean, pos, posAbove, posMedian: mid }));
const body = JSON.stringify(weights);
writeFileSync(values.out, `import type { SeamWeights } from "./seam-tagger.js";\n\nexport const SEAM_WEIGHTS: SeamWeights = ${body};\n`);
console.log(`threshold ${maxClean} on ${clean} clean seams; ${posAbove}/${pos} training insertions score above it`);
