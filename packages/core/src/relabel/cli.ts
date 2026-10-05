import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "../reference.js";
import type { ReferenceAsset } from "../reference.js";
import { relabelClip, summarize } from "./index.js";
import type { Pass1Clip } from "./index.js";

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    out: { type: "string" },
    posteriors: { type: "boolean", default: false },
    "prior-weight": { type: "string", default: "1" },
  },
});
if (positionals.length !== 1 || !values.out) {
  console.error("usage: relabel <pass1.jsonl> --out <dir> [--posteriors] [--prior-weight 0..1]");
  process.exit(2);
}

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v1.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const clips: Pass1Clip[] = readFileSync(positionals[0], "utf8")
  .split("\n")
  .filter(Boolean)
  .map((l) => JSON.parse(l));
clips.sort((a, b) => (a.shard < b.shard ? -1 : a.shard > b.shard ? 1 : a.row - b.row));

const options = { posteriors: values.posteriors, priorWeight: Number(values["prior-weight"]) };
const labels = clips.map((c) => relabelClip(index, c, options));
const summary = { ...summarize(labels), input: positionals[0], referenceVersion: index.version, options };

mkdirSync(values.out, { recursive: true });
const suffix = values.posteriors ? ".posteriors" : "";
writeFileSync(join(values.out, `proposed-labels${suffix}.jsonl`), labels.map((l) => JSON.stringify(l)).join("\n") + "\n");
writeFileSync(join(values.out, `proposed-summary${suffix}.json`), JSON.stringify(summary, null, 2) + "\n");
console.log(JSON.stringify(summary, null, 2));
