/**
 * `pnpm bench:help --model <dir>`: help eval clips through the session API on onnxruntime-node. Writes each clip's
 * evidence as `<out>/<clip>.evidence.json`, so `pnpm scoreboard --listener <name> --evidence <out>` in packages/core
 * scores the shipped runtime too, and prints both arms next to the PyTorch decode of the same clips.
 */
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "@tilawa/core";
import type { PhraseAsset, ReferenceAsset } from "@tilawa/core";
import { loadNodeModel } from "../node.js";
import { runHelpBench } from "./help.js";

const { values } = parseArgs({ options: {
  model: { type: "string", default: process.env.TILAWA_MODEL_DIR ?? "../../../tilawa/.scratch/models/spliced-clean" },
  help: { type: "string", default: process.env.HELP_DIR ?? "../../../tilawa/.scratch/help" },
  out: { type: "string", default: "../../../tilawa/.scratch/sdk-help/spliced-clean-onnx-node" },
  torch: { type: "string", default: "../../../tilawa/.scratch/gold/spliced-clean", description: "PyTorch evidence of the same clips" },
  report: { type: "string", default: "../../docs/benchmarks/sdk-help-spliced-clean-v1.json" },
  push: { type: "string", default: "320" },
  "evidence-ms": { type: "string", default: "1000" },
  limit: { type: "string" },
} });
const asset = (p: string) => JSON.parse(readFileSync(new URL(`../../../core/assets/${p}`, import.meta.url), "utf8"));
const reference = loadReferenceIndex(asset("ref-v2.json") as ReferenceAsset);
const phrases = asset("phrases-v1.json") as PhraseAsset;
const out = resolve(values.out!);
mkdirSync(out, { recursive: true });
const model = await loadNodeModel(resolve(values.model!));

let same = 0, compared = 0;
const res = await runHelpBench({
  helpDir: resolve(values.help!), reference, phrases, model, pushMs: Number(values.push), evidenceMs: Number(values["evidence-ms"]), limit: values.limit ? Number(values.limit) : undefined,
  onClip(c, done, total) {
    writeFileSync(resolve(out, `${c.id}.evidence.json`), JSON.stringify({ inventory: reference.inventoryId, frameMs: model.manifest.frameMs, nFrames: c.nFrames, phonemes: c.phonemes }));
    const torch = resolve(values.torch!, `${c.id}.evidence.json`);
    if (existsSync(torch)) {
      compared++;
      const t = JSON.parse(readFileSync(torch, "utf8")) as { phonemes: { p: string }[] };
      if (t.phonemes.map((p) => p.p).join("") === c.phonemes.map((p) => p.p).join("")) same++;
    }
    if (done % 50 === 0 || done === total) console.log(`${done}/${total} clips`);
  },
});
const pct = (x: number) => `${(100 * x).toFixed(1)}%`;
const summary = {
  model: model.manifest.id, format: model.manifest.format, runtime: model.backend.name, pushMs: Number(values.push), evidenceMs: Number(values["evidence-ms"]),
  clips: res.clips.length, audioS: res.seconds.audio, rtf: res.seconds.compute / res.seconds.audio / 2,
  decodeIdenticalToPyTorch: `${same}/${compared}`,
  free: res.free, declared: res.declared,
};
writeFileSync(resolve(values.report!), JSON.stringify(summary, null, 1) + "\n");
console.log(JSON.stringify(summary, null, 1));
console.log(`declared ${pct(res.declared.precision)} / ${pct(res.declared.recall)}; free ${pct(res.free.precision)} / ${pct(res.free.recall)}`);
