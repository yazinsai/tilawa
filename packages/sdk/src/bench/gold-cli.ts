/**
 * `pnpm bench:gold --model <dir> --name <listener>`: the gold and silver PER clips (`docs/benchmarks/gold-per-v1.json`)
 * through the streaming listener on onnxruntime-node, written as `.scratch/gold/<name>/<id>.evidence.json` for
 * `tools/gold_set.py score`. The CPU counterpart of `modal/student_decode_cpu.py::gold`, for exported/quantised models.
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { createListener } from "../listener.js";
import { loadNodeModel } from "../node.js";

const { values } = parseArgs({ options: {
  model: { type: "string" },
  name: { type: "string" },
  main: { type: "string", default: process.env.MAIN_CHECKOUT ?? "../../../tilawa", description: "checkout holding .scratch" },
  help: { type: "string", default: process.env.HELP_DIR },
} });
if (!values.model || !values.name) throw new Error("--model and --name are required");
const main = resolve(values.main!);
const help = resolve(values.help ?? `${main}/.scratch/help`);
const gold = JSON.parse(readFileSync(new URL("../../../../docs/benchmarks/gold-per-v1.json", import.meta.url), "utf8")) as {
  slices: Record<string, { id: string; source: string; audio: string }[]>;
};
const out = resolve(main, ".scratch/gold", values.name);
mkdirSync(out, { recursive: true });
const model = await loadNodeModel(resolve(values.model));
const rows = Object.values(gold.slices).flat();
const pcm = (path: string) => {
  const b = execFileSync("ffmpeg", ["-v", "error", "-i", path, "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"], { maxBuffer: 1 << 30 });
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};
let done = 0;
for (const r of rows) {
  const audio = r.source === "tlog-bucket-review" ? resolve(main, ".scratch/v1-roadmap", r.audio) : resolve(help, r.audio);
  const listener = createListener(model);
  const samples = pcm(audio);
  const phonemes = [];
  for (let a = 0; a < samples.length; a += 5120) phonemes.push(...((await listener.push(samples.slice(a, a + 5120), 16000))?.phonemes ?? []));
  phonemes.push(...((await listener.end())?.phonemes ?? []));
  writeFileSync(resolve(out, `${r.id.replaceAll(":", "-")}.evidence.json`), JSON.stringify({ inventory: model.manifest.inventory, frameMs: model.manifest.frameMs, phonemes }));
  if (++done % 100 === 0 || done === rows.length) console.log(`${done}/${rows.length}`);
}
