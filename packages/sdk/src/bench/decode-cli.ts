/**
 * `pnpm decode --model <dir> --wavs <dir> --out <dir>`: every WAV through a free-recitation session on onnxruntime-node,
 * writing its evidence as `<out>/<name>.evidence.json`, the same way `bench:help` does for the help clips. Used for
 * student decodes of the lead-in audio (`tools/leadin_build.py audio`).
 */
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { basename, resolve } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "@tilawa/core";
import type { PhraseAsset, ReferenceAsset } from "@tilawa/core";
import { loadNodeModel } from "../node.js";
import { measureLatency } from "../latency.js";
import { createEngine } from "../session.js";
import { decodeWav } from "../wav.js";

const { values } = parseArgs({ options: {
  model: { type: "string" },
  wavs: { type: "string" },
  out: { type: "string" },
  push: { type: "string", default: "20", description: "ms per push" },
  "evidence-ms": { type: "string" },
} });
if (!values.model || !values.wavs || !values.out) throw new Error("--model, --wavs and --out are required");
const asset = (p: string) => JSON.parse(readFileSync(new URL(`../../../core/assets/${p}`, import.meta.url), "utf8"));
const reference = loadReferenceIndex(asset("ref-v2.json") as ReferenceAsset);
const phrases = asset("phrases-v1.json") as PhraseAsset;
const out = resolve(values.out);
mkdirSync(out, { recursive: true });
const model = await loadNodeModel(resolve(values.model));
const engine = createEngine({ reference, phrases, model, recordEvidence: true });
const wavs = readdirSync(resolve(values.wavs)).filter((f) => f.endsWith(".wav")).sort();
let done = 0;
for (const f of wavs) {
  const dst = resolve(out, `${basename(f, ".wav")}.evidence.json`);
  if (existsSync(dst)) continue;
  const wav = decodeWav(readFileSync(resolve(values.wavs, f)));
  const s = engine.startFreeRecitation({ evidenceMs: values["evidence-ms"] ? Number(values["evidence-ms"]) : undefined });
  await measureLatency(s, wav.samples, wav.sampleRate, Number(values.push));
  const fbankFrames = 1 + Math.floor((wav.samples.length - 400) / 160);
  const nFrames = Math.floor((Math.floor((fbankFrames - 7) / 2) + 1) / 2);
  writeFileSync(dst, JSON.stringify({ inventory: reference.inventoryId, frameMs: model.manifest.frameMs, nFrames, phonemes: s.evidence().flatMap((e) => e.phonemes) }));
  if (++done % 25 === 0) console.log(`${done}/${wavs.length}`);
}
console.log(`${wavs.length} wavs -> ${out}`);
