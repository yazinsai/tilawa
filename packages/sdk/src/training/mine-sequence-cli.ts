/** Offline, training-only engine replay. All output is an experiment artifact. */
import { readFileSync, writeFileSync, mkdirSync, realpathSync, existsSync, statSync, createReadStream } from "node:fs";
import { createHash } from "node:crypto";
import { resolve, relative } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "@tilawa/core";
import { loadNodeModel } from "../node.js";
import { createEngine } from "../session.js";
import { decodeWav } from "../wav.js";
import { MODEL_REGISTRY } from "../registry.js";
import { hasTranscript, mineSequenceNegatives, trainingTakes, trainingFeatures } from "./sequence-negatives.js";
import type { TrainingTake } from "./sequence-negatives.js";

const { values } = parseArgs({ options: { help: { type: "string" }, model: { type: "string" }, out: { type: "string" } } });
if (!values.help || !values.model || !values.out) throw new Error("--help, --model and --out are required");
const source = realpathSync(values.help), out = resolve(values.out);
if (!process.env.LAB_DATA || source !== realpathSync(resolve(process.env.LAB_DATA, "help-train")))
  throw new Error("Only the supervisor-provided help-train source is authorized");
if (existsSync(resolve(out, "mining.json"))) throw new Error("Mining inventory already frozen");
mkdirSync(out, { recursive: true });
const raw = readFileSync(resolve(source, "labels.jsonl"), "utf8");
const sourceTakes = raw.split("\n").filter(Boolean).map(s => JSON.parse(s)) as TrainingTake[];
const takes = trainingTakes(sourceTakes);
const excludedSpeakers = [...new Set(sourceTakes.filter(t => t.split !== "train" || t.exclude || t.excluded).map(t => t.speaker))].sort();
const stable = (s: string) => createHash("sha256").update(s).digest("hex");
const speakers = [...new Set(takes.map(t => t.speaker))].sort((a, b) => stable(a).localeCompare(stable(b)));
if (speakers.length < 2) throw new Error("Need at least two training speakers");
const reserved = new Set(speakers.slice(0, Math.ceil(speakers.length / 5)));
const fitPositives = takes.filter(t => !reserved.has(t.speaker) && hasTranscript(t));
const asset = (name: string) => JSON.parse(readFileSync(new URL(`../../../core/assets/${name}`, import.meta.url), "utf8"));
const reference = loadReferenceIndex(asset("ref-v2.json"));
const modelDirectory = resolve(values.model);
const primary = JSON.parse(readFileSync(resolve(modelDirectory, "manifest.json"), "utf8"));
// A missing teacher must not silently change which accusations are mined.
for (const declared of [primary, ...(MODEL_REGISTRY.auxiliary ?? []).flatMap(a => [a, ...a.externalData])]) {
  const file = resolve(modelDirectory, declared.file);
  if (statSync(file).size !== declared.bytes) throw new Error(`Model asset size mismatch: ${declared.file}`);
  const hash = createHash("sha256");
  for await (const bytes of createReadStream(file)) hash.update(bytes);
  if (hash.digest("hex") !== declared.sha256) throw new Error(`Model asset checksum mismatch: ${declared.file}`);
}
const model = await loadNodeModel(modelDirectory);
if (model.manifest.sampleRate !== 16000 || model.manifest.frameMs !== 40 || model.manifest.fbank.mels !== 80)
  throw new Error("Frontend geometry differs from the deployed listener");
const engine = createEngine({ reference, phrases: asset("phrases-v1.json"), model, recordEvidence: true });
const records = [];
try {
  for (const take of takes) {
    const file = realpathSync(resolve(source, take.wav));
    if (relative(source, file).startsWith("..")) throw new Error("Recording escapes authorized training root");
    const bytes = readFileSync(file), wav = decodeWav(bytes);
    const features = trainingFeatures(wav.samples, wav.sampleRate);
    const flat = new Float32Array(features.length * 80);
    features.forEach((f, i) => flat.set(f, i * 80));
    const featureFile = resolve(out, `${stable(take.id)}.fbank`);
    const featureBytes = Buffer.alloc(flat.length * 4);
    flat.forEach((v, i) => featureBytes.writeFloatLE(v, i * 4));
    writeFileSync(featureFile, featureBytes);
    let negatives: ReturnType<typeof mineSequenceNegatives> = [];
    if (hasTranscript(take)) {
      const session = engine.startFreeRecitation();
      const step = Math.round(wav.sampleRate * .32);
      for (let at = 0; at < wav.samples.length; at += step)
        await session.pushAudio(wav.samples.subarray(at, at + step), wav.sampleRate);
      await session.end();
      negatives = mineSequenceNegatives(reference, take, session.mistakes(),
        session.evidence().flatMap(e => e.phonemes), fitPositives);
    }
    records.push({ ...take, file, source_sha256: createHash("sha256").update(bytes).digest("hex"),
      feature_file: featureFile, feature_sha256: createHash("sha256").update(featureBytes).digest("hex"),
      supervised: hasTranscript(take),
      partition: reserved.has(take.speaker) ? "validation" : "train", negatives });
    console.log(JSON.stringify({ recording: take.id, negatives: negatives.length }));
  }
} finally { await model.backend.release(); }
writeFileSync(resolve(out, "mining.json"), JSON.stringify({ schema: 1, labels_sha256: stable(raw), labels_root: source,
  reference_sha256: createHash("sha256").update(readFileSync(new URL("../../../core/assets/ref-v2.json", import.meta.url))).digest("hex"),
  primary_sha256: model.manifest.sha256, reserved: [...reserved], excluded_speakers: excludedSpeakers, records }, null, 1));
console.log(JSON.stringify({ recordings: records.length,
  trainNegatives: records.filter(r => r.partition === "train").flatMap(r => r.negatives).length,
  validationNegatives: records.filter(r => r.partition === "validation").flatMap(r => r.negatives).length }));
