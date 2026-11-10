/** Fit only on authorized none-labelled training-speaker recordings, never on practice evidence. */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { loadReferenceIndex } from "@tilawa/core";
import type { DecodedPhoneme, ReferenceAsset } from "@tilawa/core";
import { fitPathChannel } from "../../core/src/path-risk.js";
import type { PathRecording } from "../../core/src/path-risk.js";
import { loadNodeModel, decodeWav } from "./node.js";
import { createListener } from "./listener.js";

const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT are required");
const cache = process.env.PATH_CHANNEL_CACHE ?? join(out, "clean-decodes");
mkdirSync(cache, { recursive: true });
const reference = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
interface Label {
  id: string; speaker: string; split: string; kind: string; scored: boolean; ayahs: string[]; wav: string;
  events: unknown[]; non_errors: unknown[]; excluded?: unknown;
}
const all = readFileSync(join(data, "help-train/labels.jsonl"), "utf8").trim().split("\n").map(x => JSON.parse(x)) as Label[];
const excludedSpeakers = new Set(all.filter(l => l.split !== "train").map(l => l.speaker));
const labels = all.filter(l => l.split === "train" && !excludedSpeakers.has(l.speaker) && l.kind === "none" && l.scored && !l.excluded && !l.events.length && !l.non_errors.length);
if (!labels.length) throw new Error("No eligible none-labelled training recordings");
const listenerModel = await loadNodeModel(join(data, "models/spliced-clean-int8"));
try {
  const records: PathRecording[] = [];
  for (const label of labels) {
    const file = join(cache, `${listenerModel.manifest.sha256}-${label.id}.json`);
    let phonemes: DecodedPhoneme[];
    if (existsSync(file)) phonemes = JSON.parse(readFileSync(file, "utf8"));
    else {
      const wav = decodeWav(readFileSync(join(data, "help-train", label.wav)));
      const listener = createListener(listenerModel);
      phonemes = [...(await listener.push(wav.samples, wav.sampleRate))?.phonemes ?? [], ...(await listener.end())?.phonemes ?? []];
      writeFileSync(file, JSON.stringify(phonemes));
    }
    const ayahs = new Set(label.ayahs);
    const words = reference.words.filter(w => ayahs.has(`${w.surah}:${w.ayah}`));
    if (!words.length || !phonemes.length) throw new Error(`Empty eligible training recording: ${label.id}`);
    records.push({ written: reference.phonemes.slice(words[0].start, words.at(-1)!.end), heard: phonemes.map(p => p.p).join("") });
    console.log(`none-labelled training recording ${records.length}/${labels.length}: ${phonemes.length} phones`);
  }
  const channel = fitPathChannel(records, [...reference.tokens]);
  const report = { listener: listenerModel.manifest.id, modelSha256: listenerModel.manifest.sha256,
    labelsSha256: createHash("sha256").update(JSON.stringify(labels)).digest("hex"), clips: records.length,
    speakers: new Set(labels.map(l => l.speaker)).size, trainingOnly: true,
    alignment: "Existing directed weighted token edits, including duration-expanded reference symbols",
    estimator: "Jeffreys smoothing; operation probabilities and conditional identity rows normalized separately; no selected hyperparameters" };
  writeFileSync(join(out, "path-channel.ts"), `import type { PathChannel } from "./path-risk.js";\n\n/** Training-only channel; see experiment write-up for provenance. */\nexport const PATH_CHANNEL: PathChannel = ${JSON.stringify(channel)};\n`);
  writeFileSync(join(out, "fit-report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally {
  await listenerModel.backend.release();
}
