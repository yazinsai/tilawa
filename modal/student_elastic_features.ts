/** Extract native SDK observations from an explicitly audited training inventory.
 * Input JSONL: {file, speaker, kind:"none", split:"train", excluded:false, pronunciation}.
 * The inventory must be constructed from supplied split/label provenance; this
 * utility deliberately cannot reinterpret mistake scripts as transcripts.
 * Run using tsx; arguments: inventory.jsonl excluded-speakers.json output.jsonl.
 */
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { elasticFeatures } from "../packages/sdk/src/elastic-frequency.js";
import { decodeWav } from "../packages/sdk/src/wav.js";
import { Resampler } from "../packages/sdk/src/audio.js";

const [input, excludedPath, output] = process.argv.slice(2);
if (!input || !excludedPath || !output) throw new Error("Expected audited inventory, excluded speakers, and output paths");
const excluded = new Set<string>(JSON.parse(await readFile(excludedPath, "utf8")));
const lines: string[] = [];
for (const line of (await readFile(input, "utf8")).split("\n").filter(Boolean)) {
  const row = JSON.parse(line);
  if (row.kind !== "none" || row.split !== "train" || row.excluded !== false || typeof row.speaker !== "string" ||
    excluded.has(row.speaker) || typeof row.pronunciation !== "string" || !row.pronunciation.length || typeof row.file !== "string")
    throw new Error("Only audited none-labelled training-speaker recordings may supply acoustic templates");
  const bytes = await readFile(resolve(dirname(input), row.file));
  const wav = decodeWav(bytes);
  let audio = wav.samples;
  if (wav.sampleRate !== 16000) {
    const resampler = new Resampler(wav.sampleRate, 16000), first = resampler.push(audio), tail = resampler.flush();
    audio = new Float32Array(first.length + tail.length); audio.set(first); audio.set(tail, first.length);
  }
  lines.push(JSON.stringify({ ...row, audioSha256: createHash("sha256").update(bytes).digest("hex"), features: elasticFeatures(audio) }));
}
await writeFile(output, lines.join("\n") + "\n");
