// Temporary, training-only decode helper. Removed from the final experiment diff.
import { readFileSync, mkdirSync, writeFileSync, existsSync } from "node:fs";
import { loadNodeModel, readWav } from "../../sdk/src/node.js";
import { createListener } from "../../sdk/src/listener.js";
import { loadReferenceIndex } from "../src/reference.js";
import { editTrainingReference } from "../src/edit-alignment-training.js";
const data = process.env.LAB_DATA!;
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const labels = readFileSync(`${data}/help-train/labels.jsonl`, "utf8").trim().split("\n").map(s => JSON.parse(s));
if (labels.some(r => r.split !== "train")) throw new Error("Held-out training row");
const eligible = labels.filter(r => r.device === "phone" && editTrainingReference(index, r));
const shard = Number(process.env.LAB_SHARD), shards = Number(process.env.LAB_SHARDS);
const dir = "/scratch/tracker-marginal-edit-alignments/decodes";
mkdirSync(dir, { recursive: true });
const model = await loadNodeModel(`${data}/models/spliced-clean-int8`, { threads: 1 });
let count = 0;
for (const [i, row] of eligible.entries()) {
  if (i % shards !== shard) continue;
  const path = `${dir}/${row.id}.json`;
  if (!existsSync(path)) {
    const audio = await readWav(`${data}/help-train/${row.wav}`);
    const listener = createListener(model);
    let heard = "";
    const stride = Math.round(audio.sampleRate * 0.02);
    for (let at = 0; at < audio.samples.length; at += stride) {
      const e = await listener.push(audio.samples.subarray(at, at + stride), audio.sampleRate);
      if (e) heard += e.phonemes.map(p => p.p).join("");
    }
    const tail = await listener.end();
    if (tail) heard += tail.phonemes.map(p => p.p).join("");
    writeFileSync(path, JSON.stringify({ id: row.id, speaker: row.speaker, kind: row.kind,
      written: editTrainingReference(index, row), heard }));
  }
  count++;
  if (count % 10 === 0) console.log("decoded", count);
}
await model.backend.release();
console.log(JSON.stringify({ shard, count, eligible: eligible.length }));
