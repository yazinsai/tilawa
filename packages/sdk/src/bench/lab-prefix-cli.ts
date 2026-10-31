/** Practice-only acoustic decoder experiment; consumes the deployed model's unmodified top-5 probabilities. */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { createHash } from "node:crypto";
import { createPrefixDecoder } from "../ctc-prefix.js";
const root = process.env.LAB_DATA ?? "/lab/data";
const manifest = JSON.parse(readFileSync(`${root}/lab-v1.5/manifest-v1.json`, "utf8"));
const sessions = manifest.sessions.filter((s: { split: string }) => s.split === "practice");
const item = sessions[Number(process.env.LAB_SHARD ?? 0)];
if (!item) throw new Error("No practice session for shard");
const asset = JSON.parse(readFileSync(new URL("../../../core/assets/ref-v2.json", import.meta.url), "utf8"));
const tokens: string = asset.inventory.tokens;
const records = [];
for (const variant of ["phone", "studio"]) {
  const path = `${root}/lab-v1.5/sessions/${item.id}.spliced-clean-int8.${variant}.evidence.json`;
  const original = readFileSync(path, "utf8"), ev = JSON.parse(original), post = ev.posteriors;
  if (!post?.ids || !post?.probs) throw new Error("Missing acoustic probabilities");
  const decoder = createPrefixDecoder(tokens, 8, 25), phonemes = [];
  const frames = post.ids.length / post.k;
  for (let f = 0; f < frames; f++) {
    const row: [number, number][] = [];
    for (let j = 0; j < post.k; j++) row.push([post.ids[f * post.k + j], post.probs[f * post.k + j]]);
    decoder.push(row, f); phonemes.push(...decoder.take());
  }
  phonemes.push(...decoder.end());
  const out = `${root}/lab-v1.5/sessions/${item.id}.spliced-clean-int8-beam8.${variant}.evidence.json`;
  writeFileSync(out, JSON.stringify({ ...ev, phonemes }));
  records.push({ id: item.id, variant, input: path, sha256: createHash("sha256").update(original).digest("hex"), output: out,
    width: 8, lagFrames: 25, frameMs: ev.frameMs, inputPhones: ev.phonemes.length, outputPhones: phonemes.length });
}
mkdirSync(process.env.LAB_OUT!, { recursive: true });
writeFileSync(`${process.env.LAB_OUT}/decode.json`, JSON.stringify(records));
console.log(item.id, records.map(r=>`${r.variant}: ${r.inputPhones} -> ${r.outputPhones}`).join(", "));
