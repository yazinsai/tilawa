/** Training-only audio truncations and matched complete controls, decoded by the shipped listener. */
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { align, interruptionFeatures, loadReferenceIndex, minGapFrames, PosteriorFrames } from "@tilawa/core";
import type { PhonemeEvidence, ReferenceAsset, DecodedPhoneme } from "@tilawa/core";
import { createListener } from "../listener.js";
import { loadNodeModel } from "../node.js";
import { decodeWav } from "../wav.js";

interface Label { id: string; speaker: string; split: string; kind: string; wav: string; ayahs: string[]; events?: unknown[]; non_errors?: unknown[]; exclude?: unknown }
const { values } = parseArgs({ options: { "help-dir": { type: "string" }, model: { type: "string" }, out: { type: "string" } } });
if (!values["help-dir"] || !values.model || !values.out) throw new Error("--help-dir, --model and --out are required");
const dir = resolve(values["help-dir"]), out = resolve(values.out);
const labels: Label[] = readFileSync(resolve(dir, "labels.jsonl"), "utf8").trim().split("\n").map(l => JSON.parse(l));
const held = new Set(labels.filter(l => l.split !== "train").map(l => l.speaker));
const selected = labels.filter(l => l.split === "train" && !held.has(l.speaker) && l.kind === "none" && !l.events?.length && !l.non_errors?.length && !l.exclude)
  .sort((a, b) => a.id.localeCompare(b.id));
const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../../core/assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const model = await loadNodeModel(resolve(values.model), { threads: 1 });
if (model.manifest.inventory !== index.inventoryId) throw new Error("Training model inventory mismatch");
const shard = Number(process.env.LAB_SHARD ?? 0), shards = Number(process.env.LAB_SHARDS ?? 1);
mkdirSync(out, { recursive: true });
const file = resolve(out, `examples-${shard}.jsonl`);
writeFileSync(file, "");
const manifestHash = createHash("sha256").update(readFileSync(resolve(values.model, "manifest.json"))).digest("hex");

async function decode(samples: Float32Array, rate: number) {
  const listener = createListener(model, { posteriors: true });
  const chunks: PhonemeEvidence[] = [];
  const size = Math.round(rate * 0.32);
  for (let at = 0; at < samples.length; at += size) {
    const e = await listener.push(samples.subarray(at, at + size), rate);
    if (e) chunks.push(e);
  }
  const tail = await listener.end();
  if (tail) chunks.push(tail);
  const frames = new PosteriorFrames(index.tokens.size);
  for (const c of chunks) if (c.posteriors) frames.append(c.posteriors);
  // Rejoin a spike split at a listener chunk boundary.
  const phonemes: DecodedPhoneme[] = [];
  for (const p of chunks.flatMap(c => c.phonemes)) {
    const prev = phonemes.at(-1);
    if (prev && prev.p === p.p && prev.end + 1 === p.start) prev.end = p.end;
    else phonemes.push({ ...p });
  }
  return { frames, phonemes };
}

for (const [recording, label] of selected.entries()) {
  if (recording % shards !== shard) continue;
  const audio = readFileSync(resolve(dir, label.wav));
  const wav = decodeWav(audio), original = await decode(wav.samples, wav.sampleRate);
  const ws = index.words.filter(w => label.ayahs.includes(`${w.surah}:${w.ayah}`) && w.end > w.start);
  if (!ws.length) continue;
  const forms = ws.map(w => index.phonemes.slice(w.start, w.end));
  forms[forms.length - 1] = index.isolatedWordForms?.[index.words.indexOf(ws.at(-1)!)] ?? forms.at(-1)!;
  // None-labelled transcript constrains word bounds; no tracker visits supply supervision.
  const options = [forms, forms.map(w => [...w].filter((c, i) => !i || c !== w[i-1]).join(""))];
  const heard = original.phonemes.map(p => p.p).join("");
  const aligned = options.map(words => ({ words, path: align(words.join(""), heard) }));
  aligned.sort((a, b) => a.path.filter(p => p.op !== "match").length - b.path.filter(p => p.op !== "match").length);
  const best = aligned[0];
  let offset = 0;
  const spans = best.words.map(word => {
    const start = offset;
    offset += word.length;
    const matches = best.path.filter(p => p.op === "match" && p.refAt >= start && p.refAt < offset);
    const obs = matches.map(p => "obsAt" in p ? p.obsAt : -1);
    return { from: original.phonemes[obs[0]]?.start ?? -1, to: (original.phonemes[obs.at(-1)!]?.end ?? -2) + 1,
      coverage: matches.length / word.length };
  });
  const audioHash = createHash("sha256").update(audio).digest("hex");
  let count = 0;
  const save = (features: number[] | null, target: number, kind: string) => {
    if (!features) return;
    appendFileSync(file, JSON.stringify({ construction: "interruption-v1", split: "train", speaker: label.speaker,
      sourceSha256: audioHash, modelManifestSha256: manifestHash, kind, features, label: target }) + "\n");
    count++;
  };
  // Complete speech also supplies within-word controls; identities and reference locations never enter features.
  for (let i = 1; i < original.phonemes.length; i++) save(interruptionFeatures(original.frames, original.phonemes[i-1], original.phonemes[i], model.manifest.frameMs), 0, "natural-complete");
  let transitions = 0;
  for (let i = 0; i + 1 < spans.length && transitions < 6; i++) {
    const a = spans[i], b = spans[i+1];
    if (a.coverage < 0.8 || b.coverage < 0.8 || a.from < 0 || b.from < a.to ||
      b.from-a.to >= minGapFrames(model.manifest.frameMs) || (a.to-a.from)*model.manifest.frameMs < 400) continue;
    transitions++;
    const frameSample = wav.sampleRate * model.manifest.frameMs / 1000;
    const start = Math.max(0, Math.floor(a.from * frameSample - 0.64*wav.sampleRate));
    const resume = Math.floor(b.from * frameSample);
    const end = Math.min(wav.samples.length, Math.floor((b.to * frameSample) + 0.32*wav.sampleRate));
    // Both classes receive exactly the same edit operation, gap and resumed speech.
    // Only the source word's completion differs; each pair stays with its speaker.
    const gap = Math.round(Math.min(80, Math.max(0, (b.from-a.to)*model.manifest.frameMs)) * wav.sampleRate / 1000);
    for (const fraction of [1, 1/3, 2/3]) {
      // Keep the complete control through the next onset, including the source's hidden acoustic tail.
      const cut = fraction === 1 ? resume : Math.floor((a.from + fraction*(a.to-a.from)) * frameSample);
      const joined = new Float32Array(cut-start + gap + end-resume);
      joined.set(wav.samples.subarray(start, cut));
      joined.set(wav.samples.subarray(resume, end), cut-start+gap);
      const e = await decode(joined, wav.sampleRate);
      const resumeFrame = (cut-start+gap)/frameSample;
      const right = e.phonemes.findIndex(p => p.start >= resumeFrame);
      if (right <= 0) continue;
      save(interruptionFeatures(e.frames, e.phonemes[right-1], e.phonemes[right], model.manifest.frameMs), fraction === 1 ? 0 : 1,
        fraction === 1 ? "spliced-complete" : "truncated-resumed");
    }
  }
  console.log(JSON.stringify({ recording: label.id, examples: count, transitions }));
}
await model.backend.release();
