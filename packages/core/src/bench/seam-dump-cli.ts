/**
 * Turn stored listener evidence into seam-tagger examples. The features are the ones checker.ts scores.
 *
 *   pnpm --filter @tilawa/core exec tsx src/bench/seam-dump-cli.ts --evidence <dir> --meta <dir> --out <file>
 *
 * Each `<id>.evidence.json` pairs with `<id>.json` in the meta dir:
 * `{ "id", "split": "train"|"cal", "clean": boolean, "insert": [startFrame, endFrame] | null }`.
 */
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { createChecker } from "../checker.js";
import { trackerInput, createTracker } from "../tracker.js";
import type { PhonemeEvidence } from "../evidence.js";
import { loadReferenceIndex } from "../reference.js";
import type { PhraseAsset, ReferenceAsset } from "../reference.js";
import { chunk } from "./index.js";
import type { QudRecording } from "./index.js";
import { OWN_LEFT, OWN_REPEAT, OWN_RIGHT, TAG_ADDED, TAG_NOISE, TAG_OWNED, TAG_REPEAT } from "../seam-tagger.js";
import type { SeamGeom, SeamToken } from "../seam-tagger.js";

const { values } = parseArgs({ options: {
  evidence: { type: "string" },
  meta: { type: "string" },
  out: { type: "string" },
} });
if (!values.evidence || !values.meta || !values.out) throw new Error("--evidence, --meta and --out are required");

const asset = (p: string) => JSON.parse(readFileSync(new URL("../../assets/" + p, import.meta.url), "utf8"));
const index = loadReferenceIndex(asset("ref-v2.json") as ReferenceAsset);
const phrases = asset("phrases-v1.json") as PhraseAsset;

interface Meta { id: string; split: "train" | "cal"; clean: boolean; insert: [number, number] | null; g?: number; ref?: string; host?: string; label?: "added" | "repeat" | "noise" }
interface Ev { inventory: string; frameMs: number; nFrames: number; phonemes: { p: string; start: number; end: number; conf: number }[]; posteriors?: { k: number; ids: number[]; probs: number[] } }

function label(tok: SeamToken, span: readonly [number, number], at: number, insert: [number, number] | null, phones: Ev["phonemes"], metaLabel: Meta["label"]): number {
  const ph = phones[tok.obs];
  const centre = ph ? (ph.start + ph.end) / 2 : -1;
  const inSpan = at >= span[0] && at < span[1];
  if (inSpan && insert && centre >= insert[0] && centre < insert[1]) {
    if (metaLabel === "repeat") return TAG_REPEAT;
    if (metaLabel === "noise") return TAG_NOISE;
    return TAG_ADDED;
  }
  if (tok.own === OWN_REPEAT) return TAG_REPEAT;
  if (tok.own === OWN_LEFT || tok.own === OWN_RIGHT) return TAG_OWNED;
  if (tok.pause || tok.conf < 0.5) return TAG_NOISE;
  return TAG_NOISE;
}

const shards = Number(process.env.LAB_SHARDS ?? 1);
const shard = Number(process.env.LAB_SHARD ?? 0);
const files = readdirSync(values.evidence).filter((f) => f.endsWith(".evidence.json")).filter((f) => {
  if (shards <= 1) return true;
  let h = 0;
  for (const c of f) h = (Math.imul(h, 33) + c.charCodeAt(0)) >>> 0;
  return h % shards === shard;
}).sort();
const lines: string[] = [];
let kept = 0;
let keptPos = 0;
let dropped = 0;
for (const f of files) {
  const id = f.slice(0, -".evidence.json".length);
  const metaPath = `${values.meta}/${id}.json`;
  let meta: Meta;
  try {
    meta = JSON.parse(readFileSync(metaPath, "utf8")) as Meta;
  } catch {
    continue;
  }
  const ev = JSON.parse(readFileSync(`${values.evidence}/${f}`, "utf8")) as Ev;
  if (!ev.phonemes?.length || !ev.nFrames) { dropped++; continue; }
  const rec = {
    inventory: ev.inventory, frameMs: ev.frameMs, nFrames: ev.nFrames, phonemes: ev.phonemes,
    posteriors: ev.posteriors ?? { k: 1, ids: [], probs: [] },
  } as QudRecording;
  const hasPost = !!ev.posteriors && ev.posteriors.ids.length >= ev.nFrames;
  let g = meta.g;
  let ref = meta.ref;
  if ((g === undefined || !ref) && meta.host) {
    try {
      const host = JSON.parse(readFileSync(`${values.meta}/neg-${meta.host.replaceAll(":", "-")}.json`, "utf8")) as Meta;
      g = host.g;
      ref = host.ref;
    } catch { /* free mode if the host span was not stored */ }
  }
  const covered = g !== undefined && ref ? index.words.filter((w) => w.end > g && w.start < g + ref.length) : [];
  const passage = covered.length >= 2
    ? { from: { surah: covered[0].surah, ayah: covered[0].ayah, word: covered[0].word },
        to: { surah: covered.at(-1)!.surah, ayah: covered.at(-1)!.ayah, word: covered.at(-1)!.word } }
    : undefined;
  const best = new Map<string, { geom: SeamGeom; phones: Ev["phonemes"] }>();
  const checker = createChecker(index, {
    phrases,
    ...(passage ? { passage } : {}),
    seamTrace: (geom) => {
      const key = `${geom.rightWord}:${geom.obsFrom}:${geom.obsTo}`;
      const prev = best.get(key);
      if (!prev || geom.tokens.length >= prev.geom.tokens.length) best.set(key, { geom, phones: ev.phonemes });
    },
  });
  const tracker = createTracker(index, { mode: "free", phrases });
  try {
    for (const e of chunk(rec, 0.04, hasPost)) {
      const input = trackerInput(e as PhonemeEvidence);
      checker.push(e as PhonemeEvidence, tracker.push(input));
    }
    const end = tracker.end();
    checker.end(end);
  } catch (err) {
    console.log(`skip ${id}: ${(err as Error).message}`);
    dropped++;
    continue;
  }
  for (const { geom } of best.values()) {
    const y = geom.tokens.map((t, i) => label(t, geom.span, i, meta.insert, ev.phonemes, meta.label));
    const added = y.slice(geom.span[0], geom.span[1]).filter((c) => c === TAG_ADDED).length;
    const planted = meta.insert ? y.slice(geom.span[0], geom.span[1]).filter((_, i) => {
      const tok = geom.tokens[geom.span[0] + i];
      const ph = ev.phonemes[tok.obs];
      const centre = ph ? (ph.start + ph.end) / 2 : -1;
      return centre >= meta.insert![0] && centre < meta.insert![1];
    }).length : 0;
    if (meta.insert && (meta.label === "repeat" || meta.label === "noise") && planted < 2) { dropped++; continue; }
    if (meta.insert && meta.label !== "repeat" && meta.label !== "noise" && added < 2) { dropped++; continue; }
    if (!meta.insert && meta.split === "train" && geom.tokens.length < 4) continue;
    lines.push(JSON.stringify({
      id: meta.id, split: meta.split, clean: meta.clean, span: geom.span,
      tok: geom.tokens.map((t, i) => ({ id: t.id, conf: t.conf, pause: t.pause, own: t.own, step: t.step, spanLen: t.spanLen, y: y[i] })),
    }));
    kept++;
    if (!meta.clean) keptPos++;
  }
}
writeFileSync(values.out, lines.join("\n") + (lines.length ? "\n" : ""));
console.log(`kept ${kept} seams (${keptPos} insertions) from ${files.length} clips, dropped ${dropped}`);
