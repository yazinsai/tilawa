// Offline parity of the SDK structural rules with the lab prototypes.
//
// Reads tracker_correction_eval rows (with ZIPFORMER_TOKENS=1 tokens) and writes, per set, the
// SDK's candidates in the lab scripts' JSONL formats, so they diff against
// `ayah_order.py detect` / `similar_verse.py detect` output field by field.
//
//   tsx structural-parity.ts --run /tmp/sr/base/nopass --out /tmp/sr/ts/nopass --passage tracker --sets a,b
// Private data only: --run / --out live outside the repo.

import { readFileSync, mkdirSync, writeFileSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { QuranCorpus, StructuralRules, type StructuralIndexJson, type TimedToken } from "@tilawa/core";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..", "..", "..");
const arg = (k: string, d = ""): string => {
  const i = process.argv.indexOf(`--${k}`);
  return i >= 0 ? process.argv[i + 1]! : d;
};
const run = arg("run");
const out = arg("out");
const mode = arg("passage", "tracker");
const sets = arg("sets").split(",").filter(Boolean);
const corpus = new QuranCorpus(JSON.parse(readFileSync(process.env.ZIPFORMER_CORPUS
  ?? path.join(ROOT, "lab", "data", "zipformer", "quran.json"), "utf8")));
const index = JSON.parse(readFileSync(path.join(ROOT, "packages", "core", "src", "recitation", "structural-index.json"), "utf8")) as StructuralIndexJson;
const rules = new StructuralRules(corpus, index);

type Row = { id: string; split?: string; error?: string; tokens?: Array<[string, number, number]>; verses?: number[][];
  expected_verses?: Array<{ surah: number; ayah: number }>; duration_s?: number };

mkdirSync(out, { recursive: true });
const t0 = performance.now();
for (const name of sets) {
  const src = path.join(run, `${name}.jsonl`);
  if (!existsSync(src)) continue;
  const ao: string[] = [];
  const sv: string[] = [];
  for (const line of readFileSync(src, "utf8").split("\n")) {
    if (!line.trim()) continue;
    const row = JSON.parse(line) as Row;
    if (row.error) continue;
    const tokens: TimedToken[] = (row.tokens ?? []).map(([sym, frame, t]) => ({ sym, frame, t }));
    const ev = row.expected_verses ?? [];
    const surahs = new Set(ev.map((v) => Number(v.surah)));
    const single = mode === "expected" && surahs.size === 1;
    const ayahs = ev.map((v) => Number(v.ayah));
    const expected = single ? { surah: [...surahs][0]!, ayah: Math.min(...ayahs), ayahEnd: Math.max(...ayahs) } : null;
    const verses = (row.verses ?? []).map((v) => [Number(v[0]), Number(v[1])] as const);
    // ayah_order.window_of: the expected passage (one surah) or nothing; else the tracker's first verse.
    const win = mode === "expected" ? (expected ? rules.ayahOrderWindow(expected, null) : [])
      : rules.ayahOrderWindow(null, verses[0] ?? null);
    const cand = rules.ayahOrderCandidate(tokens, win);
    ao.push(JSON.stringify({ id: row.id, split: row.split ?? null, cand: cand && {
      margin: cand.margin, segs: cand.segs,
      jumps: cand.jumps.map((j) => ({ surah: j.surah, ayah: j.ayah, fit0: j.fit0, fit1: j.fit1, between: j.between,
        read_later: j.readLater, at: j.at, post_chars: j.postChars, pre_chars: j.preChars, pre_restart: j.preRestart,
        pre_on_skipped: j.preOnSkipped, skip_chars: j.skipChars, ident: j.ident })),
    } }));
    // similar_verse.passage_of
    let passage: Array<readonly [number, number]>;
    if (single) {
      passage = [];
      for (let a = expected!.ayah; a <= expected!.ayahEnd; a++) passage.push([expected!.surah, a]);
    } else {
      const seen = new Set<string>();
      passage = verses.filter(([s, a]) => !seen.has(`${s}:${a}`) && !!seen.add(`${s}:${a}`));
    }
    const known = mode === "expected" && ev.length > 0;
    const cands = rules.similarVerseCandidates(tokens, passage, known, Number(row.duration_s ?? 0));
    sv.push(JSON.stringify({ id: row.id, split: row.split ?? null, cands: cands.map((c) => ({
      slot: c.slot, surah: c.surah, ayah: c.ayah, word: c.word, kind: c.kind, family: c.family, span: c.span,
      from: c.from, margin: c.margin, margin_all: c.marginAll, known: c.known, at: c.at, loc: c.loc, ayah_d: c.ayahD,
    })) }));
  }
  mkdirSync(path.join(out, "ao"), { recursive: true });
  mkdirSync(path.join(out, "sv"), { recursive: true });
  writeFileSync(path.join(out, "ao", `${name}.jsonl`), ao.join("\n") + "\n");
  writeFileSync(path.join(out, "sv", `${name}.jsonl`), sv.join("\n") + "\n");
  console.log(name, ao.length, `${Math.round(performance.now() - t0)} ms`);
}
