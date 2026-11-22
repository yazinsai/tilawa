/**
 * Local-window warning confirmation (ticket 07): replays help clips with the checker's local-window check off and on.
 *
 *   tsx src/bench/local-window-cli.ts --split train --evidence <dev evidence dir> --nats off,0,1,2,3,5,8   # dev pilot
 *   tsx src/bench/local-window-cli.ts --split eval --nats off,3                                           # held-out, once
 *
 * `--evidence` holds `<clip>.evidence.json` by basename; without it, the teacher evidence beside each wav.
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { basename, resolve } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "../reference.js";
import type { PhraseAsset, ReferenceAsset } from "../reference.js";
import type { CheckerEvent, WordRef } from "../events.js";
import type { CheckerPolicy } from "../checker.js";
import { MEMORIZATION_CLASSES, scoreClip } from "./acceptance.js";
import type { AcceptanceClip, Arm, ClipScore, LabelEvent } from "./acceptance.js";
import type { QudRecording } from "./index.js";

const { values } = parseArgs({ options: {
  help: { type: "string", default: process.env.HELP_DIR ?? "../../.scratch/help" },
  evidence: { type: "string" },
  split: { type: "string", default: "eval" },
  nats: { type: "string", default: "off,3" },
  arms: { type: "string", default: "declared,free" },
  out: { type: "string" },
} });
const dir = resolve(values.help!);
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const phrases = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8")) as PhraseAsset;
const jsonl = (f: string) => readFileSync(resolve(dir, f), "utf8").trim().split("\n").map((s) => JSON.parse(s));
const letterPairs = new Set<string>(jsonl("raw/submissions.jsonl").filter((r) => (r.mistakes ?? []).some((m: any) => m.rule === "letter_confusion")).map((r) => r.id));
const path = (wav: string) => (values.evidence ? resolve(values.evidence, basename(wav)) : resolve(dir, wav)).replace(/\.wav$/, ".evidence.json");
const labels = jsonl("labels.jsonl").filter((l: any) => l.split === values.split && l.scored && existsSync(path(l.wav)));
const settings = values.nats!.split(",").map((s) => (s === "off" ? "off" : +s));
const arms = values.arms!.split(",") as Arm[];
const policy = (s: "off" | number): CheckerPolicy => (s === "off" ? { localWindow: false } : { localWindow: { nats: s } });

const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const sig = (e: { errorClass: string; location: WordRef }) => `${e.errorClass}/${key(e.location)}`;
const edgeWords = (ayahs: string[]) => {
  const ws = index.words.filter((w) => ayahs.includes(`${w.surah}:${w.ayah}`) && w.end > w.start);
  return new Set([key(ws[0]), key(ws.at(-1)!)]);
};

interface Clip { id: string; speaker: string; kind: string; edges: Set<string>; positives: LabelEvent[]; scores: Record<string, ClipScore> }
const clips: Clip[] = [];
for (const l of labels) {
  const evidence = JSON.parse(readFileSync(path(l.wav), "utf8")) as QudRecording;
  if (evidence.inventory !== index.inventoryId) throw new Error(`Inventory mismatch: ${l.id}`);
  const events = letterPairs.has(l.id) ? [] : l.events;
  const clip: AcceptanceClip = { id: l.id, ayahs: l.ayahs, evidence, events, nonErrors: l.non_errors.filter((n: any) => n.kind === "repeat").map((n: any) => n.location) };
  const scores: Record<string, ClipScore> = {};
  for (const arm of arms) for (const s of settings) scores[`${arm}/${s}`] = scoreClip(index, clip, { arm, phrases, policy: policy(s) });
  clips.push({ id: l.id, speaker: l.speaker, kind: l.kind, edges: edgeWords(l.ayahs), positives: events.filter((e: LabelEvent) => e.state === "confirmed" && MEMORIZATION_CLASSES.includes(e.errorClass)), scores });
  if (clips.length % 50 === 0) console.error(`${clips.length}/${labels.length}`);
}

const brief = (e: CheckerEvent) => ({ errorClass: e.errorClass, location: key(e.location), observed: e.observed, expected: e.expected });
function summary(k: string) {
  const s = (c: Clip) => c.scores[k];
  const n = (f: (c: Clip) => number) => clips.reduce((a, c) => a + f(c), 0);
  const perClass = Object.fromEntries(MEMORIZATION_CLASSES.map((cls) => [cls, {
    tp: n((c) => c.positives.filter((e) => e.errorClass === cls).length - s(c).misses.filter((e) => e.errorClass === cls).length),
    fp: n((c) => s(c).falsePositives.filter((e) => e.errorClass === cls).length),
    fn: n((c) => s(c).misses.filter((e) => e.errorClass === cls).length),
  }]));
  const fps = clips.flatMap((c) => s(c).falsePositives.map((e) => ({ clip: c.id, kind: c.kind, edge: c.edges.has(key(e.location)), ...brief(e) })));
  return {
    tp: n((c) => s(c).hits), fp: n((c) => s(c).falsePositives.length), fn: n((c) => s(c).misses.length), perClass,
    cleanClips: clips.filter((c) => c.kind === "none").length, cleanWarnings: n((c) => (c.kind === "none" ? s(c).confirmed.length : 0)),
    repeatClips: clips.filter((c) => c.kind === "repeat").length, repeatFalsePositives: n((c) => s(c).repeatFalsePositives),
    edgeFalsePositives: fps.filter((f) => f.edge).length,
    vowelClips: clips.filter((c) => c.kind === "vowel").length, vowelClipWarnings: n((c) => (c.kind === "vowel" ? s(c).confirmed.length : 0)),
    falsePositives: fps,
  };
}
const out: Record<string, unknown> = { split: values.split, clips: clips.length, speakers: new Set(clips.map((c) => c.speaker)).size, evidence: values.evidence ?? "teacher" };
for (const arm of arms) {
  const base = `${arm}/${settings[0]}`;
  for (const s of settings) {
    const k = `${arm}/${s}`;
    const newlyMissed = clips.flatMap((c) => c.scores[k].misses.filter((m) => !c.scores[base].misses.some((b) => sig(b) === sig(m))).map((m) => ({ clip: c.id, kind: c.kind, errorClass: m.errorClass, location: key(m.location) })));
    const removed = clips.flatMap((c) => c.scores[base].falsePositives.filter((b) => !c.scores[k].falsePositives.some((e) => sig(e) === sig(b))).map((e) => ({ clip: c.id, kind: c.kind, ...brief(e) })));
    out[k] = { ...summary(k), newlyMissed, removedFalsePositives: removed };
  }
}
if (values.out) writeFileSync(resolve(values.out), JSON.stringify(out, null, 1) + "\n");
for (const [k, v] of Object.entries(out)) {
  if (typeof v !== "object" || !v) continue;
  const x = v as ReturnType<typeof summary> & { newlyMissed: unknown[]; removedFalsePositives: unknown[] };
  console.log(`${k.padEnd(14)} ${x.tp}/${x.fp}/${x.fn}  ww ${x.perClass.wrong_word.tp}/${x.perClass.wrong_word.fp}/${x.perClass.wrong_word.fn}  ow ${x.perClass.omitted_word.tp}/${x.perClass.omitted_word.fp}/${x.perClass.omitted_word.fn}  clean ${x.cleanWarnings}  rep ${x.repeatFalsePositives}  edge ${x.edgeFalsePositives}  vowel ${x.vowelClipWarnings}  -FP ${x.removedFalsePositives.length}  +miss ${x.newlyMissed.length}`);
}
