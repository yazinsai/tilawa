/**
 * End-to-end check of the shipped runtime: help eval clips (WAV) through the session API on onnxruntime-node, scored
 * the way the scoreboard scores the free and declared-passage arms (`packages/core/src/bench/acceptance.ts`).
 */
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { CheckerEvent, DecodedPhoneme, EngineEvent, PhraseAsset, ReferenceIndex, WordRef } from "@tilawa/core";
import type { LoadedModel } from "../model.js";
import { createEngine } from "../session.js";
import type { Session } from "../session.js";
import { decodeWav } from "../wav.js";

const CLASSES = ["wrong_word", "omitted_word", "added_word", "skipped_passage"];

interface Label {
  id: string; speaker: string; kind: string; ayahs: string[]; wav: string; scored: boolean; split: string;
  events: { errorClass: string; state: string; location: WordRef }[];
  non_errors: { kind: string; location: WordRef }[];
}

export interface ArmScore { tp: number; fp: number; fn: number; cleanClips: number; cleanSeconds: number; cleanWarnings: number; repeatFalsePositives: number }

export interface HelpBenchClip {
  id: string;
  kind: string;
  audioS: number;
  phonemes: DecodedPhoneme[];
  nFrames: number;
  free: ClipScore;
  declared: ClipScore;
}

interface ClipScore { hits: number; fp: CheckerEvent[]; misses: Label["events"]; repeatFalsePositives: number }

export interface HelpBenchResult {
  clips: HelpBenchClip[];
  free: ArmScore & { precision: number; recall: number };
  declared: ArmScore & { precision: number; recall: number };
  seconds: { audio: number; compute: number };
}

const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const signature = (e: { errorClass: string; location: WordRef }) => `${e.errorClass}/${key(e.location)}`;

function score(events: EngineEvent[], expected: Label["events"], repeats: Set<string>): ClipScore {
  const latest = new Map<string, CheckerEvent>();
  for (const e of events) if (e.type === "checker") latest.set(e.id, e);
  const confirmed = [...latest.values()].filter((e) => e.state === "confirmed");
  const memo = expected.filter((e) => CLASSES.includes(e.errorClass));
  const wanted = memo.filter((e) => e.state === "confirmed");
  const pending = new Set(memo.filter((e) => e.state === "suspected").map(signature));
  const used = new Set<number>();
  const misses = wanted.filter((e) => {
    const i = confirmed.findIndex((p, j) => !used.has(j) && signature(p) === signature(e));
    if (i < 0) return true;
    used.add(i);
    return false;
  });
  const fp = confirmed.filter((p, j) => !used.has(j) && !pending.has(signature(p)));
  return { hits: used.size, fp, misses, repeatFalsePositives: fp.filter((p) => repeats.has(key(p.location))).length };
}

export async function runHelpBench(o: {
  helpDir: string;
  reference: ReferenceIndex;
  phrases: PhraseAsset;
  model: LoadedModel;
  /** Push size in ms of audio, as a microphone would deliver it. */
  pushMs?: number;
  evidenceMs?: number;
  limit?: number;
  onClip?: (c: HelpBenchClip, done: number, total: number) => void;
}): Promise<HelpBenchResult> {
  const jsonl = (f: string) => readFileSync(resolve(o.helpDir, f), "utf8").trim().split("\n").map((s) => JSON.parse(s));
  const labels = (jsonl("labels.jsonl") as Label[]).filter((l) => l.split === "eval" && l.scored).slice(0, o.limit);
  const submissions = resolve(o.helpDir, "raw/submissions.jsonl");
  const letterPairs = new Set<string>(existsSync(submissions)
    ? jsonl("raw/submissions.jsonl").filter((r) => (r.mistakes ?? []).some((m: any) => m.rule === "letter_confusion")).map((r) => r.id)
    : []);
  const engine = createEngine({ reference: o.reference, phrases: o.phrases, model: o.model, recordEvidence: true });
  const push = Math.round(((o.pushMs ?? 320) * 16000) / 1000);
  const clips: HelpBenchClip[] = [];
  let compute = 0;
  for (const l of labels) {
    const wav = decodeWav(readFileSync(resolve(o.helpDir, l.wav)));
    const [surah, fromAyah] = l.ayahs[0].split(":").map(Number);
    const toAyah = Number(l.ayahs.at(-1)!.split(":")[1]);
    const t0 = performance.now();
    const free = engine.startFreeRecitation({ evidenceMs: o.evidenceMs });
    const declared = engine.startMemorization({ surah, fromAyah, toAyah }, { evidenceMs: o.evidenceMs });
    const run = async (s: Session) => {
      const out: EngineEvent[] = [];
      for (let a = 0; a < wav.samples.length; a += push) out.push(...(await s.pushAudio(wav.samples.subarray(a, a + push), wav.sampleRate)));
      out.push(...(await s.end()));
      return out;
    };
    const freeEvents = await run(free);
    const declaredEvents = await run(declared);
    const phonemes = free.evidence().flatMap((e) => e.phonemes);
    const fbankFrames = 1 + Math.floor((wav.samples.length - 400) / 160);
    const nFrames = Math.floor((Math.floor((fbankFrames - 7) / 2) + 1) / 2);
    compute += (performance.now() - t0) / 1000;
    const expected = letterPairs.has(l.id) ? [] : l.events;
    const repeats = new Set(l.non_errors.filter((n) => n.kind === "repeat").map((n) => key(n.location)));
    const clip: HelpBenchClip = {
      id: l.id, kind: l.kind, audioS: wav.samples.length / wav.sampleRate, phonemes, nFrames,
      free: score(freeEvents, expected, repeats), declared: score(declaredEvents, expected, repeats),
    };
    clips.push(clip);
    o.onClip?.(clip, clips.length, labels.length);
  }
  const arm = (k: "free" | "declared") => {
    const tp = clips.reduce((n, c) => n + c[k].hits, 0);
    const fp = clips.reduce((n, c) => n + c[k].fp.length, 0);
    const fn = clips.reduce((n, c) => n + c[k].misses.length, 0);
    const clean = clips.filter((c) => c.kind === "none");
    return {
      tp, fp, fn, precision: tp / (tp + fp), recall: tp / (tp + fn),
      cleanClips: clean.length, cleanSeconds: clean.reduce((n, c) => n + c.audioS, 0),
      cleanWarnings: clean.reduce((n, c) => n + c[k].fp.length, 0),
      repeatFalsePositives: clips.reduce((n, c) => n + c[k].repeatFalsePositives, 0),
    };
  };
  return { clips, free: arm("free"), declared: arm("declared"), seconds: { audio: clips.reduce((n, c) => n + c.audioS, 0), compute } };
}
