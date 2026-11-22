import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { createChecker } from "../checker.js";
import type { CheckerEvent, PositionEvent, WordRef } from "../events.js";
import { loadReferenceIndex } from "../reference.js";
import type { PhraseAsset, ReferenceAsset } from "../reference.js";
import { createTracker } from "../tracker.js";
import { benchRecording, chunk, wordsToUnique } from "../bench/index.js";
import type { QudRecording } from "../bench/index.js";

/**
 * End to end: audio decoded by the teacher (`modal/decode_audio.py`) streamed through the tracker in 1 s chunks, against
 * the timeline of what each session plays (`tools/build_e2e.py`). Exits non-zero when any session fails its checks.
 */

const { values } = parseArgs({ options: { dir: { type: "string", default: "../../.scratch/e2e" }, verbose: { type: "boolean", default: false } } });
const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const phrases: PhraseAsset = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);

interface Truth {
  id: string;
  about: string;
  mode: "free" | "memorization";
  truth: { label: string; start_s: number; end_s: number }[];
}
interface Evidence {
  inventory: string;
  frameMs: number;
  nFrames: number;
  phonemes: QudRecording["phonemes"];
  posteriors: QudRecording["posteriors"];
}

const SPECIAL: Record<string, string> = { audhubillah: "Isti'adha:1", bismillah: "Basmala:1" };
/**
 * A start passes when located within this long of its first word, or of the point its audio allows a lock. That point
 * is interpolated within an ayah's audio by phoneme position, which is off by up to about 2 s, hence 3 s.
 */
const WITHIN_S = 5;
const AFTER_UNIQUE_S = 3;
const key = (r: WordRef | null | undefined) => (r ? `${r.surah}:${r.ayah}:${r.word}` : "–");

let failed = 0;
let totalWarnings = 0;
let totalMinutes = 0;
const dir = values.dir!;
for (const f of readdirSync(dir).filter((x) => x.endsWith(".truth.json")).sort()) {
  const truth: Truth = JSON.parse(readFileSync(join(dir, f), "utf8"));
  const evPath = join(dir, f.replace(".truth.json", ".evidence.json"));
  if (!existsSync(evPath)) {
    console.log(`${truth.id}: no evidence, run modal/decode_audio.py`);
    failed++;
    continue;
  }
  const ev: Evidence = JSON.parse(readFileSync(evPath, "utf8"));
  const ayahs = truth.truth.filter((t) => /^\d+:\d+$/.test(t.label));
  // Each ayah becomes one truth "word" spanning its audio: the scorer compares ayahs.
  const rec: QudRecording = {
    id: truth.id,
    meta: { style: "", content: "", noisy: false, passages: "", duration_s: 0, repeat_events: 0 },
    inventory: ev.inventory,
    frameMs: ev.frameMs,
    nFrames: ev.nFrames,
    phonemes: ev.phonemes,
    posteriors: ev.posteriors,
    truth: {
      words: truth.truth
        .filter((t) => /^\d+:\d+$/.test(t.label) || SPECIAL[t.label])
        .map((t) => ({ word: SPECIAL[t.label] ?? `${t.label}:1`, start_s: t.start_s, end_s: t.end_s })),
      non_quran: truth.truth.filter((t) => t.label === "silence" || t.label === "noise").map((t) => ({ start_s: t.start_s, end_s: t.end_s })),
    },
  };
  const r = benchRecording(index, rec, { posteriors: false, phrases, mode: truth.mode });

  const tracker = createTracker(index, { phrases, mode: truth.mode });
  const checker = createChecker(index, { phrases });
  const events: PositionEvent[] = [];
  const checks: CheckerEvent[] = [];
  for (const e of chunk(rec, 1, false)) {
    const out = tracker.push(e);
    for (const x of out) if (x.type === "position") events.push(x);
    checks.push(...checker.push(e, out));
  }
  const last = tracker.end();
  for (const x of last) if (x.type === "position") events.push(x);
  checks.push(...checker.end(last));
  // Professional recitations: every confirmed checker error is a false warning.
  const minutes = (ev.nFrames * ev.frameMs) / 60_000;
  const warnings = checks.filter((c) => c.state === "confirmed");
  totalWarnings += warnings.length;
  totalMinutes += minutes;

  /**
   * When a start's audio first allows a lock: its text unique in the Quran, and the tracker's minimum evidence of 2 words
   * and 10 phonemes heard. Truth here is per ayah, so the time is interpolated by phoneme position within the ayah.
   */
  const lockableAt = (word: string): number => {
    const [su, ay] = word.split(":").map(Number);
    const i = index.words.findIndex((w) => w.surah === su && w.ayah === ay && w.word === 1);
    let n = Math.max(2, wordsToUnique(index, index.words[i]));
    while (index.words[i + n - 1].end - index.words[i].start < 10) n++;
    const last = index.words[i + n - 1];
    const ayahWords = index.words.filter((w) => w.surah === last.surah && w.ayah === last.ayah);
    const span = truth.truth.find((t) => t.label === `${last.surah}:${last.ayah}`);
    if (!span) return Infinity;
    const frac = (last.end - ayahWords[0].start) / (ayahWords.at(-1)!.end - ayahWords[0].start);
    return span.start_s + frac * (span.end_s - span.start_s);
  };
  const problems: string[] = [];
  for (const s of r.starts) {
    const lockedAt = s.latencyS === null ? null : s.atS + s.latencyS;
    const ok = lockedAt !== null && (s.latencyS! <= WITHIN_S || lockedAt - lockableAt(s.word) <= AFTER_UNIQUE_S);
    if (!ok) problems.push(`start ${s.word.replace(/:1$/, "")} at ${s.atS.toFixed(1)} s located after ${s.latencyS?.toFixed(1) ?? "never"} s`);
    if (s.falseLock) problems.push(`start ${s.word.replace(/:1$/, "")}: a wrong lock was shown first`);
  }
  if (r.falseJumps) problems.push(`${r.falseJumps} false jump(s)`);
  const final = events.at(-1);
  const lastAyah = ayahs.at(-1)!.label;
  if (final?.state !== "locked" || `${final.observed[0].word.surah}:${final.observed[0].word.ayah}` !== lastAyah) {
    problems.push(`ends ${final?.state} at ${key(final?.observed[0]?.word)}, expected ${lastAyah}`);
  }
  if (truth.mode === "memorization") {
    // Once the jump commits, the expected bookmark must stay on an ayah of the first passage that was actually recited.
    const firstSurah = ayahs[0].label.split(":")[0];
    const recited = new Set(ayahs.filter((a) => a.label.split(":")[0] === firstSurah).map((a) => a.label));
    const committed = events.findIndex((e) => e.state === "locked" && String(e.observed[0].word.surah) !== firstSurah);
    const moved = events.slice(Math.max(0, committed)).filter((e) => e.expected && !recited.has(`${e.expected.surah}:${e.expected.ayah}`));
    if (moved.length) problems.push(`expected bookmark moved to ${key(moved[0].expected)} in memorization mode`);
  }

  const ok = problems.length === 0;
  if (!ok) failed++;
  console.log(`\n${ok ? "PASS" : "FAIL"} ${truth.id} (${truth.mode}): ${truth.about}`);
  for (const s of r.starts) {
    const after = s.latencyS === null ? null : s.atS + s.latencyS - lockableAt(s.word);
    console.log(`  start ${s.word.replace(/:1$/, "").padEnd(7)} at ${s.atS.toFixed(1).padStart(6)} s: located after ${s.latencyS?.toFixed(1) ?? "never"} s, ${after?.toFixed(1) ?? "–"} s after the audio allowed it`);
  }
  console.log(`  wrong-locked time ${((100 * r.wrongLockedSamples) / Math.max(1, r.lockedSamples)).toFixed(1)}%, jumps ${r.jumps} (false ${r.falseJumps}), final ${final?.state} ${key(final?.observed[0]?.word)} expected ${key(final?.expected)}, rtf ${r.rtf.toFixed(4)}`);
  const suspected = checks.filter((c) => c.state === "suspected").length;
  console.log(`  checker: ${warnings.length} false warning(s) in ${minutes.toFixed(1)} min (${((5 * warnings.length) / minutes).toFixed(2)} per 5 min), ${suspected} suspected`);
  for (const c of warnings) console.log(`    ✗ ${c.errorClass} at ${key(c.location)} ${(c.timeMs / 1000).toFixed(1)} s: expected ${c.expected}, heard ${c.observed ?? "–"}, conf ${c.confidence.toFixed(2)}`);
  for (const p of problems) console.log(`  ✗ ${p}`);
  if (values.verbose) {
    const timeline = [
      ...truth.truth.map((t) => [t.start_s, `      ▶ ${t.label}`] as const),
      ...events.map((e) => [e.timeMs / 1000, `${e.state.padEnd(9)} ${key(e.observed[0]?.word).padEnd(10)} expected ${key(e.expected)}`] as const),
    ].sort((a, b) => a[0] - b[0]);
    for (const [t, line] of timeline) console.log(`    ${t.toFixed(2).padStart(7)}  ${line}`);
  }
}
console.log(`\nchecker: ${totalWarnings} false warning(s) in ${totalMinutes.toFixed(1)} min, ${((5 * totalWarnings) / Math.max(totalMinutes, 1e-9)).toFixed(2)} per 5 min`);
console.log(`${failed ? `${failed} session(s) failed` : "all sessions passed"}`);
process.exit(failed ? 1 : 0);
