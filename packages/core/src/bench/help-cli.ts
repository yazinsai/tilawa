/** Provisional help-recording evaluation. See tools/help_eval_labels.py for audit filtering. */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { createChecker } from "../checker.js";
import { createTracker } from "../tracker.js";
import { loadReferenceIndex } from "../reference.js";
import type { PhraseAsset, ReferenceAsset } from "../reference.js";
import type { CheckerEvent, WordRef, PositionEvent } from "../events.js";
import type { QudRecording } from "./index.js";
import { chunk } from "./index.js";

const { values } = parseArgs({ options: {
  dir: { type: "string", default: "../../.scratch/help" },
  report: { type: "string", default: "../../docs/benchmarks/help-memorization-v1" },
  prior: { type: "boolean", default: false },
} });
const dir = resolve(values.dir!);
const report = resolve(values.report!);
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const phrases = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8")) as PhraseAsset;
interface Label {
  id: string; speaker: string; kind: string; ayahs: string[]; wav: string;
  audit: "verified" | "provisional"; exclude: string | null; pronunciationPair: boolean;
  events: { errorClass: string; state: string; location: WordRef }[];
}
const labels: Label[] = readFileSync(resolve(dir, "eval-labels.jsonl"), "utf8").trim().split("\n").map(s => JSON.parse(s));
const classes = ["wrong_word", "omitted_word", "added_word", "skipped_passage"];
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const ayah = (r: WordRef) => `${r.surah}:${r.ayah}`;
const signature = (e: { errorClass: string; location: WordRef }) => `${e.errorClass}/${key(e.location)}`;
const results: any[] = [];
let missing = 0;
for (const lab of labels) {
  if (lab.exclude) continue;
  const path = resolve(dir, lab.wav.replace(/\.wav$/, ".evidence.json"));
  if (!existsSync(path)) { missing++; continue; }
  const evidence = JSON.parse(readFileSync(path, "utf8")) as QudRecording;
  if (evidence.inventory !== index.inventoryId) throw new Error(`Inventory mismatch: ${lab.id}`);
  const [ps, pa] = lab.ayahs[0].split(":").map(Number);
  const tracker = createTracker(index, { mode: "memorization", phrases, ...(values.prior && { prior: { surah: ps, ayah: pa, weight: 1 } }) });
  const checker = createChecker(index, { phrases });
  const checks: CheckerEvent[] = [];
  const positions: PositionEvent[] = [];
  for (const e of chunk(evidence, 1, false)) {
    const out = tracker.push(e);
    positions.push(...out.filter((x): x is PositionEvent => x.type === "position"));
    checks.push(...checker.push(e, out));
  }
  const end = tracker.end();
  positions.push(...end.filter((x): x is PositionEvent => x.type === "position"));
  checks.push(...checker.end(end));
  const latest = new Map<string, CheckerEvent>();
  for (const e of checks) latest.set(e.id, e);
  const confirmed = [...latest.values()].filter(e => e.state === "confirmed");
  const expected = lab.pronunciationPair ? [] : lab.events.filter(e => classes.includes(e.errorClass));
  const wanted = expected.filter(e => e.state === "confirmed");
  const used = new Set<number>();
  const misses = wanted.filter(e => {
    const i = confirmed.findIndex((p, j) => !used.has(j) && signature(p) === signature(e));
    if (i < 0) return true;
    used.add(i); return false;
  });
  // A labelled skip may legitimately remain suspected; report it separately from confirmed precision/recall.
  const pending = expected.filter(e => e.state === "suspected");
  const pendingSignatures = new Set(pending.map(signature));
  const fp = confirmed.filter((p, j) => !used.has(j) && !pendingSignatures.has(signature(p)));
  const locks = positions.filter(p => p.state === "locked" && p.observed[0]);
  const firstCorrect = locks.find(p => lab.ayahs.includes(ayah(p.observed[0].word)));
  const goodLock = !!firstCorrect;
  const pendingHits = pending.filter(e => [...latest.values()].some(p => signature(p) === signature(e) && (p.state === "suspected" || p.state === "confirmed")));
  const r = { id: lab.id, speaker: lab.speaker, kind: lab.kind, audit: lab.audit,
    audioS: evidence.nFrames * evidence.frameMs / 1000, expected: wanted, confirmed, fp, misses,
    tp: used.size, pending: pending.length, pendingHits: pendingHits.length,
    correctLock: goodLock, correctLockBy5S: !!firstCorrect && firstCorrect.timeMs <= 5000,
    firstCorrectLockMs: firstCorrect?.timeMs ?? null,
    missesWithoutCorrectLock: goodLock ? 0 : misses.length,
    checks, positions,
  };
  results.push(r);
  if (results.length % 25 === 0) console.log(`${results.length} clips evaluated`);
}
function summarize(rows: typeof results) {
  const tp = rows.reduce((n, r) => n + r.tp, 0);
  const fp = rows.reduce((n, r) => n + r.fp.length, 0);
  const fn = rows.reduce((n, r) => n + r.misses.length, 0);
  const clean = rows.filter(r => r.kind === "none");
  const cleanSeconds = clean.reduce((n, r) => n + r.audioS, 0);
  const warnings = clean.reduce((n, r) => n + r.confirmed.length, 0);
  return { clips: rows.length, speakers: new Set(rows.map(r => r.speaker)).size, tp, fp, fn,
    precision: tp + fp ? tp / (tp + fp) : null, recall: tp + fn ? tp / (tp + fn) : null,
    cleanClips: clean.length, cleanSeconds, cleanWarnings: warnings,
    cleanWarningsPer5Min: cleanSeconds ? warnings * 300 / cleanSeconds : null,
    correctLockBy5S: rows.filter(r => r.correctLockBy5S).length,
    noCorrectLock: rows.filter(r => !r.correctLock).length,
    pendingSkips: rows.reduce((n, r) => n + r.pending, 0), pendingSkipHits: rows.reduce((n, r) => n + r.pendingHits, 0),
    missesWithoutCorrectLock: rows.reduce((n, r) => n + r.missesWithoutCorrectLock, 0),
    perClass: Object.fromEntries(classes.map(c => {
      const positives = rows.reduce((n, r) => n + r.expected.filter((e: any) => e.errorClass === c).length, 0);
      const fn = rows.reduce((n, r) => n + r.misses.filter((e: any) => e.errorClass === c).length, 0);
      const fp = rows.reduce((n, r) => n + r.fp.filter((e: any) => e.errorClass === c).length, 0);
      return [c, { positives, tp: positives - fn, fn, fp }];
    })),
  };
}
const all = summarize(results), verified = summarize(results.filter(r => r.audit === "verified"));
const output = { version: 1, provisional: true, reference: index.version, policy: "default; 1 s chunks; no passage prior; memorization mode",
  labels: labels.length, excluded: labels.filter(l => l.exclude).length, missingEvidence: missing,
  all, verified, results };
writeFileSync(`${report}.json`, JSON.stringify(output, null, 2));
const pct = (n: number | null) => n === null ? "unmeasured" : `${(100 * n).toFixed(1)}%`;
const lines = ["# Help recordings: provisional memorization evaluation v1", "",
  "This is a diagnostic run, not an acceptance verdict. The listening audit is incomplete. Verified clips are a selected audit subset, not a representative test set.", "",
  "Teacher evidence is inferred offline with up to 30 s of audio context, then replayed in 1 s chunks. This does not measure a streaming student's latency or accuracy.", "",
  "Held-out speaker split is preserved. Audited incorrect/unsure labels and unreviewed known prompt swaps are excluded. Only approved swaps are applied to this evaluation snapshot. Letter-confusion clips are pronunciation negatives for the memorization checker, not wrong-word positives.", "",
  `Labels: ${labels.length}; excluded: ${output.excluded}; missing evidence: ${missing}. Reference: ${index.version}.`, "",
  "| Measure | All usable labels (provisional) | Audited and verified only |", "|---|---:|---:|",
  `| Clips / speakers | ${all.clips} / ${all.speakers} | ${verified.clips} / ${verified.speakers} |`,
  `| True / false positives / misses | ${all.tp} / ${all.fp} / ${all.fn} | ${verified.tp} / ${verified.fp} / ${verified.fn} |`,
  `| Confirmed-event precision | ${pct(all.precision)} | ${pct(verified.precision)} |`,
  `| Confirmed-event recall | ${pct(all.recall)} | ${pct(verified.recall)} |`,
  `| Clean clips / minutes | ${all.cleanClips} / ${(all.cleanSeconds / 60).toFixed(1)} | ${verified.cleanClips} / ${(verified.cleanSeconds / 60).toFixed(1)} |`,
  `| Clean false warnings | ${all.cleanWarnings} | ${verified.cleanWarnings} |`,
  `| Clean warnings per 5 min (short clips) | ${all.cleanWarningsPer5Min?.toFixed(2) ?? "unmeasured"} | ${verified.cleanWarningsPer5Min?.toFixed(2) ?? "unmeasured"} |`,
  `| Any expected ayah locked by 5 s (coarse proxy) | ${all.correctLockBy5S}/${all.clips} | ${verified.correctLockBy5S}/${verified.clips} |`,
  `| No correct lock anywhere | ${all.noCorrectLock} | ${verified.noCorrectLock} |`,
  `| Suspected skip labels detected | ${all.pendingSkipHits}/${all.pendingSkips} | ${verified.pendingSkipHits}/${verified.pendingSkips} |`, "",
  "## Confirmed errors by class (all usable labels)", "", "| Class | Positives | Hits | Misses | False positives |", "|---|---:|---:|---:|---:|",
  ...Object.entries(all.perClass).map(([c, r]) => `| ${c} | ${r.positives} | ${r.tp} | ${r.fn} | ${r.fp} |`), "",
  "## Limits", "",
  "Matching requires exact class and word location, one-to-one, using the final state per event id. Suspected skip labels are reported separately; a confirmed event at a labelled suspected-skip site is not a false positive. End-of-clip uncertain events are misses for confirmed labels.", "",
  `${all.missesWithoutCorrectLock}/${all.fn} confirmed-label misses occur in clips with no correct lock. Remaining misses cannot yet be attributed between listener and checker; that needs as-said phoneme labels or manual evidence review.`, "",
  "No word/ayah onset truth is available here, so the lock-by-5-s figure is only a clip-level proxy and false jumps are not scored. Long clean sessions are still required for the false-warning acceptance bar. Pronunciation feedback is not implemented and is not scored by this report.", ""];
writeFileSync(`${report}.md`, lines.join("\n"));
console.log(JSON.stringify({ all, verified, missingEvidence: missing, report }, null, 2));
