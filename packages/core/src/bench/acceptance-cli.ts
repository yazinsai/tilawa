/** Acceptance report on the held-out help clips (ticket 07 phase 1). Run `tools/e1_report.py --help-clips` first. */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { execSync } from "node:child_process";
import { loadReferenceIndex } from "../reference.js";
import type { PhraseAsset, ReferenceAsset } from "../reference.js";
import type { WordRef } from "../events.js";
import { MEMORIZATION_CLASSES, attribute, scoreClip } from "./acceptance.js";
import type { AcceptanceClip, Arm, ClipScore, MissCause } from "./acceptance.js";
import type { QudRecording } from "./index.js";

const VERSION = process.env.ACCEPTANCE_VERSION ?? "v2";
const { values } = parseArgs({ options: {
  dir: { type: "string", default: process.env.HELP_DIR ?? "../../.scratch/help" },
  report: { type: "string", default: `../../docs/benchmarks/acceptance-${VERSION}` },
} });
const dir = resolve(values.dir!);
const report = resolve(values.report!);
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const phrases = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8")) as PhraseAsset;
const jsonl = (f: string) => readFileSync(resolve(dir, f), "utf8").trim().split("\n").map((s) => JSON.parse(s));

interface Label {
  id: string; speaker: string; kind: string; ayahs: string[]; wav: string; scored: boolean; split: string;
  events: { errorClass: string; state: string; location: WordRef }[];
  non_errors: { kind: string; location: WordRef }[];
}
const labels: Label[] = jsonl("labels.jsonl");
const verdicts = new Map<string, string>(jsonl("answers.jsonl").map((a) => [a.id, a.verdict]));
// A letter-confusion substitution is a pronunciation error, not a wrong word.
const letterPairs = new Set<string>(jsonl("raw/submissions.jsonl").filter((r) => (r.mistakes ?? []).some((m: any) => m.rule === "letter_confusion")).map((r) => r.id));
const evalLabels = labels.filter((l) => l.split === "eval");
const usable = evalLabels.filter((l) => l.scored);

interface Row {
  id: string; speaker: string; kind: string; verified: boolean; audioS: number; shortVowels: number;
  /** Classes of the confirmed memorization labels. */
  positives: string[];
  free: ClipScore; declared: ClipScore; causes: { errorClass: string; cause: MissCause; detail: string }[];
}
const rows: Row[] = [];
for (const l of usable) {
  const evidence = JSON.parse(readFileSync(resolve(dir, l.wav.replace(/\.wav$/, ".evidence.json")), "utf8")) as QudRecording;
  if (evidence.inventory !== index.inventoryId) throw new Error(`Inventory mismatch: ${l.id}`);
  const clip: AcceptanceClip = {
    id: l.id, ayahs: l.ayahs, evidence,
    events: letterPairs.has(l.id) ? [] : l.events,
    nonErrors: l.non_errors.filter((n) => n.kind === "repeat").map((n) => n.location),
  };
  const free = scoreClip(index, clip, { arm: "free", phrases });
  const declared = scoreClip(index, clip, { arm: "declared", phrases });
  rows.push({
    id: l.id, speaker: l.speaker, kind: l.kind, verified: verdicts.get(l.id) === "1",
    audioS: (evidence.nFrames * evidence.frameMs) / 1000,
    shortVowels: l.events.filter((e) => e.errorClass === "short_vowel").length,
    positives: clip.events.filter((e) => e.state === "confirmed" && MEMORIZATION_CLASSES.includes(e.errorClass)).map((e) => e.errorClass),
    free, declared,
    causes: attribute(index, clip, free, declared).map((a) => ({ errorClass: a.miss.errorClass, cause: a.cause, detail: a.cause === "checker" ? checkerDetail(declared, a.miss) : "" })),
  });
  if (rows.length % 50 === 0) console.log(`${rows.length}/${usable.length} clips`);
}

/** What the declared-passage checker did at a missed site: it shows whether the confirmation rule or the alignment lost it. */
function checkerDetail(s: ClipScore, miss: AcceptanceClip["events"][number]): string {
  const at = s.final.filter((e) => e.location.surah === miss.location.surah && e.location.ayah === miss.location.ayah && e.location.word === miss.location.word);
  const same = at.find((e) => e.errorClass === miss.errorClass);
  if (same) return `${same.state} at clip end`;
  if (at.length) return `raised as ${at[0].errorClass}`;
  const near = s.final.find((e) => e.location.surah === miss.location.surah && e.location.ayah === miss.location.ayah && Math.abs(e.location.word - miss.location.word) <= 1);
  return near ? `raised on a neighbouring word (${near.errorClass}, ${near.state})` : "not raised";
}
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const ratio = (n: number, d: number) => (d ? n / d : null);
function summarize(rs: Row[], arm: Arm) {
  const s = (r: Row) => r[arm];
  const tp = sum(rs.map((r) => s(r).hits));
  const fp = sum(rs.map((r) => s(r).falsePositives.length));
  const fn = sum(rs.map((r) => s(r).misses.length));
  const clean = rs.filter((r) => r.kind === "none");
  const cleanS = sum(clean.map((r) => r.audioS));
  const cleanWarnings = sum(clean.map((r) => s(r).confirmed.length));
  const repeats = rs.filter((r) => r.kind === "repeat");
  return {
    clips: rs.length, speakers: new Set(rs.map((r) => r.speaker)).size, tp, fp, fn,
    precision: ratio(tp, tp + fp), recall: ratio(tp, tp + fn),
    perClass: Object.fromEntries(MEMORIZATION_CLASSES.map((c) => {
      const misses = sum(rs.map((r) => s(r).misses.filter((e) => e.errorClass === c).length));
      const all = sum(rs.map((r) => r.positives.filter((e) => e === c).length));
      const fpc = sum(rs.map((r) => s(r).falsePositives.filter((e) => e.errorClass === c).length));
      return [c, { positives: all, tp: all - misses, fn: misses, fp: fpc, precision: ratio(all - misses, all - misses + fpc), recall: ratio(all - misses, all) }];
    })),
    cleanClips: clean.length, cleanMinutes: cleanS / 60, cleanWarnings,
    cleanWarningsPerClip: ratio(cleanWarnings, clean.length), cleanWarningsPer5Min: cleanS ? (cleanWarnings * 300) / cleanS : null,
    cleanClipsWithWarning: clean.filter((r) => s(r).confirmed.length > 0).length,
    repeatClips: repeats.length, repeatFalsePositives: sum(rs.map((r) => s(r).repeatFalsePositives)),
    pendingSkips: sum(rs.map((r) => s(r).pending)), pendingSkipHits: sum(rs.map((r) => s(r).pendingHits)),
  };
}
function gate(rs: Row[]) {
  const g = rs.map((r) => r.free.gate);
  return {
    starts: g.length, within5s: g.filter((x) => x.correctWithin5s).length, neverCorrect: g.filter((x) => x.firstCorrectLockMs === null).length,
    wrongLockFirst: g.filter((x) => x.wrongLockFirst).length, jumps: sum(g.map((x) => x.jumps)), falseJumps: sum(g.map((x) => x.falseJumps)),
    clipsWithFalseJump: g.filter((x) => x.falseJumps > 0).length,
    clipsWithOutsideLock: g.filter((x) => x.outsideLock).length,
  };
}
function details(rs: Row[]) {
  const out: Record<string, number> = {};
  for (const r of rs) for (const c of r.causes) if (c.cause === "checker") out[`${c.errorClass}: ${c.detail}`] = (out[`${c.errorClass}: ${c.detail}`] ?? 0) + 1;
  return out;
}
function causes(rs: Row[]) {
  const out: Record<string, Record<MissCause, number>> = {};
  for (const r of rs) for (const c of r.causes) (out[c.errorClass] ??= { evidence: 0, tracker: 0, checker: 0 })[c.cause]++;
  return out;
}
const verified = rows.filter((r) => r.verified);
const e1Path = `${report}-e1.json`;
const e1 = existsSync(e1Path) ? JSON.parse(readFileSync(e1Path, "utf8")) : null;
const output = {
  version: VERSION,
  commit: execSync("git rev-parse --short HEAD").toString().trim(),
  reference: index.version,
  policy: "default checker policy; 1 s chunks; memorization mode; no passage prior; declared arm passes the prompted passage to the checker",
  labels: { eval: evalLabels.length, unscored: evalLabels.length - usable.length, letterPairClips: rows.filter((r) => letterPairs.has(r.id)).length },
  free: { all: summarize(rows, "free"), verified: summarize(verified, "free") },
  declared: { all: summarize(rows, "declared"), verified: summarize(verified, "declared") },
  shortVowel: { positives: sum(rows.map((r) => r.shortVowels)), sites: new Set(usable.flatMap((l) => l.events.filter((e) => e.errorClass === "short_vowel").map((e) => `${l.speaker}/${e.location.surah}:${e.location.ayah}`))).size, checkerHits: 0, e1: e1?.classes ?? null },
  gate: { all: gate(rows), clean: gate(rows.filter((r) => r.kind === "none")) },
  e0: { all: causes(rows), verified: causes(verified), checkerDetail: details(rows) },
  clips: rows.map((r) => ({ id: r.id, kind: r.kind, verified: r.verified, audioS: r.audioS, gate: r.free.gate, causes: r.causes,
    free: { confirmed: r.free.confirmed.map(brief), misses: r.free.misses, falsePositives: r.free.falsePositives.map(brief) },
    declared: { confirmed: r.declared.confirmed.map(brief), misses: r.declared.misses, falsePositives: r.declared.falsePositives.map(brief) } })),
};
function brief(e: ClipScore["confirmed"][number]) {
  return { errorClass: e.errorClass, location: e.location, timeMs: e.timeMs, observed: e.observed, confidence: +e.confidence.toFixed(3) };
}
writeFileSync(`${report}.json`, JSON.stringify(output, null, 1) + "\n");

const pct = (n: number | null) => (n === null ? "—" : `${(100 * n).toFixed(1)}%`);
const frac = (n: number, d: number) => `${n}/${d}${d ? ` (${((100 * n) / d).toFixed(1)}%)` : ""}`;
const fa = output.free.all, da = output.declared.all, fv = output.free.verified, dv = output.declared.verified;
const ga = output.gate.all, gc = output.gate.clean;
const lines = [
  `# Acceptance eval ${VERSION}: help clips (ticket 07)`, "",
  `Report ${VERSION}, commit ${output.commit}, reference ${index.version}. Held-out speakers only (eval split): ${output.labels.eval} clips, ${output.labels.unscored} unscored by the import or the audit (ticket 12), ${rows.length} scored from ${fa.speakers} speakers. Labels are the final post-audit \`labels.jsonl\` with version swaps applied. ${output.labels.letterPairClips} letter-confusion substitutions are pronunciation errors, so they count as negatives here. Repeats are non-errors: a confirmed event on one is a false positive.`, "",
  "Teacher evidence is the offline Muaalem decode (30 s windows), replayed in 1 s chunks. It is not a streaming student.", "",
  "Two arms, both core engine code. **Free recitation** is the default: the checker follows the tracker lock (Decision 3). **Declared passage**: the session passes the prompted passage (`CheckerOptions.passage`); the checker aligns against it and uses the tracker lock only to follow a deliberate jump (a lock of 4+ words more than an ayah off the passage whose words don't also fit the passage). At session end a declared session confirms a suspected wrong or added word once a later word was recited right; free recitation still ends it `uncertain`.", "",
  "## Memorization", "",
  "| Measure | Free (all) | Declared (all) | Free (audit-verified) | Declared (audit-verified) | Bar |", "|---|---:|---:|---:|---:|---|",
  `| Clips / speakers | ${fa.clips} / ${fa.speakers} | ${da.clips} / ${da.speakers} | ${fv.clips} / ${fv.speakers} | ${dv.clips} / ${dv.speakers} | |`,
  `| Hits / false positives / misses | ${fa.tp} / ${fa.fp} / ${fa.fn} | ${da.tp} / ${da.fp} / ${da.fn} | ${fv.tp} / ${fv.fp} / ${fv.fn} | ${dv.tp} / ${dv.fp} / ${dv.fn} | |`,
  `| Precision | ${pct(fa.precision)} | ${pct(da.precision)} | ${pct(fv.precision)} | ${pct(dv.precision)} | ≥ 90% |`,
  `| Recall | ${pct(fa.recall)} | ${pct(da.recall)} | ${pct(fv.recall)} | ${pct(dv.recall)} | ≥ 80% |`,
  `| Clean clips / minutes | ${fa.cleanClips} / ${fa.cleanMinutes.toFixed(1)} | ${da.cleanClips} / ${da.cleanMinutes.toFixed(1)} | ${fv.cleanClips} / ${fv.cleanMinutes.toFixed(1)} | ${dv.cleanClips} / ${dv.cleanMinutes.toFixed(1)} | |`,
  `| Clean false warnings (clips with one) | ${fa.cleanWarnings} (${fa.cleanClipsWithWarning}) | ${da.cleanWarnings} (${da.cleanClipsWithWarning}) | ${fv.cleanWarnings} (${fv.cleanClipsWithWarning}) | ${dv.cleanWarnings} (${dv.cleanClipsWithWarning}) | |`,
  `| False warnings per clean clip | ${fa.cleanWarningsPerClip?.toFixed(3)} | ${da.cleanWarningsPerClip?.toFixed(3)} | ${fv.cleanWarningsPerClip?.toFixed(3) ?? "—"} | ${dv.cleanWarningsPerClip?.toFixed(3) ?? "—"} | |`,
  `| False warnings per 5 min (short clips) | ${fa.cleanWarningsPer5Min?.toFixed(2)} | ${da.cleanWarningsPer5Min?.toFixed(2)} | ${fv.cleanWarningsPer5Min?.toFixed(2) ?? "—"} | ${dv.cleanWarningsPer5Min?.toFixed(2) ?? "—"} | ≤ 1, phase 2 only |`,
  `| False positives on repeats (${fa.repeatClips} clips) | ${fa.repeatFalsePositives} | ${da.repeatFalsePositives} | ${fv.repeatFalsePositives} | ${dv.repeatFalsePositives} | |`,
  `| Suspected skip labels raised | ${fa.pendingSkipHits}/${fa.pendingSkips} | ${da.pendingSkipHits}/${da.pendingSkips} | ${fv.pendingSkipHits}/${fv.pendingSkips} | ${dv.pendingSkipHits}/${dv.pendingSkips} | |`, "",
  "### Per class (all scored clips)", "",
  "| Class | Positives | Free hits | Free FP | Free precision | Free recall | Declared hits | Declared FP | Declared precision | Declared recall |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ...MEMORIZATION_CLASSES.map((c) => {
    const f = fa.perClass[c], d = da.perClass[c];
    return `| ${c} | ${f.positives} | ${f.tp} | ${f.fp} | ${pct(f.precision)} | ${pct(f.recall)} | ${d.tp} | ${d.fp} | ${pct(d.precision)} | ${pct(d.recall)} |`;
  }), "",
  "`skipped_passage` labels are all `suspected` (a skipped ayah may be a deliberate jump), so they are not confirmed positives; they appear in the suspected-skip row above.", "",
  "## Short vowel", "",
  `${output.shortVowel.positives} positives at ${output.shortVowel.sites} speaker × ayah sites. The checker has no pronunciation classes yet, so checker recall is 0 and precision is unmeasured: \`short_vowel\` is \`unsupported\` as a feedback class. The teacher-level E1 numbers on the same clips:`, "",
  ...(e1 ? e1Table(e1) : ["E1 on help clips not run (`tools/e1_report.py --help-clips`)."]), "",
  "## Tracker gate on this audio", "",
  "Each clip is one start. No word onsets are labelled, so a start is timed from the first decoded phoneme. A correct lock is any lock on a prompted ayah; a false jump is a non-local lock change to an ayah more than 2 ayahs from the prompted passage.", "",
  "| Measure | All scored clips | Clean clips | Bar |", "|---|---:|---:|---|",
  `| Correct ayah within 5 s of onset | ${frac(ga.within5s, ga.starts)} | ${frac(gc.within5s, gc.starts)} | ≥ 95% |`,
  `| Never correctly locked | ${ga.neverCorrect} | ${gc.neverCorrect} | |`,
  `| Wrong lock before the correct one | ${ga.wrongLockFirst} | ${gc.wrongLockFirst} | |`,
  `| False jumps / jumps | ${ga.falseJumps}/${ga.jumps} | ${gc.falseJumps}/${gc.jumps} | |`,
  `| Clips with any lock outside the passage | ${frac(ga.clipsWithOutsideLock, ga.starts)} | ${frac(gc.clipsWithOutsideLock, gc.starts)} | |`,
  `| Clips with a false jump | ${frac(ga.clipsWithFalseJump, ga.starts)} | ${frac(gc.clipsWithFalseJump, gc.starts)} | ≤ 2% |`, "",
  "Error clips are user-like but adversarial for the tracker: a substitution that forms another ayah is a correct lock on what was said, scored here as wrong. The clean-clip column is the fair gate number. Clips are one short passage, so the tracker almost never changes lock after its first one; a wrong first lock shows up in the outside-lock rows, not as a jump. Ambiguous tlog clips are not included: the bucket review (ticket 06) found the sampled ones are wrong-passage or no-recitation, with no passage truth to score against.", "",
  "## E0 attribution of misses (free arm)", "",
  "`tracker`: the declared-passage arm catches it, so the free arm lost it to the lock. Otherwise the greedy decode is aligned to the prompted passage: `evidence` when at least half the planted word's consonants decode as written (the teacher shows no error), `checker` when the decode shows the error and the checker still didn't confirm it.", "",
  "| Class | Evidence | Tracker | Checker | Verified: evidence / tracker / checker |", "|---|---:|---:|---:|---|",
  ...Object.entries(output.e0.all).map(([c, x]) => {
    const v = output.e0.verified[c] ?? { evidence: 0, tracker: 0, checker: 0 };
    return `| ${c} | ${x.evidence} | ${x.tracker} | ${x.checker} | ${v.evidence} / ${v.tracker} / ${v.checker} |`;
  }), "",
  "Checker misses, by what the declared-passage checker did at the site:", "",
  "| Class: outcome | Misses |", "|---|---:|",
  ...Object.entries(output.e0.checkerDetail).sort((a, b) => b[1] - a[1]).map(([k, n]) => `| ${k} | ${n} |`), "",
  "## Limits", "",
  "- Clips are single ayahs or short ranges (mean " + (sum(rows.map((r) => r.audioS)) / rows.length).toFixed(1) + " s). Per-5-minute false warnings from short clips are not the acceptance bar; phase 2's long sessions are.",
  "- Matching needs exact class and word, one-to-one, on each event's final state. An error still pending at clip end is a miss: short clips end before the 2.5 s confirmation delay.",
  "- The evidence check is a heuristic on the greedy decode, not an as-said label. `checker` includes errors the decode shows only partly.",
  "- Deliberate jumps are covered by a synthetic test only: the help clips have none, so this eval shows jump detection doesn't misfire on substitutions, not that it catches real jumps.",
  "- End-of-session finalisation is off in free recitation. Measured with it on, the free arm went from 32 / 26 / 32 to 34 / 31 / 30 (hits / FP / misses) and clean false warnings from 8 to 11, and a clean tlog fixture gained two confirmed errors. In the declared arm it is worth it: off, 43 / 30 / 21 and 10 clean warnings.",
  "- Pronunciation classes other than short vowel have fewer than 15 eval sites (ص/س 4, madd 1) and are `unsupported` here.", "",
];
function e1Table(r: any): string[] {
  const out = ["| class | positives | sites | in greedy | canonicalized | posterior prefers error | negatives | greedy FP | posterior FP | verdict |", "|---|---:|---:|---|---|---|---:|---|---|---|"];
  for (const c of r.classes) {
    if (!c.pos && !c.neg) continue;
    out.push(`| ${c.name} | ${c.pos} | ${c.sites ?? "—"} | ${frac(c.greedy_hit, c.pos)} | ${frac(c.greedy_canon, c.pos)} | ${frac(c.llr_hit, c.pos)} | ${c.neg} | ${frac(c.greedy_fp, c.neg)} | ${frac(c.llr_fp, c.neg)} | ${c.verdict} |`);
  }
  out.push("", `Scored ${r.scored} clips; skipped ${r.skipped} (decode too far from the canonical, or unmappable vowel site). Real acted errors, next to E1 (\`e1-teacher-bias-v1\`) and the E1b splice upper bound (\`e1b-spliced-v1\`).`);
  return out;
}
writeFileSync(`${report}.md`, lines.join("\n"));
console.log(JSON.stringify({ free: fa, declared: da, gate: output.gate, e0: output.e0 }, null, 1));
