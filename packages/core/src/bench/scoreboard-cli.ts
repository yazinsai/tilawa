/**
 * Appends one scoreboard row (commit × listener) to docs/benchmarks/scoreboard.json and regenerates scoreboard.md.
 * CPU only: replays evidence already on disk through the tracker and checker, and reads the listener reports.
 */
import { execSync } from "node:child_process";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "../reference.js";
import type { PhraseAsset, ReferenceAsset } from "../reference.js";
import type { WordRef } from "../events.js";
import { MEMORIZATION_CLASSES, attribute, scoreClip, visibleInEvidence } from "./acceptance.js";
import type { AcceptanceClip, ClipScore } from "./acceptance.js";
import { benchRecording, summarise } from "./index.js";
import type { QudRecording } from "./index.js";
import { perfGate } from "./perf.js";
import { goldPerMetrics, oracleClip, renderScoreboard, scored, upsertRow } from "./scoreboard.js";
import type { Metric, OracleLabel, ScoreRow } from "./scoreboard.js";

const { values } = parseArgs({ options: {
  listener: { type: "string", default: "teacher" },
  help: { type: "string", default: process.env.HELP_DIR ?? "../../.scratch/help" },
  /** Directory holding `<clip>.evidence.json` from this listener; defaults to the teacher's, next to the wavs. */
  evidence: { type: "string" },
  qud: { type: "string", default: process.env.QUD_DIR ?? "../../.scratch/bench/qud" },
  out: { type: "string", default: "../../docs/benchmarks/scoreboard" },
  sha: { type: "string" },
  /** One line: what changed and why. */
  note: { type: "string" },
  /** Comma-separated engine modules changed: listener,tracker,checker. */
  modules: { type: "string" },
} });
const moduleNames = ["listener", "tracker", "checker"] as const;
const modules = values.modules === undefined ? undefined : [...new Set(values.modules.split(",").map((m) => m.trim()))];
if (modules?.some((m) => !moduleNames.includes(m as typeof moduleNames[number]))) {
  throw new Error("--modules must contain listener, tracker, or checker (comma-separated)");
}
const bench = resolve(values.out!, "..");
const read = (f: string) => JSON.parse(readFileSync(join(bench, f), "utf8"));
const asset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset;
const index = loadReferenceIndex(asset);
const phrases = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8")) as PhraseAsset;
const teacher = values.listener === "teacher";
const helpDir = resolve(values.help!);
const evidenceDir = values.evidence ? resolve(values.evidence) : teacher ? helpDir : null;
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const ratio = (n: number, d: number) => (d ? n / d : null);
const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return s.length ? (s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : null; };
const frac = (n: number, d: number) => `${d ? ((100 * n) / d).toFixed(1) : "—"}% (${n}/${d})`;
const metrics: Metric[] = [];
/** Backing report for the metrics added next. */
let source = "acceptance-v1.md";
const add = (m: Parameters<typeof scored>[0]) => metrics.push(scored({ source, ...m }));
const MEM = { precision: { op: ">=", target: 0.9 }, recall: { op: ">=", target: 0.8 }, per5: { op: "<=", target: 1 } } as const;

// ---- Help clips: end to end, tracker on user-like audio, checker with oracle upstream.
const jsonl = (f: string) => readFileSync(resolve(helpDir, f), "utf8").trim().split("\n").map((s) => JSON.parse(s));
type Label = OracleLabel & { speaker: string; kind: string; wav: string; scored: boolean; split: string };
const labels: Label[] = jsonl("labels.jsonl");
const letterPairs = new Set<string>(jsonl("raw/submissions.jsonl").filter((r) => (r.mistakes ?? []).some((m: any) => m.rule === "letter_confusion")).map((r) => r.id));
const usable = labels.filter((l) => l.split === "eval" && l.scored);
const evidencePath = (l: Label) => evidenceDir && resolve(evidenceDir, (values.evidence ? basename(l.wav) : l.wav).replace(/\.wav$/, ".evidence.json"));
const haveEvidence = !!evidenceDir && usable.every((l) => existsSync(evidencePath(l)!));

interface Row { l: Label; clip: AcceptanceClip; audioS: number; positives: string[]; free: ClipScore; declared: ClipScore; oracle: ClipScore | null }
const rows: Row[] = [];
if (haveEvidence) {
  for (const l of usable) {
    const evidence = JSON.parse(readFileSync(evidencePath(l)!, "utf8")) as QudRecording;
    if (evidence.inventory !== index.inventoryId) throw new Error(`Inventory mismatch: ${l.id}`);
    const events = letterPairs.has(l.id) ? [] : l.events;
    const clip: AcceptanceClip = { id: l.id, ayahs: l.ayahs, evidence, events, nonErrors: l.non_errors.filter((n) => n.kind === "repeat").map((n) => n.location) };
    const oc = oracleClip(index, asset, { ...l, events });
    rows.push({
      l, clip, audioS: (evidence.nFrames * evidence.frameMs) / 1000,
      positives: events.filter((e) => e.state === "confirmed" && MEMORIZATION_CLASSES.includes(e.errorClass)).map((e) => e.errorClass),
      free: scoreClip(index, clip, { arm: "free", phrases }),
      declared: scoreClip(index, clip, { arm: "declared", phrases }),
      oracle: oc && scoreClip(index, oc, { arm: "declared", phrases }),
    });
    if (rows.length % 100 === 0) console.log(`help ${rows.length}/${usable.length}`);
  }
}
type ArmKey = "free" | "declared" | "oracle";
function memo(rs: Row[], arm: ArmKey) {
  const s = rs.filter((r) => r[arm]).map((r) => ({ r, x: r[arm]! }));
  const tp = sum(s.map(({ x }) => x.hits)), fp = sum(s.map(({ x }) => x.falsePositives.length)), fn = sum(s.map(({ x }) => x.misses.length));
  const clean = s.filter(({ r }) => r.l.kind === "none");
  const cleanS = sum(clean.map(({ r }) => r.audioS));
  const warnings = sum(clean.map(({ x }) => x.confirmed.length));
  const perClass = Object.fromEntries(MEMORIZATION_CLASSES.map((c) => {
    const pos = sum(s.map(({ r }) => r.positives.filter((e) => e === c).length));
    const miss = sum(s.map(({ x }) => x.misses.filter((e) => e.errorClass === c).length));
    const fpc = sum(s.map(({ x }) => x.falsePositives.filter((e) => e.errorClass === c).length));
    return [c, { pos, tp: pos - miss, fp: fpc }];
  }));
  const sig = (e: { errorClass: string; location: WordRef }) => `${e.errorClass}/${e.location.surah}:${e.location.ayah}:${e.location.word}`;
  const raised = sum(s.map(({ r, x }) => r.clip.events.filter((e) => e.state === "confirmed" && MEMORIZATION_CLASSES.includes(e.errorClass) && x.final.some((p) => sig(p) === sig(e))).length));
  return { clips: s.length, tp, fp, fn, raised, perClass, cleanClips: clean.length, warnings, perClip: ratio(warnings, clean.length), per5: cleanS ? (warnings * 300) / cleanS : null,
    repeats: s.filter(({ r }) => r.l.kind === "repeat").length, repeatFp: sum(s.map(({ x }) => x.repeatFalsePositives)) };
}
function memoMetrics(prefix: string, group: string, m: ReturnType<typeof memo>, note: string, perClass: boolean, arm = "") {
  const nm = (n: string) => arm + n;
  add({ id: `${prefix}.precision`, group, name: nm("Memorization precision"), value: ratio(m.tp, m.tp + m.fp), unit: "%", display: frac(m.tp, m.tp + m.fp), bar: MEM.precision, note });
  add({ id: `${prefix}.recall`, group, name: nm("Memorization recall"), value: ratio(m.tp, m.tp + m.fn), unit: "%", display: frac(m.tp, m.tp + m.fn), bar: MEM.recall });
  if (perClass) for (const [c, x] of Object.entries(m.perClass)) {
    add({ id: `${prefix}.${c}.precision`, group, name: nm(`${c} precision`), value: ratio(x.tp, x.tp + x.fp), unit: "%", display: x.tp + x.fp ? frac(x.tp, x.tp + x.fp) : "— (0 raised)", bar: MEM.precision });
    add({ id: `${prefix}.${c}.recall`, group, name: nm(`${c} recall`), value: ratio(x.tp, x.pos), unit: "%", display: x.pos ? frac(x.tp, x.pos) : "— (0 positives)", bar: x.pos ? MEM.recall : undefined, status: x.pos ? undefined : "unmeasurable", note: x.pos ? "" : "no confirmed positives in the eval split" });
  }
  add({ id: `${prefix}.cleanPerClip`, group, name: nm("False warnings per clean clip"), value: m.perClip, display: `${m.perClip?.toFixed(3)} (${m.warnings}/${m.cleanClips})` });
  add({ id: `${prefix}.cleanPer5`, group, name: nm("False warnings per 5 min (short clips)"), value: m.per5, bar: MEM.per5, note: "short clips; the gating number needs phase-2 long sessions" });
  add({ id: `${prefix}.repeatFp`, group, name: nm("False positives on repeats"), value: m.repeatFp, display: `${m.repeatFp} (${m.repeats} repeat clips)`, bar: { op: "<=", target: 0 } });
}
const pendingHelp = (group: string, ids: [string, string][], why: string) => ids.forEach(([id, name]) => add({ id, group, name, value: null, status: "pending", note: why }));
const NO_EVIDENCE = `no help-clip evidence from ${values.listener} on disk (needs a GPU decode of the 388 eval clips)`;

const E2E = "End to end";
if (rows.length) {
  memoMetrics("e2e", E2E, memo(rows, "free"), `help eval split, held-out speakers: ${rows.length} clips, ${new Set(rows.map((r) => r.l.speaker)).size} speakers; free recitation, default policy`, true);
} else pendingHelp(E2E, [["e2e.precision", "Memorization precision"], ["e2e.recall", "Memorization recall"], ["e2e.cleanPer5", "False warnings per 5 min"]], NO_EVIDENCE);

const PRONUNCIATION = ["short_vowel", "consonant_sad_sin", "madd_natural"];
for (const c of PRONUNCIATION) {
  const pos = sum(usable.map((l) => l.events.filter((e) => e.errorClass === c).length));
  add({ id: `e2e.pron.${c}`, group: E2E, name: `Pronunciation: ${c}`, value: null, display: "unsupported", status: "unmeasurable", note: `checker has no pronunciation classes; ${pos} positives in eval split` });
}
add({ id: "e2e.pron.phase2", group: E2E, name: "Pronunciation: other pairs, shadda", value: null, status: "unmeasurable", note: "no recordings with positives (ticket 05, phase 2)" });
const perf = perfGate(index);
add({ id: "e2e.perfRtf", group: E2E, name: "Tracker real-time factor (this machine)", value: perf.rtf, bar: { op: "<=", target: perf.budgetRtf }, note: `perf.ts gate: device floor assumed ${perf.speedFactor}× slower; slowest push ${perf.maxPushMs.toFixed(1)} ms` });
add({ id: "e2e.latency", group: E2E, name: "Word highlight p95 / suspected / confirmed latency", value: null, status: "unmeasurable", note: "needs word onsets and device runs (E7, ticket 10)" });

// ---- Listener.
const LISTENER = "Listener";
const student = read("student-v1.json");
const run = student.runs.find((r: any) => r.name === values.listener);
const breakdown = read("teacher-per-breakdown-v1.json");
/** The student re-decoded for the PER breakdown; other students have no fair PER yet. */
const BREAKDOWN_RUN = "e2-libri-full";
const teacherFloor: number = breakdown.per.teacherVsSpliced.canon;
const CANON_NOTE = "canon PER vs spliced reference (teacher-per-breakdown-v1): nasal spelling and qalqala unified, runs collapsed, clip edges from the teacher; ignores shadda and duration";
for (const m of goldPerMetrics(read("gold-per-v1-scores.json"), read("listener-target-v1.json"), values.listener!)) metrics.push(m);
source = "teacher-per-breakdown-v1.md";
if (teacher) add({ id: "listener.perFair", group: LISTENER, name: "Fair PER on tlog clean-match test (canon vs spliced)", value: teacherFloor, unit: "%", status: "info", note: `teacher floor; ${CANON_NOTE}` });
else {
  const v = values.listener === BREAKDOWN_RUN ? breakdown.per.studentVsSpliced.canon : null;
  add({ id: "listener.perFair", group: LISTENER, name: "Fair PER on tlog clean-match test (canon vs spliced)", value: v, unit: "%",
    status: v === null ? "pending" : "info", note: v === null ? "not in the PER breakdown; run tools/per_breakdown.py on this run's decode" : `secondary to gold PER: teacher-labelled clips; ${CANON_NOTE}` });
}
source = "student-v1.md";
const teacherPer = student.runs[0]?.teacherPerVsRef ?? null;
add({ id: "listener.perRef", group: LISTENER, name: "Raw PER vs reference (tlog clean-match test)", value: teacher ? teacherPer : run?.perVsRef ?? null, unit: "%", status: "info", note: teacher ? "mostly clip edges and madd/ghunna length, not mishearing (teacher-per-breakdown-v1)" : "Quran-Lab zipformer 3.65% on phone audio: reference only, not a gate" });
add({ id: "listener.perTeacher", group: LISTENER, name: "Raw PER vs teacher", value: teacher ? 0 : run?.perVsTeacher ?? null, unit: "%", status: "info" });
source = "acceptance-v1.md";
if (rows.length) {
  const pos = rows.flatMap((r) => r.clip.events.filter((e) => e.state === "confirmed" && MEMORIZATION_CLASSES.includes(e.errorClass)).map((e) => visibleInEvidence(index, r.clip, e.location)));
  add({ id: "listener.visible", group: LISTENER, name: "Planted memorization errors visible in the decode", value: ratio(pos.filter(Boolean).length, pos.length), unit: "%", display: frac(pos.filter(Boolean).length, pos.length), note: "E0 heuristic: under half the planted word's consonants decode as written" });
} else pendingHelp(LISTENER, [["listener.visible", "Planted memorization errors visible in the decode"]], NO_EVIDENCE);
if (teacher) {
  source = "acceptance-v1.md";
  const e1h = read("acceptance-v1-e1.json");
  const sv = e1h.classes.find((c: any) => c.name === "short vowel");
  add({ id: "listener.e1help.svRecall", group: LISTENER, name: "E1 help clips: short vowel greedy recall", value: sv.greedy_hit / sv.pos, unit: "%", display: frac(sv.greedy_hit, sv.pos), bar: { op: ">=", target: 0.5 }, note: `E1 viable rule; ${sv.sites} sites; report ${e1h.commit}` });
  add({ id: "listener.e1help.svCanon", group: LISTENER, name: "E1 help clips: short vowel canonicalised", value: sv.greedy_canon / sv.pos, unit: "%", display: frac(sv.greedy_canon, sv.pos) });
  add({ id: "listener.e1help.svFp", group: LISTENER, name: "E1 help clips: short vowel clean greedy FP", value: sv.greedy_fp / sv.neg, unit: "%", display: frac(sv.greedy_fp, sv.neg), bar: { op: "<=", target: 0.02 } });
  const verdicts = (cs: any[]) => Object.entries(cs.reduce((o: Record<string, number>, c) => ({ ...o, [c.verdict]: (o[c.verdict] ?? 0) + 1 }), {})).map(([k, n]) => `${n} ${k}`).join(", ");
  const e1 = read("e1-teacher-bias-v1.json"), e1b = read("e1b-spliced-v1.json");
  source = "e1-teacher-bias-v1.md";
  const viable = (cs: any[]) => cs.filter((c) => c.verdict === "viable" || c.verdict === "posterior-only").length;
  add({ id: "listener.e1.classes", group: LISTENER, name: "E1 (qdat_bench + Ikhlas): classes the teacher hears", value: viable(e1.classes), display: `${viable(e1.classes)}/${e1.classes.length}: ${verdicts(e1.classes)}`, note: `report ${e1.commit}; most classes have < 20 positives` });
  source = "e1b-spliced-v1.md";
  add({ id: "listener.e1b.classes", group: LISTENER, name: "E1b splices (upper bound): classes the teacher hears", value: viable(e1b.classes), display: `${viable(e1b.classes)}/${e1b.classes.length}: ${verdicts(e1b.classes)}`, note: `report ${e1b.commit}` });
  const canon = e1b.classes.map((c: any) => [c.spliced.greedy_canon, c.spliced.pos]).reduce((a: number[], b: number[]) => [a[0] + b[0], a[1] + b[1]], [0, 0]);
  add({ id: "listener.e1b.canon", group: LISTENER, name: "E1b splices: canonicalised share (all classes)", value: canon[0] / canon[1], unit: "%", display: frac(canon[0], canon[1]) });
} else {
  for (const [id, name] of [["listener.e1help.svRecall", "E1 help clips: short vowel greedy recall"], ["listener.e1.classes", "E1: classes heard"], ["listener.e1b.classes", "E1b: classes heard"]]) {
    add({ id, group: LISTENER, name, value: null, status: "pending", note: `E1/E1b not run on ${values.listener}'s posteriors` });
  }
}

// ---- Tracker.
source = "acceptance-v1.md";
const TRACKER = "Tracker";
const T = { within: { op: ">=", target: 0.95 }, jumps: { op: "<=", target: 0.02 } } as const;
if (rows.length) {
  for (const [tag, rs, label] of [["clean", rows.filter((r) => r.l.kind === "none"), "help clean"], ["all", rows, "help all"]] as const) {
    const g = rs.map((r) => r.free.gate);
    const w = g.filter((x) => x.correctWithin5s).length;
    const fj = g.filter((x) => x.falseJumps > 0).length;
    add({ id: `tracker.${tag}.within5s`, group: TRACKER, name: `${label}: correct ayah within 5 s`, value: w / g.length, unit: "%", display: frac(w, g.length), bar: T.within, note: tag === "clean" ? "fair gate number on user-like audio" : "error clips: a substitution forming another ayah counts as wrong" });
    add({ id: `tracker.${tag}.falseJumps`, group: TRACKER, name: `${label}: clips with a false jump`, value: fj / g.length, unit: "%", display: frac(fj, g.length), bar: T.jumps });
    add({ id: `tracker.${tag}.outside`, group: TRACKER, name: `${label}: clips with any lock outside the passage`, value: g.filter((x) => x.outsideLock).length / g.length, unit: "%", display: frac(g.filter((x) => x.outsideLock).length, g.length) });
    add({ id: `tracker.${tag}.lockS`, group: TRACKER, name: `${label}: median time to correct lock`, value: median(g.filter((x) => x.firstCorrectLockMs !== null).map((x) => (x.firstCorrectLockMs! - x.onsetMs) / 1000)), unit: "s", note: "from first decoded phoneme" });
  }
} else pendingHelp(TRACKER, [["tracker.clean.within5s", "help clean: correct ayah within 5 s"], ["tracker.all.within5s", "help all: correct ayah within 5 s"]], NO_EVIDENCE);
const qudFiles = existsSync(values.qud!) ? readdirSync(values.qud!).filter((f) => f.endsWith(".json")).sort() : [];
source = "qud-v1-decoded-phrases.md";
if (teacher && qudFiles.length) {
  const s = summarise(qudFiles.map((f) => benchRecording(index, JSON.parse(readFileSync(join(values.qud!, f), "utf8")) as QudRecording, { posteriors: false, phrases })));
  add({ id: "tracker.qud.within5s", group: TRACKER, name: "QUD regression: correct ayah within 5 s", value: s.within5s / s.starts, unit: "%", display: frac(s.within5s, s.starts), bar: T.within, note: `${s.recordings} recordings, ${s.audioMinutes.toFixed(0)} min; regression report, not a gate (Decision 2)` });
  add({ id: "tracker.qud.falseJumps", group: TRACKER, name: "QUD regression: false jumps", value: s.falseJumps / s.jumps, unit: "%", display: frac(s.falseJumps, s.jumps), bar: T.jumps });
  add({ id: "tracker.qud.lockS", group: TRACKER, name: "QUD regression: median time to first lock", value: s.medianTimeToFirstLockS, unit: "s" });
} else add({ id: "tracker.qud.within5s", group: TRACKER, name: "QUD regression: correct ayah within 5 s", value: null, status: "pending", note: teacher ? "QUD evidence not on disk" : `no QUD evidence from ${values.listener}` });
source = "acceptance-v1.md";
add({ id: "tracker.tlog", group: TRACKER, name: "Ambiguous tlog clips (PLAN gate set)", value: null, status: "unmeasurable", note: "no passage truth: the sampled ambiguous clips are wrong-passage or no-recitation (ticket 06)" });
add({ id: "tracker.scripted", group: TRACKER, name: "Scripted-error recordings (PLAN gate set)", value: null, status: "unmeasurable", note: "no recordings yet (ticket 05)" });

// ---- Checker isolated.
const CHECKER = "Checker";
if (rows.length) {
  memoMetrics("checker.oracleLock", CHECKER, memo(rows, "declared"), "declared passage (CheckerOptions.passage): the prompted passage, real tracker and evidence", true, "Declared passage: ");
  const oc = memo(rows, "oracle");
  memoMetrics("checker.oracleAll", CHECKER, oc, `oracle lock + oracle evidence synthesised from labels (Decision 7); ${rows.length - oc.clips} clips with an unphonemizable planted word skipped`, true, "Oracle lock + evidence: ");
  add({ id: "checker.oracleAll.raised", group: CHECKER, name: "Oracle lock + evidence: positives raised in any state", value: ratio(oc.raised, oc.tp + oc.fn), unit: "%", display: frac(oc.raised, oc.tp + oc.fn), note: "the rest of the gap to recall is events still suspected/uncertain at clip end (confirmation delay on short clips)" });
  const causes = { evidence: 0, tracker: 0, checker: 0 };
  for (const r of rows) for (const a of attribute(index, r.clip, r.free, r.declared)) causes[a.cause]++;
  const misses = causes.evidence + causes.tracker + causes.checker;
  for (const c of ["evidence", "tracker", "checker"] as const) {
    add({ id: `checker.e0.${c}`, group: CHECKER, name: `E0: misses lost to ${c}`, value: causes[c], display: `${causes[c]}/${misses}`, note: c === "checker" ? "real upstream, free arm" : "" });
  }
} else pendingHelp(CHECKER, [["checker.oracleLock.recall", "Oracle lock: recall"], ["checker.e0.checker", "E0 attribution"]], NO_EVIDENCE);
add({ id: "checker.selfCorrection", group: CHECKER, name: "Self-correction / long-session false warnings", value: null, status: "unmeasurable", note: "no recordings (ticket 05)" });

// ---- Row.
const sha = values.sha ?? execSync("git rev-parse --short HEAD").toString().trim();
const row: ScoreRow = { sha, date: new Date().toISOString().slice(0, 10), listener: values.listener!, note: values.note, metrics };
const jsonPath = `${resolve(values.out!)}.json`;
const board: { version: number; rows: ScoreRow[] } = existsSync(jsonPath) ? JSON.parse(readFileSync(jsonPath, "utf8")) : { version: 1, rows: [] };
// Keep existing attribution when regenerating a run without a new module tag.
row.modules = modules as ScoreRow["modules"] ?? board.rows.find((r) => r.sha === row.sha && r.listener === row.listener)?.modules;
board.rows = upsertRow(board.rows, row);
writeFileSync(jsonPath, JSON.stringify(board, null, 1) + "\n");
const HEADLINE = ["e2e.precision", "e2e.recall", "e2e.cleanPer5", "listener.goldPer", "tracker.clean.within5s", "tracker.qud.within5s", "tracker.qud.falseJumps", "checker.oracleLock.recall", "checker.oracleAll.recall"];
const md = [
  "# Engine scoreboard", "",
  "One row per commit × listener, recomputed from evidence on disk (no GPU). Bars are PLAN.md's v1 acceptance bars; `unmeasurable` means the data to score it doesn't exist yet, `pending` means the evidence exists in principle but hasn't been produced for this listener.", "",
  "**How to run / add a row.** From `packages/core`: `HELP_DIR=<repo>/.scratch/help QUD_DIR=<repo>/.scratch/bench/qud pnpm scoreboard --note '<what changed and why>'` (teacher). The note is required by AGENTS.md. Tag changed engine modules with `--modules checker` (or a comma-separated list such as `--modules listener,tracker`). Tags describe the implementation change, not every metric that moved; the dashboard uses them to filter the change history. Existing tags survive reruns that omit this option. For a student, decode the help eval clips with it into a directory of `<clip>.evidence.json` and run `pnpm scoreboard --listener <run-name> --evidence <dir> --note '...'`; without `--evidence` its help metrics are `pending`. Each run replaces the row with the same commit and listener, else appends one; commit the updated `scoreboard.json` and `scoreboard.md` with the change they measure. Listener PER is gold PER from `gold-per-v1-scores.json` (`tools/gold_set.py score`), gated by the bar in `listener-target-v1.json`; raw and clean-match fair PER come from `student-v1.json` and the PER breakdown. E1/E1b from their reports; rerun those first if the listener changed. Help clips: 388 scored eval clips, 29 held-out speakers; letter-confusion substitutions count as negatives. `Declared passage` gives the checker the prompted passage (`CheckerOptions.passage`), the memorization-session arm; `oracle evidence` replaces the decode with a perfect one synthesised from the label, so its losses are the checker's own.", "",
  "**Dashboard.** `uv run tools/scoreboard_serve.py` serves the dashboard on http://127.0.0.1:8765/scoreboard.html. Choose a listener model to see end-to-end precision and recall, listener/tracker/checker summaries and trends, and missed-error attribution. The change history shows module tags, key figures and deltas, with filters for Listener, Tracker and Checker. Full notes are expandable; comparisons always use the previous run for that model, even when filtered. Expand “Explore all measurements” for sample counts, targets and backing reports. It reads `scoreboard.json`, so a new row appears on reload.", "",
  renderScoreboard(board.rows, HEADLINE),
].join("\n");
writeFileSync(`${resolve(values.out!)}.md`, md);
console.log(md);
