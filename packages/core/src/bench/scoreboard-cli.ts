/**
 * Appends one scoreboard row (commit × listener) to docs/benchmarks/scoreboard.json and regenerates scoreboard.md.
 * CPU only: replays evidence already on disk through the tracker and checker, and reads the listener reports.
 */
import { execSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "../reference.js";
import type { PhraseAsset, ReferenceAsset } from "../reference.js";
import type { WordRef } from "../events.js";
import { MEMORIZATION_CLASSES, attribute, scoreClip, slices, visibleInEvidence } from "./acceptance.js";
import type { AcceptanceClip, ClipScore } from "./acceptance.js";
import { benchRecording, summarise } from "./index.js";
import type { QudRecording } from "./index.js";
import { perfGate } from "./perf.js";
import { inClip, withContext } from "./leadin.js";
import type { ClipEvidence, Segment } from "./jumps.js";
import { CLIP_DUMP_VERSION, clipWords, dumpArm } from "./clip-dump.js";
import type { ClipDump, DumpClip } from "./clip-dump.js";
import { REPLAY_DEFAULTS } from "./component-replay.js";
import { BLOCKED, DEFAULT_CANDIDATE, GATES, goldPerMetrics, oracleClip, renderScoreboard, scored, upsertRow } from "./scoreboard.js";
import type { Board, Metric, OracleLabel, ScoreRow } from "./scoreboard.js";
import { ERROR_CLASSES, summarizeSessions } from "./sessions.js";
import type { DelaySummary, RecallSlice } from "./sessions.js";

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
  /** Lead-in manifest (`tools/leadin_build.py`): 1–2 correct ayahs stitched before each help clip, and 1–2 after it. */
  leadin: { type: "string", default: "../../docs/benchmarks/leadin-v2-manifest.json" },
  /** This listener's evidence of the tlog lead-ins and lead-outs, `<shard>_<row>.evidence.json`; defaults to the teacher's. */
  "leadin-evidence": { type: "string" },
  /** `pnpm bench:help` report (packages/sdk) for this listener: end-to-end latency on the shipped runtime. */
  latency: { type: "string" },
  /** Listener judged for release; sticks in scoreboard.json until changed. */
  candidate: { type: "string" },
  /** Rewrite scoreboard.md (and the candidate) from scoreboard.json without scoring a run. */
  render: { type: "boolean", default: false },
  /** Where per-clip outcomes go for the dashboard's mistake review (`<sha>-<listener>.json` and `index.json`). */
  "clip-dump": { type: "string", default: process.env.CLIP_DUMP_DIR ?? "../../.scratch/scoreboard/clips" },
  /** Write the clip dump for `--sha` without adding or replacing a scoreboard row. */
  "dump-only": { type: "boolean", default: false },
  /** Reported-only representative session measures, from sessions-cli.ts. Never replaces release gates. */
  sessions: { type: "string" },
} });
const jsonPath = `${resolve(values.out!)}.json`;
const board: Board = existsSync(jsonPath) ? JSON.parse(readFileSync(jsonPath, "utf8")) : { version: 1, rows: [] };
function save() {
  board.candidate = values.candidate ?? board.candidate ?? DEFAULT_CANDIDATE;
  board.gates = GATES;
  board.blocked = BLOCKED;
  writeFileSync(jsonPath, JSON.stringify(board, null, 1) + "\n");
  writeFileSync(`${resolve(values.out!)}.md`, [
    "# Engine scoreboard", "",
    "The release cockpit for the engine. One row per commit × listener, recomputed from evidence on disk (no GPU).", "",
    "- **Release readiness** judges the candidate listener (`--candidate`, sticky) against the v1 gates in `GATES` (`packages/core/src/bench/scoreboard.ts`). It is ready when every gate passes and every requirement in `BLOCKED` has data. Gates are ranked by shortfall from the bar, so the top row is what most needs work.",
    "- **Experiments** judges every run against its parent: the previous run of the same listener, or the candidate for a listener's first run. The verdict counts only gate moves larger than two clips; smaller ones are marked `~` and count as noise, since the eval set is 64 planted mistakes and 225 clean clips.",
    "- **All metrics** lists every diagnostic for the candidate.", "",
    "**Add a row.** From `packages/core`: `HELP_DIR=<repo>/.scratch/help QUD_DIR=<repo>/.scratch/bench/qud pnpm scoreboard --note '<what changed and why>' [--modules listener,tracker,checker]` (teacher). For a student, decode the help eval clips into a directory of `<clip>.evidence.json` and add `--listener <run-name> --evidence <dir>`. A run replaces the row with the same commit and listener, else appends. Commit `scoreboard.json` and `scoreboard.md` with the change. `pnpm scoreboard --render --candidate <listener>` switches the candidate without scoring.", "",
    "**Dashboard.** `uv run tools/scoreboard_serve.py`, then http://127.0.0.1:8765/scoreboard.html. **Mistake review** (on the dashboard; click a failing gate): every miss, false warning and late lock per clip, with audio and the decode per word. Each run writes its per-clip outcomes to `.scratch/scoreboard/clips/` (private, gitignored); add `--dump-only --sha <commit>` to write the dump without touching the row. See the [manual review guide](scoreboard-review.md) for incident grouping, evidence interpretation, storage migration and JSON export.", "",
    "**Sources.** Listener PER is gold PER (`tools/gold_set.py score`), barred by `listener-target-v1.json`. Help clips: 388 eval clips, 29 held-out speakers. Free memorization is gated `with lead-in` (`--leadin`, `tools/leadin_build.py`): 1–2 correctly recited preceding ayahs are stitched before each clip and 1–2 following ayahs after it (lead-out) at the evidence level (same speaker's clean clip, else another help speaker's, else held-out tlog, else tlog train), errors and false positives are scored inside the original clip, and warnings on the lead-in and lead-out are counted apart. For a student add `--leadin-evidence <dir>` with its decode of `.scratch/leadin/audio` (with posteriors) (`pnpm --filter @tilawa/sdk decode`). Without a lead-in the free numbers are info; each arm is also split into `unambiguous` and `mutashabihat` (planted errors that form another ayah, `forms_other_ayah`). `Declared passage` is the memorization-session arm (`CheckerOptions.passage`); `oracle evidence` replaces the decode with one synthesised from the label, so its losses are the checker's own. E0 attributes each missed mistake to listener, tracker or checker.", "",
    renderScoreboard(board.rows, board.candidate),
  ].join("\n"));
}
if (values.render) {
  save();
  process.exit(0);
}
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
// ---- Lead-in/out: the same clips with 1–2 correctly recited ayahs in front and after; errors scored inside the clip only.
type LeadSource = { kind: string; id: string; ayahs: string[] };
interface LeadManifest { version: string; gapMs: number; counts: Record<string, number>; followCounts?: Record<string, number>; clips: { clip: string; lead: LeadSource | null; reason?: string; follow?: LeadSource | null }[] }
const leadManifest: LeadManifest | null = existsSync(resolve(values.leadin!)) ? JSON.parse(readFileSync(resolve(values.leadin!), "utf8")) : null;
const leadDir = values["leadin-evidence"] ? resolve(values["leadin-evidence"]) : teacher ? resolve(process.env.LEADIN_DIR ?? "../../.scratch/leadin", "evidence/teacher") : null;
const byId = new Map(rows.map((r) => [r.l.id, r]));
const leadPath = (lead: { kind: string; id: string }) => lead.kind.startsWith("help")
  ? evidencePath(labels.find((l) => l.id === lead.id)!)!
  : leadDir && resolve(leadDir, `${lead.id.replace(":", "_")}.evidence.json`);
const leadEntries = leadManifest?.clips.filter((c) => byId.has(c.clip)) ?? [];
const onDisk = (s: LeadSource | null | undefined) => !s || (!!leadPath(s) && existsSync(leadPath(s)!));
const haveLeads = rows.length > 0 && leadEntries.length === rows.length && leadEntries.every((c) => onDisk(c.lead) && onDisk(c.follow));
const leadRows: Row[] = [];
interface LeadRaw { free: ClipScore; declared: ClipScore; segments: Segment[]; segmentAudio: string[]; leadAyahs: string[]; followAyahs: string[] }
const leadRaw = new Map<string, LeadRaw>();
const audioOf = (s: { kind: string; id: string }) => s.kind.startsWith("help") ? `help/${basename(labels.find((l) => l.id === s.id)!.wav)}` : `leadin/${s.id.replace(":", "_")}.wav`;
const leadWarnings = { free: 0, declared: 0, seconds: 0, clips: 0 };
const followWarnings = { free: 0, declared: 0, seconds: 0, clips: 0 };
if (haveLeads) {
  for (const c of leadEntries) {
    const r = byId.get(c.clip)!;
    if (!c.lead && !c.follow) { leadRows.push(r); continue; }
    const load = (s: LeadSource | null | undefined) => s ? { source: s.id, ayahs: s.ayahs, evidence: JSON.parse(readFileSync(leadPath(s)!, "utf8")) as ClipEvidence } : null;
    const lead = load(c.lead), follow = load(c.follow);
    const { clip, leadAyahs, followAyahs, segments } = withContext(index, r.clip, { lead, follow }, leadManifest!.gapMs);
    const rawFree = scoreClip(index, clip, { arm: "free", phrases }), rawDeclared = scoreClip(index, clip, { arm: "declared", phrases });
    const free = inClip(rawFree, leadAyahs, followAyahs);
    const declared = inClip(rawDeclared, leadAyahs, followAyahs);
    leadRaw.set(r.l.id, {
      free: rawFree, declared: rawDeclared, segments, leadAyahs, followAyahs,
      segmentAudio: [...(c.lead ? [audioOf(c.lead)] : []), `help/${basename(r.l.wav)}`, ...(c.follow ? [audioOf(c.follow)] : [])],
    });
    for (const [acc, l, n] of [[leadWarnings, lead, "leadInWarnings"], [followWarnings, follow, "leadOutWarnings"]] as const) {
      if (!l) continue;
      acc.free += free[n];
      acc.declared += declared[n];
      acc.seconds += (l.evidence.nFrames * l.evidence.frameMs) / 1000;
      acc.clips++;
    }
    leadRows.push({ ...r, clip, free: free.score, declared: declared.score, oracle: null });
    if (leadRows.length % 100 === 0) console.log(`lead-in ${leadRows.length}/${rows.length}`);
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
  const raisedOf = (e: AcceptanceClip["events"][number], x: ClipScore) => e.state === "confirmed" && MEMORIZATION_CLASSES.includes(e.errorClass) && x.final.some((p) => sig(p) === sig(e));
  const raised = sum(s.map(({ r, x }) => r.clip.events.filter((e) => raisedOf(e, x)).length));
  const mutashabihatRaised = sum(s.map(({ r, x }) => r.clip.events.filter((e) => e.forms_other_ayah && raisedOf(e, x)).length));
  const slice = (k: "unambiguous" | "mutashabihat") => s.map(({ r, x }) => slices(r.clip, x)[k]).reduce((a, b) => ({ pos: a.pos + b.pos, tp: a.tp + b.tp, fp: a.fp + b.fp }), { pos: 0, tp: 0, fp: 0 });
  return { clips: s.length, tp, fp, fn, raised, mutashabihatRaised, perClass, unambiguous: slice("unambiguous"), mutashabihat: slice("mutashabihat"), cleanClips: clean.length, warnings, perClip: ratio(warnings, clean.length), per5: cleanS ? (warnings * 300) / cleanS : null,
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
const MUTASHABIHAT = "planted errors whose recitation forms another ayah at least as well (forms_other_ayah)";
function sliceMetrics(prefix: string, group: string, m: ReturnType<typeof memo>, arm: string, mutashabihatGated: boolean) {
  for (const k of ["unambiguous", "mutashabihat"] as const) {
    const x = m[k], gated = k === "unambiguous" || mutashabihatGated;
    const note = k === "unambiguous" ? `excludes ${m.mutashabihat.pos} ${MUTASHABIHAT}; all false positives count here`
      : `${MUTASHABIHAT}; no false positives attributed${gated ? "" : "; needs declared passage or position hint"}`;
    add({ id: `${prefix}.${k}.precision`, group, name: `${arm}${k} precision`, value: ratio(x.tp, x.tp + x.fp), unit: "%", display: frac(x.tp, x.tp + x.fp), bar: gated ? MEM.precision : undefined, status: gated ? undefined : "info", note });
    add({ id: `${prefix}.${k}.recall`, group, name: `${arm}${k} recall`, value: ratio(x.tp, x.pos), unit: "%", display: frac(x.tp, x.pos), bar: gated ? MEM.recall : undefined, status: gated ? undefined : "info" });
  }
}
const pendingHelp = (group: string, ids: [string, string][], why: string) => ids.forEach(([id, name]) => add({ id, group, name, value: null, status: "pending", note: why }));
const NO_EVIDENCE = `no help-clip evidence from ${values.listener} on disk (needs a GPU decode of the 388 eval clips)`;

const E2E = "End to end";
const LEAD_NOTE = leadManifest && `${leadManifest.version}: ${leadEntries.filter((c) => c.lead).length}/${leadEntries.length} clips get 1–2 preceding ayahs (${Object.entries(leadManifest.counts).map(([k, n]) => `${k} ${n}`).join(", ")})${leadManifest.followCounts ? `, ${leadEntries.filter((c) => c.follow).length}/${leadEntries.length} 1–2 following ayahs (${Object.entries(leadManifest.followCounts).map(([k, n]) => `${k} ${n}`).join(", ")})` : ""}, ${leadManifest.gapMs} ms silence, stitched at the evidence level; errors and false positives scored inside the original clip`;
if (leadRows.length) {
  const free = memo(leadRows, "free");
  memoMetrics("e2e.leadin", E2E, free, `free recitation with lead-in, the tracked free metric; ${LEAD_NOTE}`, false, "Free, with lead-in: ");
  sliceMetrics("e2e.leadin", E2E, free, "Free, with lead-in: ", true);
  add({ id: "e2e.leadin.mutashabihat.raised", group: E2E, name: "Free, with lead-in: mutashabihat raised in any state", value: ratio(free.mutashabihatRaised, free.mutashabihat.pos), unit: "%", display: frac(free.mutashabihatRaised, free.mutashabihat.pos), note: "the rest of the gap to recall is events still suspected/uncertain when the session ends" });
  add({ id: "e2e.leadin.leadWarnings", group: E2E, name: "Free, with lead-in: false warnings in the lead-in", value: leadWarnings.seconds ? (leadWarnings.free * 300) / leadWarnings.seconds : null, bar: MEM.per5,
    display: `${(leadWarnings.seconds ? (leadWarnings.free * 300) / leadWarnings.seconds : 0).toFixed(2)} per 5 min (${leadWarnings.free} in ${(leadWarnings.seconds / 60).toFixed(1)} min, ${leadWarnings.clips} lead-ins)`, note: "confirmed events of any memorization class on the lead-in's ayahs; all false" });
  add({ id: "e2e.leadin.followWarnings", group: E2E, name: "Free, with lead-in: false warnings in the lead-out", value: followWarnings.seconds ? (followWarnings.free * 300) / followWarnings.seconds : null, bar: MEM.per5,
    display: `${(followWarnings.seconds ? (followWarnings.free * 300) / followWarnings.seconds : 0).toFixed(2)} per 5 min (${followWarnings.free} in ${(followWarnings.seconds / 60).toFixed(1)} min, ${followWarnings.clips} lead-outs)`, note: "confirmed events of any memorization class on the lead-out's ayahs; all false" });
} else if (rows.length) pendingHelp(E2E, [["e2e.leadin.precision", "Free, with lead-in: Memorization precision"], ["e2e.leadin.recall", "Free, with lead-in: Memorization recall"]],
  leadManifest ? `no lead-in evidence from ${values.listener} on disk (decode .scratch/leadin/audio, pass --leadin-evidence)` : "no lead-in manifest; run tools/leadin_build.py");
if (rows.length) {
  const free = memo(rows, "free");
  const before = metrics.length;
  memoMetrics("e2e", E2E, free, `help eval split, held-out speakers: ${rows.length} clips, ${new Set(rows.map((r) => r.l.speaker)).size} speakers; free recitation, default policy${leadRows.length ? "; no lead-in, info only: the clip starts the recitation" : ""}`, true);
  sliceMetrics("e2e", E2E, free, "Free: ", false);
  if (leadRows.length) for (const m of metrics.slice(before)) if (/^e2e\.(precision|recall|unambiguous\.|mutashabihat\.|(wrong_word|omitted_word|added_word|skipped_passage)\.)/.test(m.id)) { delete m.bar; m.status = m.value === null ? m.status : "info"; }
} else pendingHelp(E2E, [["e2e.precision", "Memorization precision"], ["e2e.recall", "Memorization recall"], ["e2e.cleanPer5", "False warnings per 5 min"]], NO_EVIDENCE);

const PRONUNCIATION = ["short_vowel", "consonant_sad_sin", "madd_natural"];
for (const c of PRONUNCIATION) {
  const pos = sum(usable.map((l) => l.events.filter((e) => e.errorClass === c).length));
  add({ id: `e2e.pron.${c}`, group: E2E, name: `Pronunciation: ${c}`, value: null, display: "unsupported", status: "unmeasurable", note: `checker has no pronunciation classes; ${pos} positives in eval split` });
}
add({ id: "e2e.pron.phase2", group: E2E, name: "Pronunciation: other pairs, shadda", value: null, status: "unmeasurable", note: "no recordings with positives (ticket 05, phase 2)" });
const perf = perfGate(index);
add({ id: "e2e.perfRtf", group: E2E, name: "Tracker real-time factor (this machine)", value: perf.rtf, bar: { op: "<=", target: perf.budgetRtf }, note: `perf.ts gate: device floor assumed ${perf.speedFactor}× slower; slowest push ${perf.maxPushMs.toFixed(1)} ms` });
if (values.latency) {
  const r = JSON.parse(readFileSync(resolve(values.latency), "utf8")) as { runtime: string; pushMs: number; latencyMs: Record<"highlight" | "suspected", { n: number; p50: number; p95: number }> };
  const note = `${r.runtime}, ${r.pushMs} ms pushes, this machine; measureLatency (packages/sdk): from the decoded phoneme, not a labelled onset. Device runs: ticket 10`;
  const ms = (x: number) => `${Math.round(x)} ms`;
  const h = r.latencyMs.highlight, q = r.latencyMs.suspected;
  add({ id: "e2e.latency.highlight", group: E2E, name: "Word highlight latency p95", value: h.p95, bar: { op: "<=", target: 600 }, display: `${ms(h.p95)} (p50 ${ms(h.p50)}, ${h.n} words)`, note });
  add({ id: "e2e.latency.suspected", group: E2E, name: "Suspected latency p95", value: q.p95, bar: { op: "<=", target: 1000 }, display: `${ms(q.p95)} (p50 ${ms(q.p50)}, ${q.n} errors)`, note: "from the onset of the word after the error; PLAN: within about 1 s" });
} else add({ id: "e2e.latency", group: E2E, name: "Word highlight p95 / suspected / confirmed latency", value: null, status: "unmeasurable", note: "pass --latency <pnpm bench:help report> (packages/sdk); device runs: ticket 10" });

// ---- Listener.
const LISTENER = "Listener";
const student = read("student-v1.json");
const studentV2 = existsSync(join(bench, "student-v2.json")) ? read("student-v2.json") : { runs: [] };
const run = [...student.runs, ...studentV2.runs].find((r: any) => r.name === values.listener);
const breakdown = read("teacher-per-breakdown-v1.json");
/** e2-libri-full is in the teacher breakdown; later students have their own `per-breakdown-<run>-v1.json`. */
const BREAKDOWN_RUN = "e2-libri-full";
const ownBreakdown = `per-breakdown-${values.listener}-v1.json`;
const teacherFloor: number = breakdown.per.teacherVsSpliced.canon;
const CANON_NOTE = "canon PER vs spliced reference (teacher-per-breakdown-v1): nasal spelling and qalqala unified, runs collapsed, clip edges from the teacher; ignores shadda and duration";
for (const m of goldPerMetrics(read("gold-per-v1-scores.json"), read("listener-target-v1.json"), values.listener!)) metrics.push(m);
source = "teacher-per-breakdown-v1.md";
if (teacher) add({ id: "listener.perFair", group: LISTENER, name: "Fair PER on tlog clean-match test (canon vs spliced)", value: teacherFloor, unit: "%", status: "info", note: `teacher floor; ${CANON_NOTE}` });
else {
  const v = values.listener === BREAKDOWN_RUN ? breakdown.per.studentVsSpliced.canon
    : existsSync(join(bench, ownBreakdown)) ? read(ownBreakdown).per.studentVsSpliced.canon : null;
  add({ id: "listener.perFair", group: LISTENER, name: "Fair PER on tlog clean-match test (canon vs spliced)", value: v, unit: "%",
    status: v === null ? "pending" : "info", note: v === null ? "not in the PER breakdown; run tools/per_breakdown.py on this run's decode" : `secondary to gold PER: teacher-labelled clips; ${CANON_NOTE}` });
}
source = studentV2.runs.includes(run) ? "student-v2.md" : "student-v1.md";
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
  const own = `e1b-student-${values.listener}-v1`;
  if (existsSync(join(bench, `${own}.json`))) {
    source = `${own}.md`;
    const e1b = read(`${own}.json`);
    const canon = e1b.classes.map((c: any) => [c.spliced.greedy_canon, c.spliced.pos]).reduce((a: number[], b: number[]) => [a[0] + b[0], a[1] + b[1]], [0, 0]);
    add({ id: "listener.e1b.canon", group: LISTENER, name: "E1b splices: canonicalised share (all classes)", value: canon[0] / canon[1], unit: "%", display: frac(canon[0], canon[1]), note: `report ${e1b.commit}` });
  }
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
source = "jumps-v1.md";
const jumps = teacher && existsSync(join(bench, "jumps-v1.json")) ? read("jumps-v1.json").groups.All : null;
const shipped = existsSync(join(bench, "stitched-warnings-v1.json"))
  ? read("stitched-warnings-v1.json").results.find((r: any) => r.arm.listener === values.listener && r.arm.posteriors && r.arm.finalise)
  : null;
if (shipped) {
  source = "stitched-warnings-v1.md";
  const t = shipped.tally;
  add({ id: "e2e.stitched.falseWords", group: E2E, name: "Stitched sessions: false wrong/added words per 5 min (free)", value: (t.warnings * 5) / t.minutes, bar: MEM.per5,
    display: `${((t.warnings * 5) / t.minutes).toFixed(2)} (${t.warnings} in ${t.minutes.toFixed(0)} min)`, note: `ticket 13 sessions with posteriors (local-window on) and session-end finalisation; ${t.mid} mid-session, ${t.tail} in the last 20 s, ${t.atEnd} confirmed by the session end; \`pnpm stitched-warnings\`` });
} else if (jumps?.checkerWarnings != null) {
  add({ id: "e2e.stitched.falseWords", group: E2E, name: "Stitched sessions: false wrong/added words per 5 min (free)", value: jumps.checkerWarningsPer5Min, bar: MEM.per5,
    display: `${jumps.checkerWarningsPer5Min.toFixed(2)} (${jumps.checkerWarnings} in ${jumps.minutes.toFixed(0)} min)`, note: "clean clips stitched end to end with real jumps (ticket 13); `pnpm jumps`" });
} else add({ id: "e2e.stitched.falseWords", group: E2E, name: "Stitched sessions: false wrong/added words per 5 min (free)", value: null, status: "pending", note: teacher ? "run `pnpm jumps`" : "teacher evidence only" });
source = "acceptance-v1.md";
add({ id: "tracker.tlog", group: TRACKER, name: "Ambiguous tlog clips (PLAN gate set)", value: null, status: "unmeasurable", note: "no passage truth: the sampled ambiguous clips are wrong-passage or no-recitation (ticket 06)" });
add({ id: "tracker.scripted", group: TRACKER, name: "Scripted-error recordings (PLAN gate set)", value: null, status: "unmeasurable", note: "no recordings yet (ticket 05)" });

// ---- Checker isolated.
const CHECKER = "Checker";
if (rows.length) {
  const declared = memo(rows, "declared");
  memoMetrics("checker.oracleLock", CHECKER, declared, "declared passage (CheckerOptions.passage): the prompted passage, real tracker and evidence", true, "Declared passage: ");
  sliceMetrics("checker.oracleLock", CHECKER, declared, "Declared passage: ", true);
  if (leadRows.length) {
    const d = memo(leadRows, "declared");
    memoMetrics("checker.leadin", CHECKER, d, `declared passage from the lead-in's first ayah; ${LEAD_NOTE}`, false, "Declared, with lead-in: ");
    sliceMetrics("checker.leadin", CHECKER, d, "Declared, with lead-in: ", true);
    add({ id: "checker.leadin.leadWarnings", group: CHECKER, name: "Declared, with lead-in: false warnings in the lead-in", value: leadWarnings.seconds ? (leadWarnings.declared * 300) / leadWarnings.seconds : null, bar: MEM.per5,
      display: `${(leadWarnings.seconds ? (leadWarnings.declared * 300) / leadWarnings.seconds : 0).toFixed(2)} per 5 min (${leadWarnings.declared} in ${(leadWarnings.seconds / 60).toFixed(1)} min)` });
    add({ id: "checker.leadin.followWarnings", group: CHECKER, name: "Declared, with lead-in: false warnings in the lead-out", value: followWarnings.seconds ? (followWarnings.declared * 300) / followWarnings.seconds : null, bar: MEM.per5,
      display: `${(followWarnings.seconds ? (followWarnings.declared * 300) / followWarnings.seconds : 0).toFixed(2)} per 5 min (${followWarnings.declared} in ${(followWarnings.seconds / 60).toFixed(1)} min)` });
  }
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

// ---- Representative sessions: reported only, with source/track/context kept apart.
if (values.sessions) {
  const sessionReport = JSON.parse(readFileSync(resolve(values.sessions), "utf8"));
  if (sessionReport.version !== 1 || sessionReport.gating !== false || sessionReport.listener !== values.listener)
    throw new Error("Session report must be version 1, non-gating, and from the selected listener");
  // Spec "Scoring, per track": measures the track would gate on, and the context level it gates at. Still reported only.
  const SPEC_GATES: Record<string, { measures: string[]; contexts: string[] }> = {
    recitation: { measures: ["cleanPer5", "wrongPositions", "located", "within5s", "nonErrorWarnings"], contexts: ["none"] },
    memorization: { measures: ["recall", "precision", "wrongPositions", "located", "within5s", "confirmedDelay.p90", "nonErrorWarnings"], contexts: ["assignment", "resume"] },
  };
  const delayText = (d: DelaySummary) => `n=${d.n}, censored by miss ${d.censoredByMiss}`;
  for (const g of summarizeSessions(sessionReport.rows)) {
    const prefix = `sessions.${g.key}`;
    const base = `${g.sessions} sessions, ${g.speakers} speakers; reported only, coverage and recorded agreement pending; ${g.unsupportedContexts} unsupported contexts`;
    const gate = SPEC_GATES[g.track] as (typeof SPEC_GATES)[string] | undefined;
    const note = (id: string, extra?: string) => [base, extra,
      gate?.measures.includes(id) ? `spec **gate** for ${g.track}${gate.contexts.includes(g.context) ? "" : ` at ${gate.contexts.join("/")}; reported at ${g.context}`} (not gating yet)` : undefined]
      .filter(Boolean).join("; ");
    const hitsOf = (s: RecallSlice) => s.errors ? `${s.hits}/${s.errors}` : undefined;
    type Row = [string, string, number | null, Metric["unit"], string?, string?];
    const measures: Row[] = [
      ["precision", "Warnings correct", g.precision, "%", g.warnings ? `${g.hits}/${g.warnings}` : undefined],
      ["recall", "Errors caught", g.recall, "%", hitsOf({ errors: g.errors, hits: g.hits, recall: g.recall }), "abstentions count as misses"],
      ...ERROR_CLASSES.map((c): Row =>
        [`recall.${c}`, `Errors caught: ${c.replace("_", " ")}`, g.byClass[c].recall, "%", hitsOf(g.byClass[c])]),
      ["recall.mutashabihat", "Errors caught: mutashabihat (forms another ayah)", g.mutashabihat.recall, "%", hitsOf(g.mutashabihat)],
      ["cleanPer5", "False warnings per 5 clean minutes", g.cleanPer5, ""],
      ["located", "Correctly located time", g.located, "%"],
      ["within5s", "Located within 5 seconds", g.within5s, "%"],
      ["wrongPositions", "Wrong position events", g.wrongPositions, ""],
      ["nonErrorWarnings", "Warnings on repeats/restarts/self-corrections", g.nonErrorWarnings, ""],
      ["suspectedDelay.p50", "Delay to suspected, p50", g.suspectedDelay.p50, "s", undefined, delayText(g.suspectedDelay)],
      ["suspectedDelay.p90", "Delay to suspected, p90", g.suspectedDelay.p90, "s", undefined, delayText(g.suspectedDelay)],
      ["confirmedDelay.p50", "Delay to confirmed, p50", g.confirmedDelay.p50, "s", undefined, delayText(g.confirmedDelay)],
      ["confirmedDelay.p90", "Delay to confirmed, p90", g.confirmedDelay.p90, "s", undefined, delayText(g.confirmedDelay)],
      ...(["uncertain", "twinAmbiguous", "ambiguous"] as const).flatMap((c): Row[] => {
        const label = { uncertain: "checker uncertain", twinAmbiguous: "tracker twin ambiguity", ambiguous: "tracker other ambiguity" }[c];
        return [[`abstain.${c}.shareOfErrors`, `Missed errors from ${label}`, g.abstentionShare[c], "%", undefined, `share of ${g.errors} labelled errors`],
          [`abstain.${c}.perCleanSession`, `${label[0].toUpperCase()}${label.slice(1)} per clean session`, g.perCleanSession[c], "", undefined, `${g.cleanSessions} clean sessions`]];
      }),
    ];
    for (const [id, name, value, unit, display, extra] of measures)
      metrics.push({ id: `${prefix}.${id}`, group: `Sessions: ${g.key}`, name, value, unit, ...(display ? { display } : {}),
        status: value === null ? "unmeasurable" : "info", source: basename(values.sessions), note: note(id, extra) });
  }
}

// ---- Row.
const sha = values.sha ?? execSync("git rev-parse --short HEAD").toString().trim();
const now = new Date().toISOString();
const row: ScoreRow = { sha, date: now.slice(0, 10), recordedAt: now, listener: values.listener!, note: values.note, metrics };
// Keep existing attribution when regenerating a run without a new module tag.
row.modules = modules as ScoreRow["modules"] ?? board.rows.find((r) => r.sha === row.sha && r.listener === row.listener)?.modules;
if (!values["dump-only"]) {
  board.rows = upsertRow(board.rows, row);
  save();
}
if (rows.length) writeClipDump();

/** Per-clip outcomes for the dashboard's mistake review; private (speaker IDs, audio paths), so under .scratch. */
function writeClipDump() {
  const dir = resolve(values["clip-dump"]!);
  mkdirSync(dir, { recursive: true });
  const leadById = new Map(leadRows.map((r) => [r.l.id, r]));
  const clips: DumpClip[] = rows.map((r) => {
    const audio = `help/${basename(r.l.wav)}`;
    const words = clipWords(index, asset, r.clip);
    const arms: DumpClip["arms"] = {
      free: dumpArm(index, r.clip, words, audio, { score: r.free, declared: r.declared }),
      declared: dumpArm(index, r.clip, words, audio, { score: r.declared }),
    };
    if (r.oracle) arms.oracle = dumpArm(index, r.clip, words, audio, { score: r.oracle });
    const lr = leadById.get(r.l.id), raw = leadRaw.get(r.l.id);
    if (lr) {
      const ctx = raw ? { segments: raw.segments, segmentAudio: raw.segmentAudio, leadAyahs: raw.leadAyahs, followAyahs: raw.followAyahs } : {};
      arms.leadFree = dumpArm(index, r.clip, words, audio, { score: lr.free, declared: lr.declared, raw: raw?.free, ...ctx });
      arms.leadDeclared = dumpArm(index, r.clip, words, audio, { score: lr.declared, raw: raw?.declared, ...ctx });
    }
    const l = r.l as Label & { device?: string; level?: string };
    return {
      id: l.id, speaker: l.speaker, device: l.device, level: l.level, kind: l.kind, ayahs: l.ayahs, audio, durationS: r.audioS,
      sourceLabel: { ...l }, labels: r.clip.events, repeats: r.clip.nonErrors.map((n) => `${n.surah}:${n.ayah}:${n.word}`), words, arms,
    };
  });
  const file = `${row.sha}-${row.listener}.json`;
  const dump: ClipDump = { version: CLIP_DUMP_VERSION, sha: row.sha, listener: row.listener, recordedAt: now, clips,
    replay: { helpEvidenceDir: evidenceDir!, helpEvidenceLayout: values.evidence ? "flat" : "wav", ...(leadDir ? { leadEvidenceDir: leadDir } : {}),
      referenceAsset: resolve("assets/ref-v2.json"), phraseAsset: resolve("assets/phrases-v1.json"),
      settings: { ...REPLAY_DEFAULTS } },
  };
  writeFileSync(join(dir, file), JSON.stringify(dump));
  const indexPath = join(dir, "index.json");
  const runs: { sha: string; listener: string; recordedAt: string; file: string }[] = existsSync(indexPath) ? JSON.parse(readFileSync(indexPath, "utf8")) : [];
  writeFileSync(indexPath, JSON.stringify([...runs.filter((x) => x.file !== file), { sha: row.sha, listener: row.listener, recordedAt: now, file }], null, 1) + "\n");
  console.log(`clip dump: ${join(dir, file)}`);
}
if (!values["dump-only"]) console.log(readFileSync(`${resolve(values.out!)}.md`, "utf8").split("\n## All metrics")[0]);
