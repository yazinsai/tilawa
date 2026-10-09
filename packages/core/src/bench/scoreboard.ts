/** The tracked engine scoreboard: one row of metrics per (commit, listener), each against its PLAN.md / spec bar. */
import type { WordRef } from "../events.js";
import type { ReferenceAsset, ReferenceIndex } from "../reference.js";
import { syntheticFixture, withPosteriors } from "../fixtures/index.js";
import type { Edit } from "../fixtures/index.js";
import type { AcceptanceClip, LabelEvent } from "./acceptance.js";

export type Status = "pass" | "fail" | "info" | "unmeasurable" | "pending";

export interface Metric {
  id: string;
  group: string;
  name: string;
  value: number | null;
  /** `%` renders a 0–1 ratio as a percentage and deltas in percentage points. */
  unit?: "%" | "s" | "";
  /** Backing report under docs/benchmarks. */
  source?: string;
  /** Shown instead of the bare value, e.g. a count behind a ratio. */
  display?: string;
  bar?: { op: ">=" | "<="; target: number };
  status: Status;
  note?: string;
}

export interface ScoreRow {
  sha: string;
  date: string;
  /** ISO timestamp of when the row was recorded; absent on early rows. */
  recordedAt?: string;
  /** `teacher`, or a student run name. */
  listener: string;
  /** One line: what changed and why. */
  note?: string;
  /** Engine modules changed by this run; omitted for older, untagged runs. */
  modules?: ("listener" | "tracker" | "checker")[];
  metrics: Metric[];
}

/** `flipOnly`: the value is timing noise, so only crossing the bar counts as a move. */
export interface Gate { id: string; label: string; area: "Product" | "Listener" | "Tracker" | "Checker"; flipOnly?: boolean }
export interface Board { version: number; candidate?: string; gates?: Gate[]; blocked?: Gate[]; rows: ScoreRow[] }

/** The v1 release gates: every one must pass on the candidate listener. Everything else on a row is diagnostic. */
export const GATES: Gate[] = [
  { id: "e2e.leadin.recall", label: "Free recitation (1–2 recited ayahs before and after): mistakes caught", area: "Product" },
  { id: "e2e.leadin.precision", label: "Free recitation (1–2 recited ayahs before and after): warnings correct", area: "Product" },
  { id: "checker.oracleLock.recall", label: "Memorization session: mistakes caught", area: "Product" },
  { id: "checker.oracleLock.precision", label: "Memorization session: warnings correct", area: "Product" },
  { id: "e2e.cleanPer5", label: "False warnings per 5 min (short clips)", area: "Product" },
  { id: "e2e.stitched.falseWords", label: "False warnings per 5 min (stitched sessions)", area: "Product" },
  { id: "e2e.repeatFp", label: "False warnings on repeats", area: "Product" },
  { id: "e2e.leadin.leadWarnings", label: "False warnings per 5 min (lead-in recitation)", area: "Product" },
  { id: "e2e.leadin.followWarnings", label: "False warnings per 5 min (lead-out recitation)", area: "Product" },
  { id: "listener.goldPer", label: "Listener gold PER", area: "Listener" },
  { id: "tracker.clean.within5s", label: "Tracker: correct ayah within 5 s", area: "Tracker" },
  { id: "tracker.clean.falseJumps", label: "Tracker: false jumps", area: "Tracker" },
  { id: "e2e.perfRtf", label: "Real-time factor", area: "Product", flipOnly: true },
];
/** Release requirements with no data yet; they block release until measured. */
export const BLOCKED: Gate[] = [
  { id: "e2e.pron.short_vowel", label: "Pronunciation: short vowels", area: "Listener" },
  { id: "e2e.pron.phase2", label: "Pronunciation: confusable pairs, shadda", area: "Listener" },
  { id: "e2e.latency", label: "Word highlight / warning latency on device", area: "Product" },
  { id: "checker.selfCorrection", label: "Self-correction and long-session false warnings", area: "Checker" },
  { id: "tracker.scripted", label: "Scripted-error recordings", area: "Tracker" },
];
export const DEFAULT_CANDIDATE = "spliced-clean-int8";

/** Shortfall from the bar as a fraction of the bar (0 when passing), so gates with different units rank together. */
export function gap(m: Metric | undefined): number {
  if (!m?.bar || m.value === null) return Infinity;
  const { op, target } = m.bar;
  const short = op === ">=" ? target - m.value : m.value - target;
  return short <= 0 ? 0 : short / (target || 1);
}

const counts = (m: Metric | undefined) => { const x = m?.display?.match(/\((\d+)\/(\d+)\)/); return x ? { hit: +x[1], of: +x[2] } : null; };
const better = (m: Metric, d: number) => (m.bar?.op === "<=" ? d < 0 : d > 0);

export interface Move { id: string; label: string; from: Metric; to: Metric; better: boolean; small: boolean }
/**
 * Gate metrics that changed between two runs. `small` is a move of at most two clips (or under 3% relative for
 * metrics without counts): same eval set, so real, but too few clips to generalise.
 */
export function moves(from: ScoreRow, to: ScoreRow, gates: Gate[] = GATES): Move[] {
  const out: Move[] = [];
  for (const g of gates) {
    const a = from.metrics.find((m) => m.id === g.id), b = to.metrics.find((m) => m.id === g.id);
    if (!a || !b || a.value === null || b.value === null || Math.abs(b.value - a.value) < 1e-9) continue;
    if (g.flipOnly && a.status === b.status) continue;
    const ca = counts(a), cb = counts(b);
    const small = ca && cb ? Math.abs(cb.hit - ca.hit) <= 2 && Math.abs(cb.of - cb.hit - (ca.of - ca.hit)) <= 2 : Math.abs(b.value - a.value) / Math.max(Math.abs(a.value), 1e-9) < 0.03;
    out.push({ id: g.id, label: g.label, from: a, to: b, better: better(b, b.value - a.value), small });
  }
  return out;
}

export function verdict(ms: Move[]): "better" | "worse" | "mixed" | "noise" | "unchanged" {
  if (!ms.length) return "unchanged";
  const big = ms.filter((m) => !m.small);
  if (!big.length) return "noise";
  const up = big.some((m) => m.better), down = big.some((m) => !m.better);
  return up && down ? "mixed" : up ? "better" : "worse";
}

/** The run a row is judged against: the previous run of its listener, else the candidate's latest earlier run. */
export function parentOf(rows: ScoreRow[], i: number, candidate: string): ScoreRow | undefined {
  const r = rows[i];
  for (let j = i - 1; j >= 0; j--) if (rows[j].listener === r.listener) return rows[j];
  for (let j = i - 1; j >= 0; j--) if (rows[j].listener === candidate) return rows[j];
  return undefined;
}

/** The engine stage losing the most planted mistakes (E0 attribution), the biggest lever on recall. */
export function biggestLever(row: ScoreRow): { stage: string; lost: number; of: number } | null {
  const e0 = (["evidence", "tracker", "checker"] as const).map((k) => ({ stage: k === "evidence" ? "listener" : k, lost: row.metrics.find((m) => m.id === `checker.e0.${k}`)?.value ?? null }));
  if (e0.some((x) => x.lost === null)) return null;
  const of = e0.reduce((n, x) => n + x.lost!, 0);
  const top = e0.reduce((a, b) => (b.lost! > a.lost! ? b : a));
  return of ? { stage: top.stage, lost: top.lost!, of } : null;
}

/** Replaces the row with the same commit and listener, else appends. */
export function upsertRow(rows: ScoreRow[], row: ScoreRow): ScoreRow[] {
  const at = rows.findIndex((r) => r.sha === row.sha && r.listener === row.listener);
  return at < 0 ? [...rows, row] : rows.map((r, i) => (i === at ? row : r));
}

/** A metric with its status derived from its bar, unless the caller already marked it unmeasurable or pending. */
export function scored(m: Omit<Metric, "status"> & { status?: Status }): Metric {
  if (m.status) return m as Metric;
  if (m.value === null || !m.bar) return { ...m, status: m.value === null ? "unmeasurable" : "info" };
  const ok = m.bar.op === ">=" ? m.value >= m.bar.target : m.value <= m.bar.target;
  return { ...m, status: ok ? "pass" : "fail" };
}

interface GoldPer { canon: number; raw: number; clips: number }
interface GoldScore { clips: number; overall: GoldPer; byDevice: Record<string, GoldPer>; groups: Record<string, { per: number; phonemes: number }> }
/** docs/benchmarks/gold-per-v1-scores.json (`tools/gold_set.py score`) and listener-target-v1.json (`calibrate`). */
export type GoldScores = Record<string, { gold: GoldScore; goldHeldOut?: GoldScore }>;
export interface ListenerTarget { bar: number; bindingMetric: string; teacherGoldPer: number }

/** Listener PER on the human-verified gold set: canon vs spliced reference, gated by the calibrated bar. */
export function goldPerMetrics(scores: GoldScores, target: ListenerTarget, listener: string): Metric[] {
  const group = "Listener", source = "listener-target-v1.md";
  const g = scores[listener]?.gold, h = scores[listener]?.goldHeldOut;
  const bar = { op: "<=", target: target.bar } as const;
  const name = "Gold PER (canon vs spliced, human-verified exact)";
  if (!g) return [scored({ id: "listener.goldPer", group, name, value: null, unit: "%", bar, source, status: "pending", note: "not decoded on the gold set; modal/student_decode_cpu.py::gold, then tools/gold_set.py score" })];
  const pct = (x: number) => `${(100 * x).toFixed(2)}%`;
  return [
    scored({ id: "listener.goldPer", group, name, value: g.overall.canon, unit: "%", bar, source, display: `${pct(g.overall.canon)} (${g.clips} clips)`,
      note: `bar: gold PER where fitted ${target.bindingMetric} falls 2 pts below the teacher (${pct(target.teacherGoldPer)})` }),
    ...Object.entries(g.byDevice).map(([d, x]) => scored({ id: `listener.goldPer.${d}`, group, name: `Gold PER: ${d}`, value: x.canon, unit: "%", source, display: `${pct(x.canon)} (${x.clips} clips)` })),
    ...Object.entries(g.groups).map(([k, x]) => scored({ id: `listener.goldPer.${k}`, group, name: `Gold PER: ${k.replace("_", " ")}`, value: x.per, unit: "%", source, display: `${pct(x.per)} (${x.phonemes} ref phonemes)` })),
    scored({ id: "listener.goldPerRaw", group, name: "Gold raw PER vs spliced reference", value: g.overall.raw, unit: "%", source }),
    ...(h ? [scored({ id: "listener.goldPerHeldOut", group, name: "Gold PER: never in tlog train", value: h.overall.canon, unit: "%", source, display: `${pct(h.overall.canon)} (${h.clips} clips)`,
      note: "help clips plus tlog test/val gold clips; fair for students trained before the gold exclusion" })] : []),
  ];
}

const fmt = (v: number, unit: Metric["unit"]) => (unit === "%" ? `${(100 * v).toFixed(1)}%` : unit === "s" ? `${v.toFixed(2)} s` : `${+v.toFixed(3)}`);
const value = (m: Metric) => m.display ?? (m.value === null ? "—" : fmt(m.value, m.unit));
const bar = (m: Metric) => (m.bar ? `${m.bar.op === ">=" ? "≥" : "≤"} ${fmt(m.bar.target, m.unit)}` : "");
function delta(m: Metric, base: Metric | undefined): string {
  if (!base || base === m || m.value === null || base.value === null) return "";
  const d = m.value - base.value;
  const s = m.unit === "%" ? `${(100 * d).toFixed(1)} pp` : m.unit === "s" ? `${d.toFixed(2)} s` : `${+d.toFixed(3)}`;
  return d > 0 ? `+${s}` : d < 0 ? s : "0";
}

const VERDICT = { better: "▲ better", worse: "▼ worse", mixed: "◆ mixed", noise: "· noise", unchanged: "= unchanged" };
const moveText = (m: Move) => `${m.better ? "▲" : "▼"}${m.small ? "~" : ""} ${m.label} ${value(m.from)} → ${value(m.to)}`;

/**
 * Release readiness of the candidate listener (gates ranked by shortfall), the experiment log (each run against its
 * parent, gate moves only), then the candidate's full metrics against its previous run.
 */
export function renderScoreboard(rows: ScoreRow[], candidate = DEFAULT_CANDIDATE, gates = GATES, blocked = BLOCKED): string {
  const out: string[] = [];
  const mine = rows.filter((r) => r.listener === candidate);
  const last = mine.at(-1), prev = mine.at(-2);
  if (last) {
    const ranked = gates.map((g) => ({ g, m: last.metrics.find((x) => x.id === g.id) })).sort((a, b) => gap(b.m) - gap(a.m));
    const failing = ranked.filter((x) => x.m?.status === "fail").length, open = ranked.filter((x) => !x.m || x.m.value === null).length;
    const unmeasured = blocked.filter((b) => !last.metrics.some((m) => (m.id === b.id || m.id.startsWith(`${b.id}.`)) && m.value !== null));
    const ready = failing === 0 && open === 0 && unmeasured.length === 0;
    out.push(`## Release readiness: ${candidate} @ ${last.sha} (${last.date})`, "",
      `**${ready ? "Ready" : "Not ready"}**: ${gates.length - failing - open}/${gates.length} gates pass, ${failing} fail, ${open} not measured on this listener, ${unmeasured.length} requirements not measurable yet. Worst shortfall first.`, "",
      "| Gate | Now | Bar | Shortfall | Δ previous run |", "|---|---:|---|---:|---:|");
    for (const { g, m } of ranked) {
      const s = gap(m);
      out.push(`| ${g.label} | ${m ? value(m) : "—"} | ${m ? bar(m) : ""} | ${s === Infinity ? "not measured" : s === 0 ? "pass" : `${(100 * s).toFixed(0)}% short`} | ${m ? delta(m, prev?.metrics.find((x) => x.id === g.id)) : ""} |`);
    }
    if (unmeasured.length) out.push("", `Not measurable yet (blocks release): ${unmeasured.map((b) => b.label).join("; ")}.`);
    out.push("");
    const lever = biggestLever(last);
    if (lever) out.push(`**Biggest lever on recall: ${lever.stage}**, which loses ${lever.lost} of the ${lever.of} planted mistakes missed in free recitation (E0 attribution).`, "");
  }
  out.push("## Experiments", "", "Newest first. Each run is judged against its parent: the previous run of the same listener, else the candidate. Only gates that moved are listed; `~` marks a move of two clips or fewer.", "",
    "| Date | Commit | Listener | vs | Verdict | Gate moves | Note |", "|---|---|---|---|---|---|---|");
  for (let i = rows.length - 1; i >= 0; i--) {
    const r = rows[i], p = parentOf(rows, i, candidate);
    const ms = p ? moves(p, r, gates) : [];
    out.push(`| ${r.date} | ${r.sha} | ${r.listener} | ${p ? `${p.listener === r.listener ? "" : `${p.listener} `}${p.sha}` : "first run"} | ${p ? VERDICT[verdict(ms)] : ""} | ${ms.map(moveText).join("<br>")} | ${r.note ?? ""} |`);
  }
  if (last) {
    out.push("", `## All metrics: ${candidate}`, "", "| Area | Metric | Value | Bar | Status | Δ previous run | Note |", "|---|---|---:|---|---|---:|---|");
    for (const m of last.metrics) out.push(`| ${m.group} | ${m.name} | ${value(m)} | ${bar(m)} | ${m.status} | ${delta(m, prev?.metrics.find((x) => x.id === m.id))} | ${m.note ?? ""} |`);
  }
  return out.join("\n") + "\n";
}

export interface OracleLabel {
  id: string;
  ayahs: string[];
  events: (LabelEvent & { observed?: string; words?: number })[];
  non_errors: { kind: string; location: WordRef }[];
}

const normalize = (s: string) =>
  s.normalize("NFC").replace(/[\u064B-\u065F\u0670\u06D6-\u06ED\u0640]/g, "").replace(/[أإآٱ]/g, "ا").replace(/ى/g, "ي").replace(/ة/g, "ه");

/**
 * Evidence synthesised from the label alone (Decision 7): the prompted passage with each planted memorization error and
 * repeat applied, decoded perfectly. Null when a planted word can't be phonemized (its written form is in no ayah).
 */
export function oracleClip(index: ReferenceIndex, asset: ReferenceAsset, label: OracleLabel): AcceptanceClip | null {
  const ayahOf = (a: string) => index.words.filter((w) => `${w.surah}:${w.ayah}` === a);
  const span = label.ayahs.flatMap(ayahOf);
  const lo = span[0].start;
  const wordIdx = (r: WordRef) => span.findIndex((w) => w.surah === r.surah && w.ayah === r.ayah && w.word === r.word);
  const edits: Edit[] = [];
  for (const e of label.events) {
    const i = wordIdx(e.location);
    if (i < 0) continue;
    if (e.errorClass === "omitted_word" || e.errorClass === "skipped_passage") {
      const last = span[Math.min(span.length, i + (e.words ?? 1)) - 1];
      edits.push({ kind: "del", at: span[i].start - lo, count: last.end - span[i].start });
    } else if (e.errorClass === "wrong_word") {
      const said = donor(asset, e.observed ?? "");
      if (said === null) return null;
      edits.push({ kind: "del", at: span[i].start - lo, count: span[i].end - span[i].start }, { kind: "ins", at: span[i].start - lo, text: said });
    }
  }
  for (const n of label.non_errors) {
    const i = wordIdx(n.location);
    if (n.kind === "repeat" && i >= 0) edits.push({ kind: "repeat", from: span[i].start - lo, to: span[i].end - lo });
  }
  const first = span[0];
  const last = span.at(-1)!;
  const e = withPosteriors(index, syntheticFixture(index, { id: label.id, from: first, to: last, edits }).evidence[0]);
  const p = e.posteriors!;
  return {
    id: label.id,
    ayahs: label.ayahs,
    events: label.events,
    nonErrors: label.non_errors.filter((n) => n.kind === "repeat").map((n) => n.location),
    evidence: { inventory: e.inventory, frameMs: e.frameMs, nFrames: p.nFrames, phonemes: e.phonemes, posteriors: { k: p.k, ids: Array.from(p.ids), probs: Array.from(p.probs) } },
  };
}

let donors: Map<string, string> | null = null;
/** The isolated phonemization of the first ayah word written as `text`. */
function donor(asset: ReferenceAsset, text: string): string | null {
  if (!donors) {
    donors = new Map();
    for (const a of asset.ayahs) a.w.forEach((w, i) => { const k = normalize(w); if (!donors!.has(k)) donors!.set(k, a.iso[i]); });
  }
  return donors.get(normalize(text)) ?? null;
}
