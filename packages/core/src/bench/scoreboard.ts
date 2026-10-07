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
  /** `teacher`, or a student run name. */
  listener: string;
  /** One line: what changed and why. */
  note?: string;
  metrics: Metric[];
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

const fmt = (v: number, unit: Metric["unit"]) => (unit === "%" ? `${(100 * v).toFixed(1)}%` : unit === "s" ? `${v.toFixed(2)} s` : `${+v.toFixed(3)}`);
const value = (m: Metric) => m.display ?? (m.value === null ? "—" : fmt(m.value, m.unit));
const bar = (m: Metric) => (m.bar ? `${m.bar.op === ">=" ? "≥" : "≤"} ${fmt(m.bar.target, m.unit)}` : "");
function delta(m: Metric, base: Metric | undefined): string {
  if (!base || base === m || m.value === null || base.value === null) return "";
  const d = m.value - base.value;
  const s = m.unit === "%" ? `${(100 * d).toFixed(1)} pp` : m.unit === "s" ? `${d.toFixed(2)} s` : `${+d.toFixed(3)}`;
  return d > 0 ? `+${s}` : d < 0 ? s : "0";
}

/**
 * Latest snapshot per listener (delta vs that listener's first row, the baseline), then the history of headline
 * metrics across every row.
 */
export function renderScoreboard(rows: ScoreRow[], headline: string[] = []): string {
  const out: string[] = [];
  const listeners = [...new Set(rows.map((r) => r.listener))];
  for (const l of listeners) {
    const mine = rows.filter((r) => r.listener === l);
    const base = mine[0];
    const last = mine.at(-1)!;
    out.push(`## Latest: ${l} @ ${last.sha} (${last.date})`, "", `Delta is against the baseline row ${base.sha} (${base.date}).`, "");
    out.push("| Area | Metric | Value | Bar | Status | Δ baseline | Note |", "|---|---|---:|---|---|---:|---|");
    for (const m of last.metrics) {
      const b = base.metrics.find((x) => x.id === m.id);
      out.push(`| ${m.group} | ${m.name} | ${value(m)} | ${bar(m)} | ${m.status} | ${delta(m, b === m ? undefined : b)} | ${m.note ?? ""} |`);
    }
    out.push("");
  }
  const ids = headline.length ? headline : rows[0]?.metrics.filter((m) => m.bar).map((m) => m.id) ?? [];
  const names = ids.map((id) => rows.flatMap((r) => r.metrics).find((m) => m.id === id)).map((m, i) => (m ? `${m.group}: ${m.name}` : ids[i]));
  out.push("## History", "", `| Date | Commit | Listener | ${names.join(" | ")} | Note |`, `|---|---|---|${ids.map(() => "---:").join("|")}|---|`);
  for (const r of rows) {
    out.push(`| ${r.date} | ${r.sha} | ${r.listener} | ${ids.map((id) => { const m = r.metrics.find((x) => x.id === id); return m ? value(m) : "—"; }).join(" | ")} | ${r.note ?? ""} |`);
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
