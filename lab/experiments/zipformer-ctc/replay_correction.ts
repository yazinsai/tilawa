// Replay correction-controller traces (`correction_eval.py run --trace`)
// through the current CorrectionController, offline and without ONNX. Issues
// are dismissed on sight like harness.ts. The trace was recorded with a
// controller that never flags, so after the first flag the live session would
// diverge (it re-locks at the resume point); confirm final rules with a live run.
//
//   tsx --tsconfig ../web/frontend/tsconfig.json experiments/zipformer-ctc/replay_correction.ts \
//     --in /tmp/correction_eval/traces/shipped --out /tmp/correction_eval/replay/x/shipped \
//     [--thresholds '{"gopFlag": -1e309, "settle": false}']
//
// --grid <file.json> replays many rule sets in one pass: the file is
// [{"name": "...", "thresholds": {...}}, ...] and each writes <out>/<name>/.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

import { CorrectionController, DEFAULT_CORRECTION_THRESHOLDS, type CorrectionIssue, type WordVerdict } from "@tilawa/core";

const args = process.argv.slice(2);
const opt = (k: string): string | undefined => {
  const i = args.indexOf(k);
  return i >= 0 ? args[i + 1] : undefined;
};
const inDir = opt("--in");
const outDir = opt("--out");
if (!inDir || !outDir) throw new Error("--in and --out required");
const gridFile = opt("--grid");
const grid = gridFile
  ? (JSON.parse(readFileSync(gridFile, "utf8")) as Array<{ name: string; thresholds: Record<string, number | boolean> }>)
  : [{ name: "", thresholds: JSON.parse(opt("--thresholds") ?? "{}") as Record<string, number | boolean> }];

const STATES = ["ok", "unsure", "wrong", "skipped", "pending"] as const;
const num = (x: number | null): number => (x === null ? NaN : x);
const opt3 = (x: number | null): number | undefined => (x === null ? undefined : x);
function unpack(a: Array<number | null>): WordVerdict {
  const v: WordVerdict = {
    surah: a[0]!, ayah: a[1]!, word: a[2]!, wordIndex: a[3]!, state: STATES[a[4]!]!,
    distance: num(a[5]!), heardRatio: num(a[6]!), margin: num(a[7]!), vowelErrors: a[8] ?? 0, vowelMargin: num(a[9]!),
  };
  const extra: Record<string, number | undefined> = { gop: opt3(a[10]!), gopTwice: opt3(a[11]!), gopNone: opt3(a[12]!), repGain: opt3(a[13]!), pairGop: opt3(a[14]!) };
  for (const [k, x] of Object.entries(extra)) if (x !== undefined) (v as unknown as Record<string, number>)[k] = x;
  return v;
}

interface Op {
  op: "observe" | "clear" | "raise" | "settle";
  t: number;
  frame?: number;
  cursor?: { surah: number; ayah: number; word: number };
  v?: Array<Array<number | null>>;
  j?: number[][];
  issue?: CorrectionIssue;
}
interface Clip { id: string; duration_s: number; split?: string; trace?: Op[]; error?: string }

type Ctl = CorrectionController & { settle?: (v: WordVerdict[], c: { surah: number; ayah: number; word: number }) => boolean };

function replay(clip: Clip, overrides: Record<string, number | boolean>, ops: Array<{ op: Op; v?: WordVerdict[] }>): string {
  const ctl = new CorrectionController() as Ctl;
  ctl.thresholds = { ...DEFAULT_CORRECTION_THRESHOLDS, ...overrides } as typeof ctl.thresholds;
  ctl.setMode("correction");
  const issues: Array<Record<string, unknown>> = [];
  const notes: Array<Record<string, unknown>> = [];
  let cursor = { surah: 0, ayah: 0, word: 0 };
  for (const { op, v } of ops) {
    let raised = false;
    if (op.op === "observe") {
      cursor = op.cursor ?? cursor;
      raised = ctl.observe(v ?? [], cursor, op.frame ?? 0);
    } else if (op.op === "settle") {
      if (ctl.settle) raised = ctl.settle(v ?? [], cursor);
    } else if (op.op === "clear") {
      ctl.clearEvidence();
    } else if (op.op === "raise" && op.issue) {
      raised = ctl.raise(op.issue, cursor);
    }
    for (const n of ctl.takeNotes()) notes.push({ kind: n.kind, surah: n.surah, ayah: n.ayah, word: n.word, wordIndex: n.wordIndex, atSeconds: op.t });
    if (raised && ctl.state.issue) {
      const { kind, surah, ayah, word, wordIndex, words } = ctl.state.issue;
      issues.push({ kind, surah, ayah, word, wordIndex, ...(words ? { words } : {}), atSeconds: op.t });
      ctl.act("dismiss");
    }
  }
  return JSON.stringify({ id: clip.id, duration_s: clip.duration_s, split: clip.split, issues, notes });
}

const outputs = grid.map(() => new Map<string, string[]>());
for (const file of readdirSync(inDir).filter((f) => f.endsWith(".jsonl"))) {
  outputs.forEach((m) => m.set(file, []));
  for (const line of readFileSync(path.join(inDir, file), "utf8").split("\n")) {
    if (!line.trim()) continue;
    const clip = JSON.parse(line) as Clip;
    if (clip.error) {
      const row = JSON.stringify({ id: clip.id, duration_s: clip.duration_s, split: clip.split, error: clip.error, issues: [] });
      outputs.forEach((m) => m.get(file)!.push(row));
      continue;
    }
    const ops = (clip.trace ?? []).map((op) => ({ op, v: op.v ? op.v.map(unpack) : undefined }));
    grid.forEach((g, i) => outputs[i]!.get(file)!.push(replay(clip, g.thresholds, ops)));
  }
}
grid.forEach((g, i) => {
  const dir = path.join(outDir, g.name);
  mkdirSync(dir, { recursive: true });
  for (const [file, rows] of outputs[i]!) writeFileSync(path.join(dir, file), rows.join("\n") + "\n");
});
