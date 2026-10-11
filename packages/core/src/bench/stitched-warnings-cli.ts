/**
 * Free-recitation checker false warnings on the ticket 13 stitched sessions, per listener, with session-end
 * finalisation on and off, split by where in the session they land.
 *
 *   pnpm stitched-warnings --arm teacher:../../.scratch/jumps/evidence:../../.scratch/help/wav \
 *     --arm spliced-clean-int8:../../.scratch/jumps/evidence-int8:../../.scratch/sdk-help/spliced-clean-int8-node-lw
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { execSync } from "node:child_process";
import { loadReferenceIndex } from "../reference.js";
import type { PhraseAsset, ReferenceAsset } from "../reference.js";
import { scoreSession, stitchSession } from "./jumps.js";
import type { ClipEvidence, SessionScore } from "./jumps.js";

const { values } = parseArgs({ options: {
  arm: { type: "string", multiple: true, default: [] },
  manifest: { type: "string", default: "../../docs/benchmarks/jumps-v1-manifest.json" },
  report: { type: "string", default: "../../docs/benchmarks/stitched-warnings-v1" },
  "tail-s": { type: "string", default: "20" },
} });
const TAIL_MS = Number(values["tail-s"]) * 1000;
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const phrases = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8")) as PhraseAsset;

interface ManifestSegment { clip: string; ayahs: string[]; gapBeforeMs: number; gapFill: "silence" | "room" }
interface ManifestSession { id: string; kind: string; source: "tlog" | "help"; segments: ManifestSegment[] }
const manifest = JSON.parse(readFileSync(resolve(values.manifest!), "utf8")) as { version: string; sessions: ManifestSession[] };

interface Arm { listener: string; posteriors: boolean; finalise: boolean }
interface Tally { sessions: number; minutes: number; warnings: number; atEnd: number; tail: number; mid: number; missing: number }

function score(listener: string, tlogDir: string, helpDir: string, posteriors: boolean, finalise: boolean) {
  const t: Tally = { sessions: 0, minutes: 0, warnings: 0, atEnd: 0, tail: 0, mid: 0, missing: 0 };
  const detail: { id: string; audioS: number; warnings: SessionScore["checkerWarningDetail"] }[] = [];
  for (const s of manifest.sessions) {
    const paths = s.segments.map((g) => s.source === "help" ? resolve(helpDir, `${g.clip}.evidence.json`) : resolve(tlogDir, `${g.clip.replace(":", "_")}.evidence.json`));
    if (!paths.every(existsSync)) { t.missing++; continue; }
    const parts = s.segments.map((g, i) => ({ source: g.clip, ayahs: g.ayahs, gapBeforeMs: g.gapBeforeMs, gapFill: g.gapFill, evidence: JSON.parse(readFileSync(paths[i], "utf8")) as ClipEvidence }));
    const r = scoreSession(index, stitchSession(index, parts), { phrases, posteriors, checker: true, policy: { finaliseFree: finalise } });
    const endMs = r.audioS * 1000;
    t.sessions++;
    t.minutes += r.audioS / 60;
    for (const w of r.checkerWarningDetail) {
      t.warnings++;
      if (w.atEnd) t.atEnd++;
      else if (w.timeMs >= endMs - TAIL_MS) t.tail++;
      else t.mid++;
    }
    detail.push({ id: s.id, audioS: +r.audioS.toFixed(1), warnings: r.checkerWarningDetail });
  }
  console.log(listener, { posteriors, finalise }, t);
  return { arm: { listener, posteriors, finalise } satisfies Arm, tally: t, detail };
}

const results = values.arm!.flatMap((spec) => {
  const [listener, tlogDir, helpDir] = spec.split(":");
  return [
    ...(listener === "teacher" ? [score(listener, tlogDir, helpDir, false, true)] : []),
    score(listener, tlogDir, helpDir, true, true),
    score(listener, tlogDir, helpDir, true, false),
  ];
});

const commit = execSync("git rev-parse --short HEAD").toString().trim();
writeFileSync(`${resolve(values.report!)}.json`, JSON.stringify({ version: "stitched-warnings-v1", manifest: manifest.version, commit, tailS: TAIL_MS / 1000, results }, null, 1) + "\n");
const per5 = (n: number, t: Tally) => (t.minutes ? ((n * 5) / t.minutes).toFixed(2) : "—");
const tailMin = (t: Tally) => (t.sessions * TAIL_MS) / 60000;
const lines = [
  "| Listener | Posteriors | Finalise | Sessions / min | Warnings (per 5 min) | Mid-session (per 5 min of mid) | Last 20 s, live | Confirmed by session end | End region per 5 min |",
  "|---|---|---|---:|---:|---:|---:|---:|---:|",
  ...results.map(({ arm, tally: t }) => `| ${arm.listener} | ${arm.posteriors ? "on" : "off"} | ${arm.finalise ? "on" : "off"} | ${t.sessions} / ${t.minutes.toFixed(0)} | ${t.warnings} (${per5(t.warnings, t)}) | ${t.mid} (${((t.mid * 5) / (t.minutes - tailMin(t))).toFixed(2)}) | ${t.tail} | ${t.atEnd} | ${(((t.tail + t.atEnd) * 5) / tailMin(t)).toFixed(2)} |`),
];
console.log(lines.join("\n"));
