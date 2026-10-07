/** Stitched-session false-jump report (ticket 13). Run `tools/jumps_build.py` first. */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { execSync } from "node:child_process";
import { loadReferenceIndex } from "../reference.js";
import type { PhraseAsset, ReferenceAsset } from "../reference.js";
import { scoreSession, stitchSession } from "./jumps.js";
import type { ClipEvidence, SessionScore, StitchedSession } from "./jumps.js";

const { values } = parseArgs({ options: {
  dir: { type: "string", default: process.env.JUMPS_DIR ?? "../../.scratch/jumps" },
  help: { type: "string", default: process.env.HELP_DIR ?? "../../.scratch/help" },
  manifest: { type: "string", default: "../../docs/benchmarks/jumps-v1-manifest.json" },
  report: { type: "string", default: "../../docs/benchmarks/jumps-v1" },
  posteriors: { type: "boolean", default: false },
} });
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const phrases = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8")) as PhraseAsset;

interface ManifestSegment { clip: string; ayahs: string[]; duration_s: number; speaker: string; gapBeforeMs: number; gapFill: "silence" | "room" }
interface ManifestSession { id: string; kind: string; source: "tlog" | "help"; hardSegments: number; segments: ManifestSegment[] }
const manifest = JSON.parse(readFileSync(resolve(values.manifest!), "utf8")) as { version: string; sessions: ManifestSession[] };

const evidencePath = (s: ManifestSession, g: ManifestSegment) =>
  s.source === "help" ? resolve(values.help!, `wav/${g.clip}.evidence.json`) : resolve(values.dir!, `evidence/${g.clip.replace(":", "_")}.evidence.json`);

interface Row { id: string; kind: string; source: string; hardSegments: number; speakerChanges: number; stitched: StitchedSession; score: SessionScore }
const rows: Row[] = [];
for (const s of manifest.sessions) {
  const parts = s.segments.map((g) => ({
    source: g.clip, ayahs: g.ayahs, gapBeforeMs: g.gapBeforeMs, gapFill: g.gapFill,
    evidence: JSON.parse(readFileSync(evidencePath(s, g), "utf8")) as ClipEvidence,
  }));
  const stitched = stitchSession(index, parts);
  const score = scoreSession(index, stitched, { phrases, posteriors: values.posteriors, checker: true });
  const speakerChanges = s.segments.filter((g, i) => i > 0 && g.speaker !== s.segments[i - 1].speaker).length;
  rows.push({ id: s.id, kind: s.kind, source: s.source, hardSegments: s.hardSegments, speakerChanges, stitched, score });
  if (rows.length % 20 === 0) console.log(`${rows.length}/${manifest.sessions.length} sessions`);
}

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const quantile = (xs: number[], q: number) => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
};
function summarize(rs: Row[]) {
  const seconds = sum(rs.map((r) => r.score.audioS));
  const falseJumps = sum(rs.map((r) => r.score.falseJumps));
  const jumps = sum(rs.map((r) => r.score.jumps));
  const planted = sum(rs.map((r) => r.score.planted));
  const latencies = rs.flatMap((r) => r.score.followLatencyMs.filter((x): x is number => x !== null));
  const seams = sum(rs.map((r) => r.stitched.segments.length - 1));
  return {
    sessions: rs.length, minutes: seconds / 60, segments: sum(rs.map((r) => r.stitched.segments.length)),
    seams, speakerChanges: sum(rs.map((r) => r.speakerChanges)), hardSegments: sum(rs.map((r) => r.hardSegments)),
    jumps, falseJumps, sessionsWithFalseJump: rs.filter((r) => r.score.falseJumps > 0).length,
    falseJumpsPerSession: rs.length ? falseJumps / rs.length : null,
    falseJumpsPerJump: jumps ? falseJumps / jumps : null,
    falseJumpsPer5Min: seconds ? (falseJumps * 300) / seconds : null,
    planted, followed: sum(rs.map((r) => r.score.followed)),
    followLatencyMedianMs: quantile(latencies, 0.5), followLatencyP90Ms: quantile(latencies, 0.9),
    followedWithin5s: latencies.filter((x) => x <= 5000).length,
    startWithin5s: rs.filter((r) => r.score.startWithin5s).length,
    checkerWarnings: sum(rs.map((r) => r.score.checkerWarnings ?? 0)),
    checkerWarningsPer5Min: seconds ? (sum(rs.map((r) => r.score.checkerWarnings ?? 0)) * 300) / seconds : null,
  };
}
const groups: [string, Row[]][] = [
  ["All", rows],
  ["Contiguous (tlog)", rows.filter((r) => r.kind === "contiguous")],
  ["Planted jumps (tlog)", rows.filter((r) => r.kind === "jump")],
  ["Hard runs (tlog)", rows.filter((r) => r.kind === "hard")],
  ["Help speakers", rows.filter((r) => r.kind === "help")],
  ["Any session with 3+ near-duplicate segments", rows.filter((r) => r.hardSegments >= 3)],
];
const output = {
  version: manifest.version,
  commit: execSync("git rev-parse --short HEAD").toString().trim(),
  reference: index.version,
  policy: `teacher evidence; tracker in memorization mode; 1 s chunks; ${values.posteriors ? "with" : "without"} posteriors; no passage prior`,
  groups: Object.fromEntries(groups.map(([k, rs]) => [k, summarize(rs)])),
  sessions: rows.map((r) => ({
    id: r.id, kind: r.kind, source: r.source, hardSegments: r.hardSegments, speakerChanges: r.speakerChanges,
    audioS: +r.score.audioS.toFixed(2), startWithin5s: r.score.startWithin5s, firstCorrectLockMs: r.score.firstCorrectLockMs,
    jumps: r.score.jumps, falseJumps: r.score.falseJumpDetail, planted: r.score.planted, followLatencyMs: r.score.followLatencyMs,
    timeline: r.stitched.segments,
  })),
};
writeFileSync(`${resolve(values.report!)}.json`, JSON.stringify(output, null, 1) + "\n");

const pct = (n: number | null) => (n === null ? "—" : `${(100 * n).toFixed(1)}%`);
const frac = (n: number, d: number) => `${n}/${d}${d ? ` (${((100 * n) / d).toFixed(1)}%)` : ""}`;
const sec = (ms: number | null) => (ms === null ? "—" : `${(ms / 1000).toFixed(1)} s`);
const G = groups.map(([k]) => [k, output.groups[k]] as const);
const row = (label: string, f: (g: ReturnType<typeof summarize>) => string, bar = "") => `| ${label} | ${G.map(([, g]) => f(g)).join(" | ")} | ${bar} |`;
const falseJumps = output.sessions.flatMap((s) => s.falseJumps.map((j) => ({ id: s.id, ...j })));
const lines = [
  `# Stitched-session jump test ${manifest.version} (ticket 13)`, "",
  `Report ${manifest.version}, commit ${output.commit}, reference ${index.version}. ${rows.length} sessions, ${output.groups.All.minutes.toFixed(0)} minutes, stitched from clean held-out clips (manifest \`jumps-v1-manifest.json\`, built by \`tools/jumps_build.py\`). Clips are joined with 0.3–1.5 s gaps of blank frames or the clip's own lead-in (room tone).`, "",
  "Teacher evidence is the offline Muaalem decode of each clip, stitched at the evidence level and replayed in 1 s chunks. It is not a streaming student, and no session audio was decoded as one recording.", "",
  "A jump is a non-local lock change: to an ayah more than 2 from the previous lock. It is false when the new ayah is more than 2 from every ayah recited in the last 4 s (or starting within 0.5 s), as in `acceptance-v3`. A planted jump is a seam whose next clip starts more than 2 ayahs from where the last one ended; it is followed by the first lock on the new run's ayahs before the next planted jump, timed from the new clip's first decoded phoneme. The start is the first lock on the opening run within 5 s of its first phoneme.", "",
  "## Results", "",
  `| Measure | ${G.map(([k]) => k).join(" | ")} | Bar |`, `|---|${G.map(() => "---:").join("|")}|---|`,
  row("Sessions / minutes", (g) => `${g.sessions} / ${g.minutes.toFixed(0)}`),
  row("Seams / speaker changes", (g) => `${g.seams} / ${g.speakerChanges}`),
  row("Near-duplicate segments", (g) => `${g.hardSegments}`),
  row("False jumps / jumps", (g) => `${g.falseJumps}/${g.jumps}`),
  row("False jumps per jump", (g) => pct(g.falseJumpsPerJump), "≤ 2%"),
  row("Sessions with a false jump", (g) => frac(g.sessionsWithFalseJump, g.sessions)),
  row("False jumps per session", (g) => g.falseJumpsPerSession?.toFixed(2) ?? "—"),
  row("False jumps per 5 min", (g) => g.falseJumpsPer5Min?.toFixed(2) ?? "—"),
  row("Free-recitation checker: false wrong/added words (per 5 min)", (g) => `${g.checkerWarnings} (${g.checkerWarningsPer5Min?.toFixed(2) ?? "—"})`, "≤ 1 per 5 min"),
  row("Planted jumps followed", (g) => frac(g.followed, g.planted)),
  row("Followed within 5 s", (g) => frac(g.followedWithin5s, g.planted)),
  row("Follow latency median / p90", (g) => `${sec(g.followLatencyMedianMs)} / ${sec(g.followLatencyP90Ms)}`),
  row("Correct ayah within 5 s of session start", (g) => frac(g.startWithin5s, g.sessions), "≥ 95%"), "",
  "## False jumps", "",
  ...(falseJumps.length
    ? ["| Session | Time | From | To | Being recited |", "|---|---:|---|---|---|", ...falseJumps.map((j) => `| ${j.id} | ${(j.timeMs / 1000).toFixed(1)} s | ${j.from} | ${j.to} | ${j.truth.join(", ")} |`)]
    : ["None."]), "",
  "## Limits", "",
  "- The ≤ 2% bar in `acceptance-v3` is per clip. Here it is shown per session and per jump; a 3–10 minute session with no false jump is a much stricter test than a 10 s clip with none.",
  "- **Speakers.** tlog has no speaker IDs, and its clusters are coarse. Held-out clips rarely cover consecutive ayahs within one cluster: the longest same-cluster run is about 3.5 minutes. So tlog sessions keep the previous clip's cluster when one is there and change speaker at most seams (the speaker-change row). Teacher evidence is decoded per clip, so to the tracker a speaker change is only a change in evidence quality. A student decoding stitched audio would see more. Help sessions are single-speaker.",
  "- **Contiguity.** A contiguous session steps 1 or 2 ayahs (at most one ayah missing, which is local). Clip spans are the tlog-v2 labels, which can trim an edge word. The tlog pool is the clean-match test and val splits, both held out from student training by cluster.",
  "- **Help sessions** are each eval speaker's clean clips in recording order, 2–10 minutes, for speakers with at least 2 minutes. Nearly every seam is a planted jump, often to a single short ayah: an ar-Rahman refrain ayah, 113:x or 44:1 حم can't be located from its own words, and most unfollowed jumps are of that kind.",
  "- **Hard cases.** Most near-duplicate ayahs (exact phoneme duplicates more than 2 ayahs apart, plus 79:39 / 79:41 and al-Kafirun) fall inside long contiguous runs, so only two dedicated hard sessions were built. The last column slices every session with 3+ near-duplicate segments.",
  "- Stitch seams are cleaner than real pauses: no breath, restart or hesitation. A pass here is necessary, not sufficient; real long sessions (ticket 05) remain the final check.",
  "- Only teacher evidence is scored. Stitched help audio is written under `.scratch/jumps/audio/` for student runs; tlog audio is not local, so tlog sessions have no stitched audio yet.", "",
];
writeFileSync(`${resolve(values.report!)}.md`, lines.join("\n"));
console.log(JSON.stringify(output.groups, null, 1));
