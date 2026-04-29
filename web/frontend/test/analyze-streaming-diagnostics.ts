/**
 * Analyze diagnose-streaming.ts JSON with tracker-invariant oriented summaries.
 *
 * This is intentionally diagnostic-only. It does not tune thresholds or encode
 * corpus-specific rules; it surfaces structural patterns:
 *   - commits whose selected ref is not the best ranked candidate by existing scores
 *   - stale exits after recent tracking cycles still had progress evidence
 *   - pending-emission lifecycle events
 *   - exact-match/extra/missing shape using the sample's expected refs
 *
 * Usage:
 *   npx tsx test/analyze-streaming-diagnostics.ts test/debug/foo.json
 *   npx tsx test/analyze-streaming-diagnostics.ts test/debug/a.json test/debug/b.json
 */

import { readFileSync } from "node:fs";
import { basename, resolve } from "node:path";

type DiagnosticEvent =
  | {
      type: "discovery_cycle";
      text: string;
      final_flush: boolean;
      candidates: CandidateDiagnostic[];
      time_sec: number;
    }
  | {
      type: "tracking_cycle";
      ref: string;
      text_length: number;
      word_matches: number;
      acoustic_word: number | null;
      char_word: number | null;
      advanced: boolean;
      final_flush: boolean;
      time_sec: number;
    }
  | {
      type: "pending_emission";
      action: "created" | "confirmed" | "final_flush_emit" | "dropped";
      ref: string;
      margin: number | null;
      fresh_samples: number;
      matched_indices?: number[];
      time_sec: number;
    }
  | {
      type: "commit";
      ref: string;
      reason: string;
      confidence: number;
      time_sec: number;
    }
  | {
      type: "stale_exit";
      ref: string;
      stale_cycles: number;
      time_sec: number;
    }
  | { type: "silence_skip"; mode: string; reason: string; time_sec: number }
  | { type: "flush"; mode: string; duration_sec: number; time_sec: number }
  | { type: "rollback"; reason: string; restored_ref: string | null; time_sec: number };

interface CandidateDiagnostic {
  ref: string;
  kind: "single" | "span";
  stageA: number;
  acoustic: number;
  acousticMargin?: number;
  lengthFit?: number;
  fusion?: number;
  feasible?: boolean;
}

interface DiagnosedSample {
  id: string;
  category: string;
  source: string;
  expected: string[];
  discovered: string[];
  passed: boolean;
  diagnostics: DiagnosticEvent[];
}

interface DiagnosticRun {
  corpus: string;
  generated_at: string;
  samples: DiagnosedSample[];
}

interface CommitAnalysis {
  sampleId: string;
  timeSec: number;
  ref: string;
  reason: string;
  selectedRank: number | null;
  topRef: string | null;
  topFusion: number | null;
  selectedFusion: number | null;
  selectedFeasible: boolean | null;
  selectedAcousticMargin: number | null;
  selectedLengthFit: number | null;
  finalFlush: boolean | null;
  transcriptChars: number | null;
}

interface StaleExitAnalysis {
  sampleId: string;
  ref: string;
  timeSec: number;
  staleCycles: number;
  trackingCycles: number;
  advancedCycles: number;
  primaryWordEvidenceCycles: number;
  acousticFallbackCycles: number;
  charFallbackCycles: number;
  maxTextLength: number;
}

function expandRef(ref: string): string[] {
  const span = ref.match(/^(\d+):(\d+)-(\d+)$/);
  if (!span) return [ref];
  const surah = Number(span[1]);
  const start = Number(span[2]);
  const end = Number(span[3]);
  const refs: string[] = [];
  for (let ayah = start; ayah <= end; ayah++) refs.push(`${surah}:${ayah}`);
  return refs;
}

function refMatches(commitRef: string, candidateRef: string): boolean {
  if (commitRef === candidateRef) return true;
  const commitSet = new Set(expandRef(commitRef));
  const candidateSet = new Set(expandRef(candidateRef));
  if (commitSet.size !== candidateSet.size) return false;
  return [...commitSet].every((ref) => candidateSet.has(ref));
}

function classifyShape(expected: string[], discovered: string[]): string {
  if (
    expected.length === discovered.length &&
    expected.every((ref, idx) => discovered[idx] === ref)
  ) {
    return "exact";
  }
  if (discovered.length === 0) return "no_emit";

  const expectedSet = new Set(expected);
  const discoveredSet = new Set(discovered);
  const missing = expected.filter((ref) => !discoveredSet.has(ref));
  const extras = discovered.filter((ref) => !expectedSet.has(ref));
  if (extras.length === 0 && missing.length > 0) return "missing_only";
  if (missing.length > 0 && expected.length > 1) return "partial_multi";
  if (!expectedSet.has(discovered[0])) return "wrong_initial";

  const expectedPositions = expected
    .map((ref) => discovered.indexOf(ref))
    .filter((idx) => idx >= 0);
  const extrasBeforeExpected =
    expectedPositions.length > 0 &&
    discovered.slice(0, Math.min(...expectedPositions)).some((ref) => !expectedSet.has(ref));
  if (extrasBeforeExpected) return "extra_before_expected";

  const lastExpectedPos = expectedPositions.length ? Math.max(...expectedPositions) : -1;
  const extraAfterExpected = discovered
    .slice(lastExpectedPos + 1)
    .some((ref) => !expectedSet.has(ref));
  if (extraAfterExpected) {
    const expectedSurahs = new Set(expected.map((ref) => ref.split(":")[0]));
    const wrongSurah = discovered
      .slice(lastExpectedPos + 1)
      .some((ref) => !expectedSurahs.has(ref.split(":")[0]));
    return wrongSurah ? "wrong_surah_jump" : "extra_after_expected";
  }
  return "other";
}

function analyzeSample(sample: DiagnosedSample): {
  commits: CommitAnalysis[];
  staleExits: StaleExitAnalysis[];
  pendingActions: DiagnosticEvent[];
  shape: string;
} {
  const commits: CommitAnalysis[] = [];
  const staleExits: StaleExitAnalysis[] = [];
  const pendingActions: DiagnosticEvent[] = [];
  let lastDiscovery: Extract<DiagnosticEvent, { type: "discovery_cycle" }> | null = null;
  let trackingSinceCommit: Extract<DiagnosticEvent, { type: "tracking_cycle" }>[] = [];

  for (const event of sample.diagnostics) {
    if (event.type === "discovery_cycle") {
      lastDiscovery = event;
    } else if (event.type === "tracking_cycle") {
      trackingSinceCommit.push(event);
    } else if (event.type === "pending_emission") {
      pendingActions.push(event);
    } else if (event.type === "commit") {
      const candidates = lastDiscovery?.candidates ?? [];
      const selectedRank = candidates.findIndex((candidate) =>
        refMatches(event.ref, candidate.ref),
      );
      const selected = selectedRank >= 0 ? candidates[selectedRank] : null;
      const top = candidates[0] ?? null;
      commits.push({
        sampleId: sample.id,
        timeSec: event.time_sec,
        ref: event.ref,
        reason: event.reason,
        selectedRank: selectedRank >= 0 ? selectedRank + 1 : null,
        topRef: top?.ref ?? null,
        topFusion: top?.fusion ?? null,
        selectedFusion: selected?.fusion ?? null,
        selectedFeasible: selected?.feasible ?? null,
        selectedAcousticMargin: selected?.acousticMargin ?? null,
        selectedLengthFit: selected?.lengthFit ?? null,
        finalFlush: lastDiscovery?.final_flush ?? null,
        transcriptChars: lastDiscovery?.text.length ?? null,
      });
      trackingSinceCommit = [];
      lastDiscovery = null;
    } else if (event.type === "stale_exit") {
      staleExits.push({
        sampleId: sample.id,
        ref: event.ref,
        timeSec: event.time_sec,
        staleCycles: event.stale_cycles,
        trackingCycles: trackingSinceCommit.length,
        advancedCycles: trackingSinceCommit.filter((cycle) => cycle.advanced).length,
        primaryWordEvidenceCycles: trackingSinceCommit.filter(
          (cycle) => cycle.word_matches > 0,
        ).length,
        acousticFallbackCycles: trackingSinceCommit.filter(
          (cycle) => cycle.acoustic_word !== null,
        ).length,
        charFallbackCycles: trackingSinceCommit.filter(
          (cycle) => cycle.char_word !== null,
        ).length,
        maxTextLength: Math.max(0, ...trackingSinceCommit.map((cycle) => cycle.text_length)),
      });
      trackingSinceCommit = [];
    }
  }

  return {
    commits,
    staleExits,
    pendingActions,
    shape: classifyShape(sample.expected, sample.discovered),
  };
}

function pct(numerator: number, denominator: number): string {
  if (denominator === 0) return "n/a";
  return `${((numerator / denominator) * 100).toFixed(1)}%`;
}

function main() {
  const paths = process.argv.slice(2);
  if (paths.length === 0) {
    throw new Error("Usage: tsx test/analyze-streaming-diagnostics.ts <diagnostic.json> [...]");
  }

  const allSamples: DiagnosedSample[] = [];
  for (const path of paths) {
    const run: DiagnosticRun = JSON.parse(readFileSync(resolve(path), "utf-8"));
    allSamples.push(...run.samples);
  }

  const analyses = allSamples.map((sample) => ({ sample, ...analyzeSample(sample) }));
  const commits = analyses.flatMap((analysis) => analysis.commits);
  const staleExits = analyses.flatMap((analysis) => analysis.staleExits);
  const pendingActions = analyses.flatMap((analysis) => analysis.pendingActions);
  const exact = analyses.filter(({ shape }) => shape === "exact").length;

  const nonTopCommits = commits.filter(
    (commit) => commit.selectedRank !== null && commit.selectedRank > 1,
  );
  const missingCandidateCommits = commits.filter((commit) => commit.selectedRank === null);
  const infeasibleCommits = commits.filter((commit) => commit.selectedFeasible === false);
  const staleAfterProgress = staleExits.filter(
    (exit) =>
      exit.advancedCycles > 0 ||
      exit.primaryWordEvidenceCycles > 0 ||
      exit.acousticFallbackCycles > 0 ||
      exit.charFallbackCycles > 0,
  );

  const shapeCounts = new Map<string, number>();
  for (const { shape } of analyses) {
    shapeCounts.set(shape, (shapeCounts.get(shape) ?? 0) + 1);
  }

  console.log(`Diagnostic analysis: ${paths.map((p) => basename(p)).join(", ")}`);
  console.log(`Samples: ${allSamples.length} | exact: ${exact}/${allSamples.length} (${pct(exact, allSamples.length)})`);
  console.log(
    `Shapes: ${[...shapeCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([shape, count]) => `${shape}: ${count}`)
      .join(", ")}`,
  );
  console.log();

  console.log(`Commits: ${commits.length}`);
  console.log(`  non-top selected candidate: ${nonTopCommits.length}`);
  console.log(`  selected candidate missing from last discovery list: ${missingCandidateCommits.length}`);
  console.log(`  selected candidate infeasible by CTC: ${infeasibleCommits.length}`);
  for (const commit of nonTopCommits.slice(0, 12)) {
    console.log(
      `    ${commit.sampleId} @${commit.timeSec.toFixed(1)}s ${commit.ref} rank=${commit.selectedRank} top=${commit.topRef} reason=${commit.reason} fusion=${commit.selectedFusion ?? "?"}/${commit.topFusion ?? "?"}`,
    );
  }
  if (missingCandidateCommits.length > 0) {
    console.log("  Missing-candidate commits:");
    for (const commit of missingCandidateCommits.slice(0, 12)) {
      console.log(
        `    ${commit.sampleId} @${commit.timeSec.toFixed(1)}s ${commit.ref} reason=${commit.reason}`,
      );
    }
  }
  console.log();

  console.log(`Stale exits: ${staleExits.length}`);
  console.log(`  stale exits after some tracking evidence: ${staleAfterProgress.length}`);
  for (const exit of staleAfterProgress.slice(0, 12)) {
    console.log(
      `    ${exit.sampleId} @${exit.timeSec.toFixed(1)}s ${exit.ref} cycles=${exit.trackingCycles} advanced=${exit.advancedCycles} primary=${exit.primaryWordEvidenceCycles} acoustic=${exit.acousticFallbackCycles} char=${exit.charFallbackCycles} maxText=${exit.maxTextLength}`,
    );
  }
  console.log();

  const pendingCounts = new Map<string, number>();
  for (const event of pendingActions) {
    if (event.type !== "pending_emission") continue;
    pendingCounts.set(event.action, (pendingCounts.get(event.action) ?? 0) + 1);
  }
  console.log(
    `Pending emissions: ${
      pendingActions.length
        ? [...pendingCounts.entries()].map(([k, v]) => `${k}: ${v}`).join(", ")
        : "none"
    }`,
  );
}

main();
