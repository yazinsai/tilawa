import type { CompletionExample } from "../../core/dist/completion-survival.js";
import type { LocalSnapshot } from "../../core/dist/local.js";

export interface CompletionLabel {
  id: string; speaker: string; split: string; kind: string; scored: boolean;
  wav: string; ayahs: string[]; events: unknown[]; non_errors: unknown[]; excluded?: unknown;
}

export function completionTrainingLabels(all: readonly CompletionLabel[]): CompletionLabel[] {
  const excluded = new Set(all.filter(l => l.split !== "train").map(l => l.speaker));
  return all.filter(l => l.split === "train" && !excluded.has(l.speaker) && l.speaker?.length > 0
    && l.kind === "none" && l.scored === true && !l.excluded && Array.isArray(l.events) && !l.events.length
    && Array.isArray(l.non_errors) && !l.non_errors.length);
}

/**
 * Actual tracker alternatives near its existing commit margin, labelled by later transcript-consistent progress.
 * A new run that bypasses the candidate retires it. No later consistent progress means right censoring, including
 * a recording ending partway through the final word. The transcript is used only here, never during tracking.
 */
export function completionExamples(trace: readonly LocalSnapshot[], transcript: ReadonlySet<number>): CompletionExample[] {
  const out: CompletionExample[] = [];
  for (let t = 0; t < trace.length; t++) {
    const seen = new Set<string>();
    for (const c of trace[t].candidates) {
      const p = c.prefix;
      if (!p || !p.observations || !p.matches || p.progress >= p.length
        || c.cost > 10 * Math.max(1, 30 / Math.max(1, trace[t].token))) continue;
      const key = JSON.stringify([p.word, p.entry, p.progress, p.observations, p.matches, p.editCost, p.lastCost, p.stalled, c.origin]);
      if (seen.has(key)) continue;
      seen.add(key);
      let outcome: CompletionExample["outcome"] = "censored", delay = trace.length - t;
      for (let j = t + 1; j < trace.length; j++) {
        if (trace[j].token <= trace[j - 1].token) {
          // A new local search or explicit rewind retires this candidate before its later lookalike can complete it.
          outcome = "abandon"; delay = j - t; break;
        }
        const future = trace[j].selected;
        if (!future || !transcript.has(future.word) || !future.matches || future.lastCost !== 0) continue;
        const selected = trace[j].candidates.find(x => x.prefix === future);
        if (!selected) continue;
        if (transcript.has(p.word) && future.word === p.word + 1 && selected.origin === c.origin) {
          outcome = "complete"; delay = j - t; break;
        }
        if (future.word > p.word || !transcript.has(p.word)) {
          outcome = "abandon"; delay = j - t; break;
        }
      }
      out.push({ prefix: { ...p }, delay: Math.max(1, delay), outcome });
    }
  }
  // Equal total weight for each training recording; many alternatives do not duplicate recording supervision.
  for (const e of out) e.weight = 1 / Math.max(1, out.length);
  return out;
}
