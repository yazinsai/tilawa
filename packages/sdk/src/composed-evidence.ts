import { channelFromCounts, composedReversal, createChecker, createTracker, trackerInput, wrongWordKey } from "@tilawa/core";
import type { ChannelCounts, CheckerOptions, PhonemeEvidence, PhraseAsset, ReferenceIndex, WrongWordCandidate, WrongWordEvidence } from "@tilawa/core";
import { ComposedChannel } from "./composed-channel.js";
import { insertionCells } from "./seam-insertion.js";

/** Both comparisons share complete pronunciations, frame coordinates and one frozen channel. */
export function composedWrongWordScorer(counts: ChannelCounts, index: ReferenceIndex,
  observe?: (candidate: WrongWordCandidate, evidence: WrongWordEvidence | null) => void): NonNullable<CheckerOptions["wrongWordEvidence"]> {
  const channel = new ComposedChannel(channelFromCounts(counts), [...index.tokens]);
  const cache = new Map<string, WrongWordEvidence | null>();
  return (candidate, frames) => {
    const key = wrongWordKey(candidate);
    if (cache.has(key)) return cache.get(key)!;
    const { written, accused, fromFrame: from, toFrame: to } = candidate;
    const composedWritten = channel.acousticLogP(written, frames, from, to);
    const composedAccused = channel.acousticLogP(accused, frames, from, to);
    let result: WrongWordEvidence | null = null;
    if (composedWritten !== null && composedAccused !== null) {
      const greedy = channel.greedy(frames, from, to);
      const greedyWritten = channel.stringLogP(written, greedy), greedyAccused = channel.stringLogP(accused, greedy);
      if (greedyWritten !== null && greedyAccused !== null)
        result = { ...candidate, greedyWritten, greedyAccused, composedWritten, composedAccused };
    }
    cache.set(key, result);
    observe?.(candidate, result);
    return result;
  };
}

/** Discover comparisons with the unchanged engine; carry only valid reversals on a primary token.
 * The scorer never substitutes tokens or feeds its preferences back into discovery.
 */
export function addComposedEvidence(counts: ChannelCounts, chunks: PhonemeEvidence[], index: ReferenceIndex,
  phrases?: PhraseAsset, observe?: (candidate: WrongWordCandidate, evidence: WrongWordEvidence | null) => void): PhonemeEvidence[] {
  const cells = insertionCells(chunks);
  if (!cells.length) return chunks;
  const metadata: WrongWordEvidence[] = [];
  const score = composedWrongWordScorer(counts, index, (c, e) => {
    observe?.(c, e);
    if (e && composedReversal(e)) metadata.push(e);
  });
  const tracker = createTracker(index, { mode: "free", phrases, acquireMs: 1000 });
  const checker = createChecker(index, { phrases, wrongWordEvidence(c, frames) { score(c, frames); return null; } });
  for (const e of cells) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  if (!metadata.length) return chunks;
  const at = chunks.findIndex(c => c.phonemes.length);
  return chunks.map((c, i) => i !== at ? c : { ...c,
    phonemes: c.phonemes.map((p, j) => j ? p : { ...p, wrongWordEvidence: [...(p.wrongWordEvidence ?? []), ...metadata] }) });
}
