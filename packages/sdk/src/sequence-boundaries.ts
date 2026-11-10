import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceBoundaryEvidence } from "@tilawa/core";
import { conditionedComparison } from "./conditioned-ctc.js";
import { insertionCells } from "./seam-insertion.js";

/** Certificates travel with a primary token, preserving every primary value. */
export function carryBoundaryEvidence(chunks: PhonemeEvidence[], metadata: SequenceBoundaryEvidence[]) {
  if (!metadata.length) return chunks;
  const at = chunks.findIndex(c => c.phonemes.length);
  return chunks.map((c, i) => i !== at ? c : { ...c, phonemes: c.phonemes.map((p, j) => j ? p :
    { ...p, sequenceBoundaries: [...(p.sequenceBoundaries ?? []), ...metadata] }) });
}

/** Read-only primary replay; no waveform inference or auxiliary model is needed. */
export function addBoundaryEvidence(chunks: PhonemeEvidence[], index: ReferenceIndex, phrases?: PhraseAsset) {
  const cells = insertionCells(chunks);
  if (!cells.length) return chunks;
  const tokens = new Map([...index.tokens].map((p, i) => [p, i + 1]));
  const seen = new Set<string>(), metadata: SequenceBoundaryEvidence[] = [];
  const tracker = createTracker(index, { mode: "free", phrases });
  const checker = createChecker(index, { phrases, sequenceEvidence(c, frames) {
    const result = conditionedComparison(frames, c, tokens);
    if (result?.reverses) {
      const m = { word: c.word, written: c.written, accused: c.accused, fromFrame: c.fromFrame, toFrame: c.toFrame,
        resetMargin: c.margin, conditionalMargin: result.margin };
      const key = JSON.stringify(m);
      if (!seen.has(key)) { seen.add(key); metadata.push(m); }
    }
    return true;
  } });
  for (const e of cells) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  return carryBoundaryEvidence(chunks, metadata);
}
