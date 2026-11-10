import { createChecker, createTracker, PosteriorFrames, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { insertionCells } from "./seam-insertion.js";
import { coupledTiming } from "./coupled-timing.js";

/** Synchronous recheck on the same rows already available to a streaming checker. */
export function sequenceEvidenceRecheck(index: ReferenceIndex) {
  const frames = new PosteriorFrames(index.tokens.size);
  const ids = new Map([...index.tokens].map((p,i) => [p,i+1]));
  const encode = (s: string) => [...s].map(p => ids.get(p) ?? 0);
  const cache = new Map<string,boolean>();
  return {
    append(e: PhonemeEvidence) { if (e.posteriors) frames.append(e.posteriors); },
    permits(c: SequenceEvidenceCandidate): boolean {
      const key = JSON.stringify(c);
      const cached = cache.get(key);
      if (cached !== undefined) return cached;
      // Insufficient rows abstain without caching, since a later push can complete them.
      if (frames.startFrame < 0 || c.fromFrame < frames.startFrame || c.toFrame > frames.endFrame) return true;
      const result = coupledTiming(frames,encode(c.written),encode(c.rival),c.fromFrame,c.toFrame);
      const permitted = !(result && result.independent < 0 && result.normalizedIndependent < 0 &&
        result.uncoupled < 0 && result.coupled > 0 && result.sharedBoundariesDiffer);
      cache.set(key,permitted);
      return permitted;
    },
  };
}

/** Discover only already-supported substitutions; preserve primary rows, phones and insertion certificates. */
export function addSequenceEvidence(chunks: PhonemeEvidence[], index: ReferenceIndex, phrases?: PhraseAsset): PhonemeEvidence[] {
  const cells = insertionCells(chunks);
  if (!cells.length) return chunks;
  const recheck = sequenceEvidenceRecheck(index);
  for (const e of cells) recheck.append(e);
  const certificates: SequenceEvidenceCandidate[] = [], seen = new Set<string>();
  const tracker = createTracker(index, { mode: "free" });
  const checker = createChecker(index, { phrases, sequenceEvidence(c) {
    const key = JSON.stringify(c);
    if (!seen.has(key)) {
      seen.add(key);
      // Require a coupling reversal, not a change caused solely by timing-prior normalization.
      if (!recheck.permits(c)) certificates.push(c);
    }
    return true;
  } });
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  if (!certificates.length) return chunks;
  const at = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e,i) => i !== at ? e : { ...e, phonemes: e.phonemes.map((p,j) => j ? p :
    { ...p, sequenceVetoes: [...(p.sequenceVetoes ?? []),...certificates] }) });
}
