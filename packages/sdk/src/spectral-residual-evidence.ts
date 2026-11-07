import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { SpectralResidualEvidence, SpectralResidualRequest, PhonemeEvidence, PhraseAsset, ReferenceIndex } from "@tilawa/core";
import { spectralObservations, type createSpectralResidualScorer } from "./spectral-residual.js";
import { insertionCells } from "./seam-insertion.js";

/** Discovery observes final primary decisions; it cannot alter alignment or insertion ownership. */
export function spectralResidualRequests(chunks: PhonemeEvidence[], index: ReferenceIndex, phrases?: PhraseAsset): SpectralResidualRequest[] {
  const requests = new Map<string, SpectralResidualRequest>();
  const tracker = createTracker(index, { mode: "free", phrases, acquireMs: 1000 });
  const checker = createChecker(index, { phrases, spectralDecision(_id, r) { requests.set(JSON.stringify(r), r); } });
  for (const e of insertionCells(chunks)) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  return [...requests.values()];
}

export function carrySpectralResidualEvidence(chunks: PhonemeEvidence[], certificates: SpectralResidualEvidence[]): PhonemeEvidence[] {
  const first = chunks.findIndex(c => c.phonemes.length);
  return !certificates.length ? chunks : chunks.map((c, i) => i !== first ? c : {
    ...c, phonemes: c.phonemes.map((p, j) => j ? p : { ...p, spectralResiduals: certificates }),
  });
}

/** Re-extracts the very same globally aligned FFT windows; no acoustic context is needed by this frame-local scorer. */
export function compareSpectralWindow(scorer: ReturnType<typeof createSpectralResidualScorer>, audio: Float32Array,
  r: SpectralResidualRequest, frameMs = 40) {
  const first = r.fromFrame * frameMs * 16, last = r.toFrame * frameMs * 16 + 240;
  if (![first, last].every(Number.isInteger) || first < 0 || last <= first) return null;
  // The last 25 ms analysis window starts one 10 ms hop before the exclusive end.
  const features = spectralObservations(audio.subarray(first, last), scorer.parameters);
  return scorer.compare(features, { ...r, fromFrame: 0, toFrame: r.toFrame - r.fromFrame }, frameMs);
}

export function addSpectralResidualEvidence(scorer: ReturnType<typeof createSpectralResidualScorer>, audio: Float32Array,
  chunks: PhonemeEvidence[], index: ReferenceIndex, phrases?: PhraseAsset): PhonemeEvidence[] {
  if (!chunks.length) return chunks;
  const certificates: SpectralResidualEvidence[] = [];
  for (const r of spectralResidualRequests(chunks, index, phrases)) {
    const score = compareSpectralWindow(scorer, audio, r, chunks[0].frameMs);
    if (!score || score.writtenLogLikelihood <= score.accusedLogLikelihood) continue;
    const { word, ...bounds } = r, w = index.words[word];
    certificates.push({ ...bounds, word: { surah: w.surah, ayah: w.ayah, word: w.word }, ...score });
  }
  return carrySpectralResidualEvidence(chunks, certificates);
}
