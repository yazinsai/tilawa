import { createChecker, createTracker, GAP_PAD_S, PosteriorFrames, trackerInput } from "@tilawa/core";
import type { CheckerOptions, PhonemeEvidence, ReferenceIndex, SharedTranscriptEvidence } from "@tilawa/core";
import { decodeIsolated } from "./gap-decode.js";
import type { LoadedModel } from "./model.js";
import { insertionCells } from "./seam-insertion.js";

export type TranscriptCandidate = Omit<SharedTranscriptEvidence, "cropMargin">;

/** Normalized equal-weight geometric pool. Each NLL must already sum complete CTC alignments in its own view.
 * Normalizing each view over the two transcripts cancels in the pooled log odds; no frame correspondence is used. */
export function poolTranscriptScores(primary: readonly [number, number], crop: readonly [number, number]) {
  if (![...primary, ...crop].every(Number.isFinite)) return null;
  const primaryMargin = primary[0] - primary[1], cropMargin = crop[0] - crop[1];
  const pooledMargin = (primaryMargin + cropMargin) / 2;
  const writtenProbability = pooledMargin >= 0
    ? Math.exp(-pooledMargin) / (1 + Math.exp(-pooledMargin)) : 1 / (1 + Math.exp(pooledMargin));
  return { primaryMargin, cropMargin, pooledMargin, writtenProbability,
    veto: primaryMargin > 0 && pooledMargin < 0 };
}

/** Candidate discovery does not alter the primary checker, tracker or confirmation bars. */
export function transcriptCandidates(chunks: PhonemeEvidence[], index: ReferenceIndex, options: CheckerOptions = {}) {
  const candidates = new Map<string, TranscriptCandidate>();
  const tracker = createTracker(index, { mode: options.passage ? "memorization" : "free", phrases: options.phrases });
  const checker = createChecker(index, { ...options, sharedTranscript(c) {
    candidates.set(JSON.stringify(c), c);
  } });
  for (const e of insertionCells(chunks)) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  return [...candidates.values()];
}

export function carryTranscriptEvidence(chunks: PhonemeEvidence[], certificates: SharedTranscriptEvidence[]) {
  if (!certificates.length) return chunks;
  const at = chunks.findIndex(c => c.phonemes.length);
  return chunks.map((c, i) => i !== at ? c : { ...c, phonemes: c.phonemes.map((p, j) => j ? p : {
    ...p, sharedTranscripts: [...(p.sharedTranscripts ?? []), ...certificates],
  }) });
}

/** Re-encode a bounded cold crop while retaining its collar, then score only the original accusation interval.
 * The primary view retains its original streaming context. Neither view's alignments constrain the other. */
export async function transcriptCertificates(model: LoadedModel, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, candidates: TranscriptCandidate[], report?: (c: TranscriptCandidate, result: ReturnType<typeof poolTranscriptScores>) => void) {
  const certificates: SharedTranscriptEvidence[] = [];
  if (!candidates.length || !audio.length || !chunks.length || chunks.some(c => !c.posteriors)) return certificates;
  const frameMs = chunks[0].frameMs, rate = model.manifest.sampleRate, k = chunks[0].posteriors!.k;
  if (frameMs !== model.manifest.frameMs || model.manifest.inventory !== index.inventoryId || model.manifest.tokens !== [...index.tokens].join(""))
    throw new Error("Shared-transcript view clock or inventory differs from primary evidence");
  const frameSamples = frameMs * rate / 1000, pad = Math.ceil(GAP_PAD_S * 1000 / frameMs);
  const ids = new Map([...index.tokens].map((p, i) => [p, i + 1]));
  const cache = new Map<string, PosteriorFrames>();
  for (const c of candidates) {
    const start = Math.max(0, c.fromFrame - pad), end = Math.min(Math.ceil(audio.length / frameSamples), c.toFrame + pad);
    if (end <= start || (end - start) * frameMs > 8000 || !c.written || !c.rival || c.written === c.rival ||
      ![...c.written, ...c.rival].every(p => ids.has(p))) { report?.(c, null); continue; }
    const key = `${start}:${end}`;
    let frames = cache.get(key);
    if (!frames) {
      const crop = await decodeIsolated(model, audio.subarray(Math.round(start * frameSamples), Math.round(end * frameSamples)), rate, k);
      frames = new PosteriorFrames(index.tokens.size);
      frames.append(crop);
      cache.set(key, frames);
    }
    const a = c.fromFrame - start, b = c.toFrame - start;
    if (a < frames.startFrame || b > frames.endFrame) { report?.(c, null); continue; }
    const labels = (s: string) => [...s].map(p => ids.get(p)!);
    const result = poolTranscriptScores([c.primaryMargin, 0], [frames.nll(labels(c.written), a, b), frames.nll(labels(c.rival), a, b)]);
    report?.(c, result);
    if (result?.veto) certificates.push({ ...c, cropMargin: result.cropMargin });
  }
  return certificates;
}

export async function addTranscriptEvidence(model: LoadedModel, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, options: CheckerOptions = {}) {
  const candidates = transcriptCandidates(chunks, index, options);
  const certificates = await transcriptCertificates(model, audio, chunks, index, candidates);
  return carryTranscriptEvidence(chunks, certificates);
}
