import { completePronunciationMatch, createChecker, createTracker, trackerInput, PosteriorFrames, collapseHeld } from "@tilawa/core";
import type { PhonemeEvidence, PhraseAsset, ReferenceIndex, WrongWordRecheckCandidate, WrongWordRepairEvidence } from "@tilawa/core";
import { createListener } from "./listener.js";
import type { LoadedModel } from "./model.js";
import { NONLOCAL_GEOMETRY, NonlocalAcousticDenoiser } from "./nonlocal-acoustic.js";
import type { NonlocalGeometry } from "./nonlocal-acoustic.js";
import { insertionCells } from "./seam-insertion.js";

/** Discover actual checker windows while preserving its primary tokens, positions and decisions. */
export function wrongWordRechecks(chunks: PhonemeEvidence[], index: ReferenceIndex, phrases?: PhraseAsset): WrongWordRecheckCandidate[] {
  const candidates = new Map<string, WrongWordRecheckCandidate>();
  const tracker = createTracker(index, { mode:"free", phrases, acquireMs:1000 });
  const checker = createChecker(index,{ phrases, wrongWordRecheck(c) { candidates.set(JSON.stringify(c),c); return true; } });
  for (const e of insertionCells(chunks)) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  return [...candidates.values()];
}

/** A crop's origin is on the native frame grid; both replay arms use the same state and padding. */
export async function replayAcousticSlot(model: LoadedModel, audio: Float32Array, c: WrongWordRecheckCandidate,
  frameMs: number): Promise<{ phones:string; writtenNll:number; rivalNll:number }> {
  const from = Math.max(0,c.fromFrame-Math.ceil(8000/frameMs));
  const to = Math.min(audio.length,Math.ceil((c.toFrame*frameMs+300)*16));
  const listener = createListener(model,{posteriors:true,topK:model.manifest.tokens.length+1});
  const frames = new PosteriorFrames(model.manifest.tokens.length);
  const phones = [];
  const start = Math.round(from*frameMs*16);
  for (let at = start; at < to; at += 320) {
    const e = await listener.push(audio.subarray(at,Math.min(to,at+320)),16000);
    if (e) {
      phones.push(...e.phonemes);
      if (e.posteriors) frames.append({...e.posteriors,startFrame:e.posteriors.startFrame+from});
    }
  }
  const end = await listener.end();
  if (end) {
    phones.push(...end.phonemes);
    if (end.posteriors) frames.append({...end.posteriors,startFrame:end.posteriors.startFrame+from});
  }
  const ids = new Map([...model.manifest.tokens].map((p,i) => [p,i+1]));
  const nll = (s:string) => {
    const labels = collapseHeld([...s]).map(p => ids.get(p));
    return labels.some(p => p === undefined) || c.fromFrame < frames.startFrame || c.toFrame > frames.endFrame
      ? Infinity : frames.nll(labels as number[],c.fromFrame,c.toFrame);
  };
  return {phones:phones.filter(p => p.start+from >= c.fromFrame && p.start+from < c.toFrame).map(p => p.p).join(""),
    writtenNll:nll(c.written),rivalNll:nll(c.observed)};
}

/** No assets or text enter the denoiser. Only positive, reconstruction-specific readings become certificates. */
export async function addNonlocalEvidence(model: LoadedModel, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset, geometry: NonlocalGeometry | null = NONLOCAL_GEOMETRY): Promise<PhonemeEvidence[]> {
  if (!geometry || !audio.length || !chunks.length || model.manifest.sampleRate !== 16000) return chunks;
  const candidates = wrongWordRechecks(chunks,index,phrases);
  if (!candidates.length) return chunks;
  const denoiser = new NonlocalAcousticDenoiser(audio,geometry), metadata: WrongWordRepairEvidence[] = [];
  const frameMs = chunks[0].frameMs;
  for (const c of candidates) {
    const a = Math.round(c.fromFrame*frameMs*16), b = Math.round(c.toFrame*frameMs*16);
    if (a < 0 || b > audio.length || b <= a) continue;
    const repaired = denoiser.repair(a,b);
    if (!repaired.patches.length) continue;
    const raw = await replayAcousticSlot(model,audio,c,frameMs);
    // A crop that already recovers the word is not evidence for nonlocal reconstruction.
    if (completePronunciationMatch(c.written,raw.phones)) continue;
    const recovered = await replayAcousticSlot(model,repaired.samples,c,frameMs);
    if (!completePronunciationMatch(c.written,recovered.phones) || completePronunciationMatch(c.observed,recovered.phones) ||
      !Number.isFinite(recovered.writtenNll) || !Number.isFinite(recovered.rivalNll) || recovered.writtenNll > recovered.rivalNll) continue;
    const w = index.words[c.word];
    metadata.push({ word:{surah:w.surah,ayah:w.ayah,word:w.word}, written:c.written, observed:c.observed,
      fromFrame:c.fromFrame,toFrame:c.toFrame,recovered:recovered.phones,writtenNll:recovered.writtenNll,rivalNll:recovered.rivalNll });
  }
  if (!metadata.length) return chunks;
  const at = chunks.findIndex(c => c.phonemes.length);
  return chunks.map((c,i) => i !== at ? c : { ...c, phonemes:c.phonemes.map((p,j) => j ? p : {
    ...p,wrongWordRepairs:[...(p.wrongWordRepairs ?? []),...metadata],
  }) });
}
