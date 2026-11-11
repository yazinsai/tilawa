import { completePronunciationMatch, createChecker, createTracker, PosteriorFrames, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence, PhraseAsset, ReferenceIndex, WrongWordRecheckCandidate } from "@tilawa/core";
import { createListener } from "./listener.js";
import type { LoadedModel } from "./model.js";
import { insertionCells } from "./seam-insertion.js";

export interface RobustNormVerdict {
  candidate: WrongWordRecheckCandidate;
  recovered: string;
  writtenNll: number;
  accusedNll: number;
  gainMargin: number;
  veto: boolean;
}

/** Complete-reading witness plus the original confirmation bar on exactly the original acoustic coordinates. */
export function robustNormVerdict(c: WrongWordRecheckCandidate, evidence: PhonemeEvidence,
  index: ReferenceIndex, control: PhonemeEvidence): RobustNormVerdict | null {
  const posts = evidence.posteriors;
  if (evidence.inventory !== index.inventoryId || !posts || c.fromFrame < posts.startFrame ||
      c.toFrame > posts.startFrame + posts.nFrames) return null;
  if (control.inventory !== evidence.inventory || control.frameMs !== evidence.frameMs ||
      !control.posteriors || c.fromFrame < control.posteriors.startFrame ||
      c.toFrame > control.posteriors.startFrame + control.posteriors.nFrames) return null;
  const frames = new PosteriorFrames(index.tokens.size);
  frames.append(posts);
  const gainFrames = new PosteriorFrames(index.tokens.size);
  gainFrames.append(control.posteriors);
  const ids = new Map([...index.tokens].map((p, i) => [p, i + 1]));
  const seq = (s: string) => [...s].map(p => ids.get(p) ?? 0);
  const writtenNll = frames.nll(seq(c.written), c.fromFrame, c.toFrame);
  const accusedNll = frames.nll(seq(c.accused), c.fromFrame, c.toFrame);
  const gainMargin = gainFrames.nll(seq(c.written), c.fromFrame, c.toFrame) -
    gainFrames.nll(seq(c.accused), c.fromFrame, c.toFrame);
  const recovered = evidence.phonemes.filter(p => p.start >= c.slotFromFrame && p.start < c.slotToFrame)
    .map(p => p.p).join("");
  const veto = Number.isFinite(writtenNll) && Number.isFinite(accusedNll) &&
    completePronunciationMatch(c.wordWritten, recovered) && !completePronunciationMatch(c.wordAccused, recovered) &&
    Number.isFinite(gainMargin) && writtenNll - accusedNll < c.need &&
    writtenNll - accusedNll < gainMargin;
  return { candidate: c, recovered, writtenNll, accusedNll, gainMargin, veto };
}

/** Independent full-stream states isolate normalization from crop and grid effects. */
export async function decodeRobustNorm(model: LoadedModel, audio: Float32Array, k: number): Promise<PhonemeEvidence> {
  const listener = createListener(model, { posteriors: true, topK: k });
  const chunks: PhonemeEvidence[] = [];
  const step = Math.round(model.manifest.sampleRate * 0.02);
  for (let a = 0; a < audio.length; a += step) {
    const e = await listener.push(audio.subarray(a, a + step), model.manifest.sampleRate);
    if (e) chunks.push(e);
  }
  const tail = await listener.end();
  if (tail) chunks.push(tail);
  return { inventory: model.manifest.inventory, frameMs: model.manifest.frameMs,
    phonemes: chunks.flatMap(e => e.phonemes), posteriors: { k, startFrame: 0,
      nFrames: chunks.reduce((n, e) => n + (e.posteriors?.nFrames ?? 0), 0),
      ids: chunks.flatMap(e => Array.from(e.posteriors?.ids ?? [])),
      probs: chunks.flatMap(e => Array.from(e.posteriors?.probs ?? [])) } };
}

/** Discover through the ordinary unchanged tracker/checker; labels never select inference windows. */
export async function recheckWrongWords(model: LoadedModel, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset): Promise<RobustNormVerdict[]> {
  if (!model.robustNormModel || !model.robustNormGainModel || !audio.length || !chunks[0]?.posteriors) return [];
  const candidates = new Map<string, WrongWordRecheckCandidate>();
  const tracker = createTracker(index, { mode: "free", phrases });
  const checker = createChecker(index, { phrases, wrongWordRecheck(c) {
    candidates.set(JSON.stringify(c), c);
    return true;
  } });
  for (const e of insertionCells(chunks)) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  if (!candidates.size) return [];
  const side = await decodeRobustNorm(model.robustNormModel, audio, chunks[0].posteriors.k);
  const control = await decodeRobustNorm(model.robustNormGainModel, audio, chunks[0].posteriors.k);
  const last = chunks.at(-1)?.posteriors;
  for (const view of [side, control]) if (view.frameMs !== chunks[0].frameMs ||
      view.posteriors?.startFrame !== chunks[0].posteriors.startFrame || !last ||
      view.posteriors.startFrame + view.posteriors.nFrames !== last.startFrame + last.nFrames)
      throw new Error("RobustNorm clock differs from the primary listener");
  return [...candidates.values()].map(c => robustNormVerdict(c, side, index, control))
    .filter((v): v is RobustNormVerdict => v !== null);
}

/** Serialize span-specific witnesses on a primary token; the tracker ignores auxiliary fields. */
export async function addRobustNormEvidence(model: LoadedModel, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset): Promise<PhonemeEvidence[]> {
  const verdicts = await recheckWrongWords(model, audio, chunks, index, phrases);
  const metadata = verdicts.filter(v => v.veto).map(v => ({ ...v.candidate, recovered: v.recovered,
    writtenNll: v.writtenNll, accusedNll: v.accusedNll, gainMargin: v.gainMargin }));
  if (!metadata.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e, i) => i !== first ? e : { ...e, phonemes: e.phonemes.map((p, j) => j ? p :
    { ...p, robustNormRechecks: metadata }) });
}
