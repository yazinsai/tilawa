import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { OmissionReadCandidate, PhonemeEvidence, PhraseAsset, ReferenceIndex, SegmentPresenceEvidence } from "@tilawa/core";
import type { LoadedModel } from "./model.js";
import { insertionCells } from "./seam-insertion.js";
import { segmentReadMargin } from "./segment-presence.js";
import type { SegmentPresenceModel } from "./segment-presence.js";

export function omissionReadCertificate(scorer: SegmentPresenceModel, features: ArrayLike<number>[],
  index: ReferenceIndex, c: OmissionReadCandidate): SegmentPresenceEvidence | null {
  if (c.fromFrame<0 || c.toFrame>features.length || c.fromFrame>=c.toFrame) return null;
  const word=index.words[c.word];
  if (!word) return null;
  const ayah=index.words.map((w,i)=>({w,i})).filter(({w})=>w.surah===word.surah && w.ayah===word.ayah && w.end>w.start);
  const result=segmentReadMargin(scorer,features.slice(c.fromFrame,c.toFrame),
    ayah.map(({w})=>index.phonemes.slice(w.start,w.end)),ayah.findIndex(({i})=>i===c.word));
  return result && result.margin>0 ? {word:{surah:word.surah,ayah:word.ayah,word:word.word},
    fromFrame:c.fromFrame,toFrame:c.toFrame,read:result.read,omit:result.omit} : null;
}

/** Score only existing single-word proposals on the checker's unchanged ayah
 * intervals. Primary audio inference, tokens, posteriors and tracker events
 * remain intact. No labels or scoring infrastructure enter discovery. */
export async function addSegmentPresenceEvidence(model: LoadedModel, audio: Float32Array,
  chunks: PhonemeEvidence[], index: ReferenceIndex, phrases?: PhraseAsset): Promise<PhonemeEvidence[]> {
  if (!model.segmentPresence || !chunks.length || !audio.length) return chunks;
  const scorer = model.segmentPresence.model;
  if (scorer.inventory !== index.inventoryId || scorer.tokens !== [...index.tokens].join("")) throw new Error("Segment scorer inventory differs from primary evidence");
  const cells = insertionCells(chunks);
  if (!cells.length) return chunks;
  const candidates: OmissionReadCandidate[] = [];
  const seen = new Set<string>();
  const tracker = createTracker(index, { mode: "free", phrases });
  const checker = createChecker(index, { phrases, omissionRead(c) {
    const key = JSON.stringify(c);
    if (!seen.has(key)) { candidates.push(c); seen.add(key); }
    return false;
  } });
  for (const e of cells) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  if (!candidates.length) return chunks;
  const features = await model.segmentPresence.features(audio);
  const metadata: SegmentPresenceEvidence[] = [];
  for (const c of candidates) {
    const result=omissionReadCertificate(scorer,features,index,c);
    if (result) metadata.push(result);
  }
  if (!metadata.length) return chunks;
  const at=chunks.findIndex(c=>c.phonemes.length);
  return chunks.map((c,i)=>i!==at?c:{...c,phonemes:c.phonemes.map((p,j)=>j?p:{...p,segmentPresence:metadata})});
}
