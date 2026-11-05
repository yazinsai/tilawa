import type { ReferenceIndex } from "./reference.js";
export { SEAM_REPLAYS } from "./seam-replay-data.js";

/** Fixed clean replays, built with the shipped listener and unmodified tracker.
 * Words are the canonical forms returned by the same lexical enumeration.
 * No distance, confidence or learned threshold enters this existence test. */
export interface SeamReplayAsset {
  reference: string;
  inventory: string;
  seams: Record<string, { replays: number; words: string[] }>;
}

/** Text, seam offset and adjacent ayahs identify the replay context. Repeated
 * words in different surrounding text must not share their negative set. */
export function seamTextKey(index: ReferenceIndex, word: number): string {
  const right = index.words[word], left = index.words[word - 1];
  if (!right || !left || left.surah !== right.surah) return "";
  const span = (wi: number) => {
    const w = index.words[wi];
    let a = wi, b = wi;
    while (a > 0 && index.words[a - 1].surah === w.surah && index.words[a - 1].ayah === w.ayah) a--;
    while (b + 1 < index.words.length && index.words[b + 1].surah === w.surah && index.words[b + 1].ayah === w.ayah) b++;
    return { start: index.words[a].start, end: index.words[b].end };
  };
  const l = span(word - 1), r = span(word);
  return JSON.stringify([index.phonemes.slice(l.start, l.end),
    index.phonemes.slice(r.start, r.end), right.start - r.start]);
}

export function replayHasCoverage(asset: SeamReplayAsset, index: ReferenceIndex, word: number): boolean {
  if (asset.reference !== index.version || asset.inventory !== index.inventoryId) return false;
  const entry = asset.seams[seamTextKey(index, word)];
  return !!entry && Number.isInteger(entry.replays) && entry.replays >= 2;
}

export function replayAllows(asset: SeamReplayAsset, index: ReferenceIndex, word: number, spelling: string): boolean {
  return replayHasCoverage(asset, index, word) && !asset.seams[seamTextKey(index, word)].words.includes(spelling);
}

