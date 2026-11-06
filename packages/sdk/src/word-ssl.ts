import { seamWordAgreement, seamWordCompetitors, seamWordFit } from "@tilawa/core";
import type { DecodedPhoneme, PhonemeEvidence, PhraseAsset, ReferenceIndex, SeamWordAgreementEvidence } from "@tilawa/core";
import type { LoadedModel } from "./model.js";
import { wordEarCandidates, wordEarSpans } from "./word-ear.js";

/** Whole proposed identities on fixed native SSL intervals. Confidence and
 * primary token ownership never enter the closed rank comparison. */
export function confirmedWordInterval(said: string, phones: readonly DecodedPhoneme[], rivals: readonly string[]) {
  let best: { cost: number; hard: number; rival: number; nativeFrom: number; nativeTo: number } | undefined;
  for (let a = 0; a < phones.length; a++) for (let b = a + 1; b <= phones.length; b++) {
    const observed = phones.slice(a, b).map(p => p.p).join("");
    if (seamWordFit(said, observed, 1).hard > 1) continue;
    const fit = seamWordAgreement(said, observed, rivals);
    if (!fit.agrees || (best && best.cost <= fit.cost)) continue;
    best = { cost: fit.cost, hard: fit.hard, rival: fit.rival,
      nativeFrom: phones[a].start, nativeTo: phones[b - 1].end + 1 };
  }
  return best;
}

export async function addWordSslEvidence(model: LoadedModel, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset): Promise<PhonemeEvidence[]> {
  if (!model.wordEar || !model.sslEar || !audio.length || !chunks.length) return chunks;
  if (model.manifest.sampleRate !== 16000) throw new Error("Word/SSL agreement requires 16 kHz audio");
  const frameMs = chunks[0].frameMs, spans = wordEarSpans(chunks, index, phrases);
  const proposed: import("@tilawa/core").SeamInsertionCandidate[] = [];
  for (const span of spans) {
    const from = Math.round(span.fromFrame * frameMs * 16), to = Math.min(audio.length, Math.round(span.toFrame * frameMs * 16));
    if (to <= from) continue;
    const heard = await model.wordEar.decode(audio.subarray(from, to));
    for (const candidate of wordEarCandidates(index, span, heard, phrases)) proposed.push(candidate);
  }
  if (!proposed.length) return chunks;
  const phones = await model.sslEar.decodeSeams(audio, proposed.map(c => ({
    fromFrame: Math.round(c.fromFrame * frameMs / 20), toFrame: Math.round(c.toFrame * frameMs / 20),
  })));
  const metadata: SeamWordAgreementEvidence[] = [];
  for (const c of proposed) {
    const kept = phones.filter(p => p.start * 20 >= c.fromFrame * frameMs && (p.end + 1) * 20 <= c.toFrame * frameMs);
    const hit = confirmedWordInterval(c.said, kept, seamWordCompetitors(index, c.word));
    if (!hit) continue;
    const w = index.words[c.word];
    metadata.push({ word: { surah: w.surah, ayah: w.ayah, word: w.word }, observed: c.said,
      fromFrame: Math.floor(hit.nativeFrom * 20 / frameMs), toFrame: Math.ceil(hit.nativeTo * 20 / frameMs),
      cost: hit.cost, hard: hit.hard, rival: hit.rival });
  }
  if (!metadata.length) return chunks;
  const at = chunks.findIndex(c => c.phonemes.length);
  if (at < 0) return chunks;
  return chunks.map((c, i) => i !== at ? c : { ...c, phonemes: c.phonemes.map((p, j) => j ? p :
    { ...p, seamWordAgreements: [...(p.seamWordAgreements ?? []), ...metadata] }) });
}
