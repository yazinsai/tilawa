import { createChecker, createTracker, trackerInput, PosteriorFrames, PHRASE_SWITCH, seamInsertionMarginal, collapseHeld, lexicalWords } from "../engine/index.js";
import type { PhonemeEvidence, PhraseAsset, ReferenceIndex, SeamInsertionCandidate, SeamInsertionEvidence } from "../engine/index.js";
import type { LoadedModel } from "./model.js";

/** One native frame at a time, extending a cell through a token's entire run.
 * This is an engine evidence clock; no scoring code or labels participate. */
export function insertionCells(chunks: PhonemeEvidence[]): PhonemeEvidence[] {
  if (!chunks.length || chunks.some(c => !c.posteriors)) return [];
  const phones = chunks.flatMap(c => c.phonemes), k = chunks[0].posteriors!.k;
  const ids = chunks.flatMap(c => Array.from(c.posteriors!.ids)), probs = chunks.flatMap(c => Array.from(c.posteriors!.probs));
  const first = chunks[0].posteriors!.startFrame, end = first + ids.length / k;
  const out: PhonemeEvidence[] = [];
  let p = 0;
  for (let a = first; a < end;) {
    let b = a + 1; const phonemes = [];
    while (p < phones.length && phones[p].start < b) { phonemes.push(phones[p]); b = Math.max(b, phones[p++].end + 1); }
    b = Math.min(b, end);
    out.push({ inventory: chunks[0].inventory, frameMs: chunks[0].frameMs, phonemes, decodedTo: b,
      posteriors: { startFrame: a, nFrames: b - a, k, ids: ids.slice((a-first)*k,(b-first)*k), probs: probs.slice((a-first)*k,(b-first)*k) } });
    a = b;
  }
  return out;
}

/** Auxiliary metadata is carried by the first primary token, with all primary values intact. */
export function carryInsertionEvidence(chunks: PhonemeEvidence[], metadata: SeamInsertionEvidence[]): PhonemeEvidence[] {
  if (!metadata.length) return chunks;
  const at = chunks.findIndex(c => c.phonemes.length);
  return chunks.map((c,i) => i !== at ? c : { ...c, phonemes: c.phonemes.map((p,j) => j ? p : { ...p, seamInsertions: metadata }) });
}

/** Read-only candidate discovery and warm teacher judgment. The tracker consumes only primary tokens. */
export async function addInsertionEvidence(model: LoadedModel, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset): Promise<PhonemeEvidence[]> {
  if (!model.insertionTeacher || !audio.length) return chunks;
  if (model.manifest.sampleRate !== 16000) throw new Error("Insertion teacher requires 16 kHz audio");
  const cells = insertionCells(chunks);
  if (!cells.length) return chunks;
  const candidates: SeamInsertionCandidate[] = [];
  const seen = new Set<string>();
  const tracker = createTracker(index, { mode: "free", phrases });
  const checker = createChecker(index, { phrases, seamInsertion(c) {
    const key = JSON.stringify(c);
    if (!seen.has(key)) { seen.add(key); candidates.push(c); }
    // Discovery runs the unchanged generator, including its issue state.
    return true;
  } });
  for (const e of cells) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const lex = lexicalWords(index);
  const frameMs = cells[0].frameMs, ids = new Map([...index.tokens].map((p,i) => [p,i+1]));
  const forms = (wi: number) => [...new Set([index.phonemes.slice(index.words[wi].start,index.words[wi].end),
    index.isolatedWordForms?.[wi] ?? ""].filter(Boolean).map(collapse))];
  const cache = new Map<string, { frames: PosteriorFrames; shift: number }>();
  const metadata: SeamInsertionEvidence[] = [];
  for (const c of candidates) {
    const prev = previousWord(index,c.word), prev2 = previousWord(index,prev), next = followingWord(index,c.word);
    if (prev < 0 || next >= index.words.length) continue;
    const before = prev2 >= 0 && index.words[prev2].ayah === index.words[prev].ayah &&
      index.words[prev2].surah === index.words[prev].surah ? [prev2,prev] : [prev];
    // A whole two-word prefix needs its actual onset; no tight candidate-frame fallback.
    if (before.length === 2 && c.prefixFrame === undefined) continue;
    const from = c.prefixFrame ?? c.seamFromFrame;
    const start = Math.max(0, from - Math.round(4000/frameMs));
    const end = c.seamToFrame + Math.round(300/frameMs), key = `${start}:${end}`;
    let crop = cache.get(key);
    if (!crop) {
      const offset = Math.round(start*frameMs*16), limit = Math.min(audio.length,Math.round(end*frameMs*16));
      if (limit <= offset) continue;
      const e = await model.insertionTeacher.decode(audio.subarray(offset,limit),index.tokens.size+1);
      if (e.inventory !== index.inventoryId || e.frameMs !== frameMs || !e.posteriors) throw new Error("Insertion teacher clock or inventory differs from primary evidence");
      const frames = new PosteriorFrames(index.tokens.size);
      frames.append({ ...e.posteriors, startFrame: e.posteriors.startFrame + start });
      crop = { frames, shift: start }; cache.set(key,crop);
    }
    const map = (ws: number[]) => ws.map(w => forms(w).map(f => [...f].map(p => ids.get(p)!)));
    // The candidate's exact soft-only spelling is already certified by the generator.
    const spelling = lex.surplusWord(c.word, c.said);
    if (!spelling) continue;
    const candidateForms = new Set([spelling]);
    const norm = (s: string) => collapseHeld([...s].filter(p => !"ڇؙۜ".includes(p))).join("");
    for (const list of index.wordForms ?? []) for (const f of list)
      if (norm(f) === spelling) candidateForms.add(collapse(f));
    const candidate = [...candidateForms].map(f => [...f].map(p => ids.get(p)!));
    const marginal = seamInsertionMarginal(crop.frames,map(before),candidate,map([c.word,next]),from,
      Math.min(end,crop.frames.endFrame),PHRASE_SWITCH);
    if (marginal !== null) metadata.push({ word: { surah:index.words[c.word].surah, ayah:index.words[c.word].ayah, word:index.words[c.word].word },
      observed:c.said,fromFrame:c.fromFrame,toFrame:c.toFrame,marginal });
  }
  return carryInsertionEvidence(chunks,metadata);
}

// Same held-sound collapse as the core's spelling comparison; ordinary doubled consonants stay doubled.
function collapse(s: string): string {
  return collapseHeld([...s]).join("");
}
function previousWord(index: ReferenceIndex, at: number): number { do { at--; } while (at >= 0 && index.words[at].end === index.words[at].start); return at; }
function followingWord(index: ReferenceIndex, at: number): number { do { at++; } while (at < index.words.length && index.words[at].end === index.words[at].start); return at; }
