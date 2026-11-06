import { createTracker, minGapFrames, PosteriorFrames, trackerInput, waqfMarked } from "@tilawa/core";
import type { DecodedPhoneme, EngineEvent, PhonemeEvidence, PhraseAsset, ReferenceIndex, SeamSpellingEvidence } from "@tilawa/core";
import { insertionCells } from "./seam-insertion.js";
import type { LoadedModel } from "./model.js";

export interface OfflineSeam {
  /** First written word after the ayah/inner-waqf boundary. */
  word: number;
  fromFrame: number;
  toFrame: number;
}

/** Label pauses on primary evidence and bound secondary crops with primary
 * tracker steps. No greedy word-fit or auxiliary phones choose these spans. */
function labelledPauses(chunks: PhonemeEvidence[], index: ReferenceIndex): number[] {
  if (!chunks.length || chunks.some(c => !c.posteriors)) return [];
  const clock = chunks[0].frameMs;
  const frames = new PosteriorFrames(index.tokens.size);
  for (const e of chunks) if (e.posteriors) frames.append(e.posteriors);
  const phones = chunks.flatMap(c => c.phonemes);
  const pauses: number[] = [];
  const tokenIds = new Map([...index.tokens].map((p, i) => [p, i + 1]));
  const argmax = (f: number) => {
    const row = frames.row(f);
    let best = 0;
    for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
    return best;
  };
  for (let i = 0; i + 1 < phones.length; i++) {
    let a = phones[i].end + 1;
    const b = phones[i + 1].start, id = tokenIds.get(phones[i].p);
    if (a < frames.startFrame || b > frames.endFrame) continue;
    while (a < b && argmax(a) === id) a++;
    if (b - a < minGapFrames(clock)) continue;
    let blank = true;
    for (let f = a; f < b; f++) if (frames.row(f)[0] < Math.log(.5)) { blank = false; break; }
    if (blank) pauses.push(a);
  }
  return pauses;
}

export function offlineSeams(chunks: PhonemeEvidence[], index: ReferenceIndex, phrases?: PhraseAsset, positions?: EngineEvent[]): OfflineSeam[] {
  const cells = insertionCells(chunks);
  if (!cells.length) return [];
  const clock = cells[0].frameMs, pauses = labelledPauses(chunks,index);
  const lookup = new Map(index.words.map((w, i) => [`${w.surah}:${w.ayah}:${w.word}`, i]));
  const steps: { word: number; frame: number; run: number }[] = [];
  let run = 0;
  const note = (events: EngineEvent[]) => {
    for (const e of events) {
      if (e.type !== "position") continue;
      if (e.state !== "locked" || e.relocated) { run++; continue; }
      // A held position across silence is not a new step or a relocation.
      if (e.hold) continue;
      if (!e.expected) continue;
      const wi = lookup.get(`${e.expected.surah}:${e.expected.ayah}:${e.expected.word}`);
      if (wi !== undefined && (steps.at(-1)?.word !== wi || steps.at(-1)?.run !== run))
        steps.push({ word: wi, frame: Math.round(e.timeMs / clock), run });
    }
  };
  if (positions) note(positions);
  else {
    const tracker = createTracker(index, { mode: "free", phrases });
    for (const c of cells) note(tracker.push(trackerInput(c)));
    note(tracker.end());
  }
  const nextWord = (wi: number) => { do { wi++; } while (wi < index.words.length && index.words[wi].end === index.words[wi].start); return wi; };
  const out: OfflineSeam[] = [];
  for (let i = 1; i + 1 < steps.length; i++) {
    const prev = steps[i-1], one = steps[i];
    if (prev.run !== one.run || nextWord(prev.word) !== one.word) continue;
    const a = index.words[prev.word], b = index.words[one.word];
    if (a.surah !== b.surah || (a.ayah === b.ayah && !waqfMarked(prev.word))) continue;
    const following = nextWord(one.word);
    // Word one may be revisited before the word-two step. Keep the original
    // seam entry; a read-only crop must not inherit a new onset from a backstep.
    let two: typeof one | undefined;
    for (let j = i + 1; j < steps.length; j++) {
      const step = steps[j];
      if (step.run !== one.run || step.word > following) break;
      if (step.word === following) { two = step; break; }
    }
    if (!two) continue;
    for (const pause of pauses) if (pause >= prev.frame && pause < two.frame)
      out.push({ word: one.word, fromFrame: pause, toFrame: two.frame + Math.round(300 / clock) });
  }
  return out;
}

/** Keep whole utterances from the secondary crop as spelling evidence only.
 * Primary token values, posterior rows and tracker inputs remain intact. */
export async function addOfflineEvidence(model: LoadedModel, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset, positions?: EngineEvent[]): Promise<PhonemeEvidence[]> {
  if (!model.offlineStudent || !audio.length) return chunks;
  if (model.manifest.sampleRate !== 16000 || chunks[0]?.frameMs !== 40)
    throw new Error("Offline student requires the 16 kHz / 40 ms evidence clock");
  const metadata: SeamSpellingEvidence[] = [];
  for (const span of offlineSeams(chunks,index,phrases,positions)) {
    const offset = span.fromFrame * 640, limit = Math.min(audio.length,span.toFrame * 640);
    if (limit <= offset) continue;
    const e: PhonemeEvidence = await model.offlineStudent.decode(audio.subarray(offset,limit),index.tokens.size+1);
    if (e.inventory !== index.inventoryId || e.frameMs !== chunks[0].frameMs || !e.posteriors)
      throw new Error("Offline student clock or inventory differs from primary evidence");
    const pauses = labelledPauses([e],index);
    const w = index.words[span.word];
    let utterance: DecodedPhoneme[] = [];
    const save = () => {
      if (utterance.length) metadata.push({ word:{surah:w.surah,ayah:w.ayah,word:w.word}, toFrame:span.toFrame,
        phonemes:utterance.map(p => ({...p,start:p.start+span.fromFrame,end:p.end+span.fromFrame})) });
      utterance = [];
    };
    for (const p of e.phonemes) {
      if (utterance.length && pauses.some(f => f > utterance.at(-1)!.end && f < p.start)) save();
      utterance.push(p);
    }
    save();
  }
  if (!metadata.length) return chunks;
  const at = chunks.findIndex(c => c.phonemes.length);
  return chunks.map((c,i) => i !== at ? c : { ...c, phonemes:c.phonemes.map((p,j) => j ? p :
    {...p,seamSpellings:[...(p.seamSpellings ?? []),...metadata]}) });
}
