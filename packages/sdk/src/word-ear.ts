import { createTracker, trackerInput, PosteriorFrames, minGapFrames, waqfMarked } from "@tilawa/core";
import type { PhonemeEvidence, ReferenceIndex, PhraseAsset } from "@tilawa/core";
import { insertionCells } from "./seam-insertion.js";
import { lexicalWords, collapseHeld } from "@tilawa/core";
import type { SeamInsertionCandidate } from "@tilawa/core";
import { normalizeWord, type HeardWord } from "./word-decoder.js";

export interface WordEarSpan {
  /** Word immediately after the ayah or marked inner stop. */
  word: number;
  fromFrame: number;
  toFrame: number;
  seamFromFrame: number;
  seamToFrame: number;
  prefixFrame?: number;
}

/** Read-only seam clock. No greedy word anchor or reference pronunciation fit. */
export function wordEarSpans(chunks: PhonemeEvidence[], index: ReferenceIndex, phrases?: PhraseAsset): WordEarSpan[] {
  const cells = insertionCells(chunks);
  if (!cells.length) return [];
  const frameMs = cells[0].frameMs;
  const frames = new PosteriorFrames(index.tokens.size);
  for (const c of chunks) frames.append(c.posteriors!);
  const steps: { word: number; frame: number }[] = [];
  const positions = new Map(index.words.map((w, i) => [`${w.surah}:${w.ayah}:${w.word}`, i]));
  const tracker = createTracker(index, { mode: "free", phrases });
  for (const c of cells) for (const e of tracker.push(trackerInput(c))) {
    if (e.type !== "position" || e.state !== "locked" || !e.observed.length) continue;
    const w = e.observed[0].word;
    const word = positions.get(`${w.surah}:${w.ayah}:${w.word}`);
    if (word !== undefined && steps.at(-1)?.word !== word) steps.push({ word, frame: Math.round(e.timeMs/frameMs) });
  }
  const phones = chunks.flatMap(c => c.phonemes);
  const tokenIds = new Map([...index.tokens].map((p, i) => [p, i+1]));
  const pauses: { from: number; to: number }[] = [];
  for (let i = 1; i < phones.length; i++) {
    let from = phones[i-1].end+1;
    const to = phones[i].start, id = tokenIds.get(phones[i-1].p);
    while (from < to && from >= frames.startFrame && to <= frames.endFrame) {
      const row = frames.row(from);
      let best = 0;
      for (let k = 1; k < row.length; k++) if (row[k] > row[best]) best = k;
      if (best !== id) break;
      from++;
    }
    if (to-from < minGapFrames(frameMs) || from < frames.startFrame || to > frames.endFrame) continue;
    let blank = true;
    for (let f = from; f < to; f++) if (frames.row(f)[0] < Math.log(.5)) { blank = false; break; }
    if (blank) pauses.push({ from, to });
  }
  const spans: WordEarSpan[] = [];
  for (let i = 1; i+1 < steps.length; i++) {
    const previous = steps[i-1], first = steps[i];
    const second = steps.slice(i+1).find(s => s.word === first.word+1);
    if (first.word !== previous.word+1 || !second) continue;
    const p = index.words[previous.word], w = index.words[first.word];
    const boundary = p.surah !== w.surah || p.ayah !== w.ayah || waqfMarked(previous.word);
    if (!boundary) continue;
    for (const pause of pauses) {
      if (pause.from < previous.frame || pause.from >= first.frame || pause.to > second.frame) continue;
      spans.push({ word: first.word, fromFrame: pause.from,
        toFrame: second.frame+Math.round(300/frameMs), seamFromFrame: previous.frame,
        seamToFrame: second.frame, prefixFrame: steps[i-2]?.word === previous.word-1 ? steps[i-2].frame : undefined });
    }
  }
  return spans;
}

/** Sequence edit diff: matched written words stay in order; unmatched heard words
 * are proposals only. No autoregressive log-probability participates. */
export function extraWords(expected: string[], heard: HeardWord[]): string[] {
  const a = expected.map(normalizeWord), b = heard.map(w => normalizeWord(w.text));
  const lengths = Array.from({ length: a.length+1 }, () => new Uint32Array(b.length+1));
  for (let i = a.length-1; i >= 0; i--) for (let j = b.length-1; j >= 0; j--)
    lengths[i][j] = a[i] === b[j] ? lengths[i+1][j+1]+1 : Math.max(lengths[i+1][j], lengths[i][j+1]);
  const extras: string[] = [];
  let i = 0, j = 0;
  while (j < b.length) {
    if (i < a.length && a[i] === b[j]) { i++; j++; }
    else if (i < a.length && lengths[i+1][j] > lengths[i][j+1]) i++;
    else extras.push(b[j++]);
  }
  return extras;
}

export function wordEarCandidates(index: ReferenceIndex, span: WordEarSpan, heard: HeardWord[],
  phrases?: PhraseAsset): SeamInsertionCandidate[] {
  if (!index.wordText) return [];
  const lex = lexicalWords(index), extras = extraWords(index.wordText.slice(span.word,span.word+2), heard);
  const formulaWords = new Set((phrases?.phrases ?? []).flatMap(p => p.text.split(/\s+/).map(normalizeWord)));
  const out: SeamInsertionCandidate[] = [], seen = new Set<string>();
  for (const text of extras) {
    if (!text || formulaWords.has(text)) continue;
    for (let wi = 0; wi < index.words.length; wi++) {
      if (normalizeWord(index.wordText[wi]) !== text) continue;
      const forms = index.wordForms?.[wi] ?? [index.phonemes.slice(index.words[wi].start,index.words[wi].end)];
      for (const raw of forms) {
        const said = collapseHeld([...raw].filter(p => !"ڇؙۜ".includes(p))).join("");
        if (!said || seen.has(said) || !lex.surplusWord(span.word,said) || lex.fragmentOfSeam(span.word,said)) continue;
        seen.add(said);
        out.push({ word: span.word, said, fromFrame: span.fromFrame, toFrame: span.toFrame,
          seamFromFrame: span.seamFromFrame, seamToFrame: span.seamToFrame, prefixFrame: span.prefixFrame });
      }
    }
  }
  return out;
}
