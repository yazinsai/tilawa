import { hypothesis } from "./ayah-ledger.js";
import { PosteriorFrames } from "./ctc.js";

/** a, i, u, ā, ī, ū. Every other inventory symbol, including blank, is the projected blank. */
const VOWELS = "َُِاۦۥ";
const vowelId = (phone: string) => VOWELS.indexOf(phone) + 1;

/** Sum probabilities, not log probabilities; the source rows and their reconstruction stay untouched. */
export function projectVowels(frames: PosteriorFrames, inventory: readonly string[], from: number, to: number): PosteriorFrames | null {
  if (!Number.isInteger(from) || !Number.isInteger(to) || frames.startFrame < 0 ||
    from < frames.startFrame || to > frames.endFrame || to <= from) return null;
  const classes = [0, ...inventory.map(vowelId)];
  const ids: number[] = [], probs: number[] = [];
  const k = VOWELS.length + 1;
  for (let f = from; f < to; f++) {
    const row = frames.row(f);
    const mass = new Float64Array(k);
    for (let i = 0; i < row.length; i++) mass[classes[i]] += Math.exp(row[i]);
    for (let i = 0; i < k; i++) { ids.push(i); probs.push(mass[i]); }
  }
  const projected = new PosteriorFrames(VOWELS.length);
  projected.append({ k, startFrame: from, nFrames: to - from, ids, probs });
  return projected;
}

/** Closed with-word versus without-word fit, preserving the ledger's word and ayah repeat arcs. */
export function vowelPresenceMargin(
  frames: PosteriorFrames,
  inventory: readonly string[],
  ayahs: readonly (readonly string[])[],
  drop: { ayah: number; word: number },
  from: number,
  to: number,
): number | null {
  const vowels = ayahs.map(words => words.map(word => [...word].filter(p => vowelId(p) > 0).join("")));
  if (!vowels[drop.ayah]?.[drop.word]) return null;
  const read = hypothesis(vowels, vowelId);
  const omit = hypothesis(vowels.map((words, a) => a === drop.ayah ? words.filter((_, w) => w !== drop.word) : words), vowelId);
  if (!omit.ids.length) return null;
  const projected = projectVowels(frames, inventory, from, to);
  if (!projected) return null;
  const readNll = projected.nll(read.ids, from, to, read.repeats);
  const omitNll = projected.nll(omit.ids, from, to, omit.repeats);
  return Number.isFinite(readNll) && Number.isFinite(omitNll) ? omitNll - readNll : null;
}
