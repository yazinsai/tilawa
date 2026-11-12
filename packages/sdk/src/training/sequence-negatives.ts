/** Training-only selection for a frozen phone-head refit. No runtime veto. */
import { collapseHeld, lexicalWords } from "@tilawa/core";
import type { CheckerEvent, DecodedPhoneme, ReferenceIndex } from "@tilawa/core";
import { Resampler } from "../audio.js";
import { Fbank } from "../fbank.js";

export interface TrainingTake {
  id: string;
  speaker: string;
  split: string;
  kind: string;
  ayahs: string[];
  wav: string;
  scored?: boolean;
  exclude?: boolean;
  excluded?: boolean;
  events?: unknown[];
  non_errors?: unknown[];
  unsupported?: unknown[];
}

/** Exclude an entire speaker if any of their takes is outside training. */
export function trainingTakes(takes: readonly TrainingTake[]): TrainingTake[] {
  if (takes.some(t => !t.speaker || !t.id || !t.split || !t.wav || !t.ayahs?.length))
    throw new Error("Missing training provenance");
  if (new Set(takes.map(t => t.id)).size !== takes.length) throw new Error("Duplicate training recording id");
  const excluded = new Set(takes.filter(t => t.split !== "train" || t.exclude || t.excluded).map(t => t.speaker));
  return takes.filter(t => t.split === "train" && !excluded.has(t.speaker));
}

/** The deployed listener's frontend, including the final resampler flush. */
export function trainingFeatures(samples: Float32Array, sampleRate: number): Float32Array[] {
  const bank = new Fbank();
  const resampler = sampleRate === 16000 ? null : new Resampler(sampleRate, 16000);
  const frames: Float32Array[] = [];
  const step = Math.round(sampleRate * .32);
  if (step < 1 || !Number.isFinite(sampleRate)) throw new Error("Invalid sample rate");
  for (let at = 0; at < samples.length; at += step) {
    const part = samples.subarray(at, at + step);
    frames.push(...bank.push(resampler ? resampler.push(part) : part));
  }
  if (resampler) frames.push(...bank.push(resampler.flush()));
  return frames;
}

export function hasTranscript(take: TrainingTake): boolean {
  return take.kind === "none" && take.scored !== false &&
    ![take.events, take.non_errors, take.unsupported].some(a => a?.length);
}

const pronunciation = (s: string) => collapseHeld([...s]).join("");

export interface SequenceNegative {
  recording: string;
  speaker: string;
  word: number;
  from: number;
  to: number;
  written: string;
  heard: string;
  /** Independent fit-speaker takes containing the negatively targeted pronunciation. */
  positives: string[];
}

/**
 * Select a complete observed sequence, never a list of offending phone classes.
 * The unique contiguous token witness gives its original acoustic coordinates.
 * This runs after the unchanged engine has finished each clean recording.
 */
export function mineSequenceNegatives(
  index: ReferenceIndex, take: TrainingTake, warnings: readonly CheckerEvent[],
  tokens: readonly DecodedPhoneme[], positiveTakes: readonly TrainingTake[],
): SequenceNegative[] {
  if (trainingTakes([take]).length !== 1 || !hasTranscript(take)) return [];
  positiveTakes = trainingTakes(positiveTakes);
  const lawful = lexicalWords(index);
  const result: SequenceNegative[] = [];
  for (const warning of warnings) {
    if (warning.state !== "confirmed" || warning.errorClass !== "wrong_word") continue;
    const word = index.words.findIndex(w => w.surah === warning.location.surah &&
      w.ayah === warning.location.ayah && w.word === warning.location.word);
    if (word < 0 || !take.ayahs.includes(`${warning.location.surah}:${warning.location.ayah}`)) continue;
    const heard = warning.observed;
    if (!heard || [...heard].some(p => !index.tokens.has(p)) || lawful.matches(word, heard)) continue;
    const written = index.phonemes.slice(index.words[word].start, index.words[word].end);
    if (pronunciation(written) === pronunciation(heard)) continue;
    const matches: { from: number; to: number }[] = [];
    for (let a = 0; a < tokens.length; a++) {
      let sequence = "";
      for (let b = a; b < tokens.length && sequence.length < heard.length; b++) {
        sequence += tokens[b].p;
        if (sequence === heard) matches.push({ from: tokens[a].start, to: tokens[b].end + 1 });
      }
    }
    if (matches.length !== 1 || matches[0].from < 0 || matches[0].to <= matches[0].from) continue;
    const positives = positiveTakes.filter(p => p.id !== take.id && hasTranscript(p) && index.words.some(w =>
      p.ayahs.includes(`${w.surah}:${w.ayah}`) &&
        pronunciation(index.phonemes.slice(w.start, w.end)) === pronunciation(heard),
    )).map(p => p.id);
    if (!positives.length) continue;
    if (!result.some(n => n.from === matches[0].from && n.to === matches[0].to && n.heard === heard))
      result.push({ recording: take.id, speaker: take.speaker, word, ...matches[0], written, heard, positives });
  }
  return result;
}
