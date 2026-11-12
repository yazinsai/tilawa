import { phoneClass } from "@tilawa/core";
import type { PhoneRecording, ReferenceIndex, WordRef } from "@tilawa/core";

export interface SpokenPhoneLabel {
  id: string;
  speaker: string;
  split: string;
  kind: string;
  scored: boolean;
  excluded?: unknown;
  ayahs: string[];
  wav: string;
  events: { errorClass: string; location: WordRef }[];
  unsupported?: { location: WordRef }[];
  non_errors: unknown[];
}

/** Exclude entire non-train speakers, extra mistakes, lexical scripts and repetitions. */
export function spokenPhoneTrainingLabels(all: readonly SpokenPhoneLabel[]): SpokenPhoneLabel[] {
  const excluded = new Set(all.filter(l => l.split !== "train").map(l => l.speaker));
  const lexical = new Set(["wrong_word", "omitted_word", "added_word", "skipped_passage"]);
  return all.filter(l => l.split === "train" && !excluded.has(l.speaker) && l.scored && !l.excluded && !l.non_errors.length
    && !l.events.some(e => lexical.has(e.errorClass))
    && (l.kind === "none" ? !l.events.length && !(l.unsupported?.length)
      : ["vowel", "tajweed"].includes(l.kind) && l.events.length + (l.unsupported?.length ?? 0) > 0));
}

/** Scripted phone errors may vary only at their labelled word; reference ownership stays intact. */
export function spokenPhoneRecording(index: ReferenceIndex, label: SpokenPhoneLabel, heard: string): PhoneRecording {
  const ayahs = new Set(label.ayahs), words = index.words.filter(w => ayahs.has(`${w.surah}:${w.ayah}`));
  if (!words.length || !heard || words.some((w, i) => i > 0 && words[i - 1].end !== w.start)) throw new Error("Empty or non-contiguous training recording");
  const from = words[0].start, to = words.at(-1)!.end, written = index.phonemes.slice(from, to);
  const variable = Array(written.length).fill(false) as boolean[];
  const sites = [...label.events, ...(label.unsupported ?? [])];
  for (const { location } of sites) {
    const w = words.find(w => w.surah === location.surah && w.ayah === location.ayah && w.word === location.word);
    if (!w) throw new Error("Pronunciation label outside training reference");
    for (let i = w.start; i < w.end; i++) if (label.kind === "tajweed" || phoneClass(index.phonemes[i]) === "vowel") variable[i - from] = true;
  }
  if (label.kind !== "none" && !variable.some(Boolean)) throw new Error("Pronunciation label has no supported reference phones");
  return { written, heard, variable };
}
