import type { ReferenceIndex } from "@tilawa/core";

export interface TeacherObservationLabel {
  id: string;
  speaker: string;
  split: string;
  kind: string;
  scored: boolean;
  excluded?: unknown;
  ayahs: string[];
  wav: string;
  events: unknown[];
  unsupported?: unknown[];
  non_errors: unknown[];
}

/** Entire speakers are excluded if any annotation belongs to a non-training split. */
export function teacherObservationTrainingLabels(all: readonly TeacherObservationLabel[]): TeacherObservationLabel[] {
  const excluded = new Set(all.filter(l => l.split !== "train").map(l => l.speaker));
  return all.filter(l => l.split === "train" && !!l.speaker && !excluded.has(l.speaker)
    && l.scored === true && !l.excluded && l.kind === "none"
    && Array.isArray(l.events) && !l.events.length && !(l.unsupported?.length)
    && Array.isArray(l.non_errors) && !l.non_errors.length);
}

/** All takes of a speaker remain in one deterministic, training-only validation partition. */
export function teacherObservationFolds(labels: readonly TeacherObservationLabel[]): Map<string, number> {
  const speakers = [...new Set(labels.map(l => l.speaker))].sort(), folds = Math.min(3, speakers.length);
  if (folds < 2) throw new Error("Observation selection needs multiple training speakers");
  return new Map(speakers.map((s, i) => [s, i % folds]));
}

/** Authorized pronunciation, not a claim that either acoustic decoder is a verified transcription. */
export function teacherObservationReference(index: ReferenceIndex, label: TeacherObservationLabel): string {
  const ayahs = new Set(label.ayahs), words = index.words.filter(w => ayahs.has(`${w.surah}:${w.ayah}`));
  if (!words.length || words.some((w, i) => i > 0 && words[i - 1].end !== w.start)
    || new Set(words.map(w => `${w.surah}:${w.ayah}`)).size !== ayahs.size) throw new Error("Invalid authorized training reference");
  return index.phonemes.slice(words[0].start, words.at(-1)!.end);
}
