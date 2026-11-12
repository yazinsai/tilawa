export interface CopyTrainingLabel {
  id: string; speaker: string; split: string; kind: string; scored: boolean;
  ayahs: string[]; wav: string; events: unknown[]; non_errors: unknown[]; excluded?: unknown;
}

/** A speaker with any non-training annotation is excluded in full, including mixed-split speakers. */
export function copyTrainingLabels(all: readonly CopyTrainingLabel[]): CopyTrainingLabel[] {
  const excluded = new Set(all.filter(l => l.split !== "train").map(l => l.speaker));
  return all.filter(l => l.split === "train" && !excluded.has(l.speaker) && l.kind === "none" && l.scored === true
    && !l.excluded && Array.isArray(l.events) && l.events.length === 0 && Array.isArray(l.non_errors)
    && l.non_errors.length === 0 && typeof l.speaker === "string" && l.speaker.length > 0);
}
export function copySpeakerFolds(labels: readonly CopyTrainingLabel[]): Map<string, number> {
  const speakers = [...new Set(labels.map(l => l.speaker))].sort();
  if (speakers.length < 2) throw new Error("Copy calibration needs at least two eligible training speakers");
  return new Map(speakers.map((s, i) => [s, i % Math.min(3, speakers.length)]));
}
