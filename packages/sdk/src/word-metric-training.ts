/** Explicit training authorization, including exclusion of speakers with any held-out row. */
export interface WordTrainingLabel {
  id: string;
  speaker: string;
  split: string;
  kind: string;
  scored: boolean;
  ayahs: string[];
  wav: string;
  events: unknown[];
  non_errors: unknown[];
  excluded?: unknown;
}

export function wordTrainingLabels(all: readonly WordTrainingLabel[]): WordTrainingLabel[] {
  const heldOut = new Set(all.filter(l => l.split !== "train").map(l => l.speaker));
  return all.filter(l => l.split === "train" && !heldOut.has(l.speaker) && l.kind === "none"
    && l.scored === true && !l.excluded && Array.isArray(l.events) && l.events.length === 0
    && Array.isArray(l.non_errors) && l.non_errors.length === 0);
}

export function wordSpeakerFolds(labels: readonly WordTrainingLabel[]): Map<string, number> {
  const speakers = [...new Set(labels.map(l => l.speaker))].sort();
  if (speakers.length < 2) throw new Error("Word metric calibration needs at least two training speakers");
  return new Map(speakers.map((speaker, i) => [speaker, i % Math.min(5, speakers.length)]));
}
