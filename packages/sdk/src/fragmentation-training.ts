/** Training eligibility excludes every speaker with any non-training annotation. */
export interface FragmentationLabel {
  id: string;
  speaker: string;
  kind: string;
  split: string;
  scored: boolean;
  ayahs: string[];
  wav: string;
  events: unknown[];
  non_errors: unknown[];
  excluded?: unknown;
}
export function fragmentationTrainingLabels(all: readonly FragmentationLabel[]): FragmentationLabel[] {
  const excluded = new Set(all.filter(l => l.split !== "train").map(l => l.speaker));
  return all.filter(l => l.split === "train" && !excluded.has(l.speaker) && l.kind === "none"
    && l.scored && !l.excluded && l.events.length === 0 && l.non_errors.length === 0);
}
export function fragmentationSpeakerFolds(labels: readonly FragmentationLabel[]): Map<string, number> {
  const speakers = [...new Set(labels.map(l => l.speaker))].sort();
  if (speakers.length < 2) throw new Error("Need at least two eligible training speakers");
  return new Map(speakers.map((s, i) => [s, i % Math.min(5, speakers.length)]));
}
