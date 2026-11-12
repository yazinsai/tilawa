/** A recording's path or filename never authorizes its use for calibration. */
export interface RestartTrainingLabel {
  id: string; speaker: string; split: string; kind: string; scored: boolean;
  ayahs: string[]; wav: string; events: unknown[]; non_errors: unknown[]; excluded?: unknown;
}
export function restartTrainingLabels(all: readonly RestartTrainingLabel[]): RestartTrainingLabel[] {
  const excluded = new Set(all.filter(l => l.split !== "train").map(l => l.speaker));
  return all.filter(l => l.split === "train" && !excluded.has(l.speaker) && l.kind === "none"
    && l.scored && !l.excluded && l.events.length === 0 && l.non_errors.length === 0);
}
export function restartSpeakerFolds(labels: readonly RestartTrainingLabel[]): Map<string, number> {
  const speakers = [...new Set(labels.map(l => l.speaker))].sort();
  if (speakers.length < 2) throw new Error("At least two eligible training speakers are required");
  return new Map(speakers.map((s, i) => [s, i % Math.min(5, speakers.length)]));
}
