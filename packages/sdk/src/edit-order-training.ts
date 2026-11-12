/** Eligibility is explicit in help-train labels, never guessed from a recording's path. */
export interface EditOrderLabel {
  id: string; speaker: string; split: string; kind: string; scored: boolean;
  ayahs: string[]; wav: string; events: unknown[]; non_errors: unknown[]; excluded?: unknown;
}

export function editOrderTrainingLabels(labels: readonly EditOrderLabel[]): EditOrderLabel[] {
  const excludedSpeakers = new Set(labels.filter(l => l.split !== "train").map(l => l.speaker));
  return labels.filter(l => l.split === "train" && !excludedSpeakers.has(l.speaker) && l.kind === "none"
    && l.scored && !l.excluded && l.events.length === 0 && l.non_errors.length === 0);
}

export function editOrderSpeakerFolds(labels: readonly Pick<EditOrderLabel, "speaker">[]): Map<string, number> {
  const speakers = [...new Set(labels.map(l => l.speaker))].sort();
  if (speakers.length < 3) throw new Error("Edit-order fitting requires at least three training speakers");
  return new Map(speakers.map((s, i) => [s, i % Math.min(5, speakers.length)]));
}
