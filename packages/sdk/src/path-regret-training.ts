/** Eligibility is explicit; never infer training authorization from a recording's name. */
export interface RegretTrainingLabel {
  id: string; speaker: string; split: string; kind: string; scored: boolean;
  ayahs: string[]; wav: string; events: unknown[]; non_errors: unknown[]; excluded?: unknown;
}

export function regretTrainingLabels(all: readonly RegretTrainingLabel[]): RegretTrainingLabel[] {
  const heldOut = new Set(all.filter(l => l.split !== "train").map(l => l.speaker));
  return all.filter(l => l.split === "train" && !heldOut.has(l.speaker) && l.kind === "none"
    && l.scored === true && !l.excluded && Array.isArray(l.events) && l.events.length === 0
    && Array.isArray(l.non_errors) && l.non_errors.length === 0);
}

/** Up to five disjoint speaker partitions. Every fitted partition is retained (100% empirical coverage). */
export function regretSpeakerPartitions(labels: readonly Pick<RegretTrainingLabel, "speaker">[]): Map<string, number> {
  const speakers = [...new Set(labels.map(l => l.speaker))].sort();
  if (speakers.length < 2) throw new Error("At least two authorized training speakers required");
  return new Map(speakers.map((s, i) => [s, i % Math.min(5, speakers.length)]));
}
