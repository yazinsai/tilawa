export interface TrainingLabel {
  id: string; speaker: string; split: string; kind: string; ayahs: string[]; wav: string;
  events?: unknown[]; non_errors?: unknown[]; unsupported?: unknown[];
  exclude?: unknown; excluded?: unknown; scored?: boolean;
}

/** A mixed-split speaker is excluded as a unit, including their otherwise eligible takes. */
export function trainingNoneLabels(labels: TrainingLabel[]): TrainingLabel[] {
  const excluded = new Set(labels.filter(r => r.split !== "train").map(r => r.speaker));
  return labels.filter(r => r.split === "train" && !excluded.has(r.speaker) && r.kind === "none"
    && r.id && r.speaker && r.ayahs?.length && r.wav && !r.events?.length && !r.non_errors?.length
    && !r.unsupported?.length && !r.exclude && !r.excluded && r.scored !== false)
    .sort((a, b) => a.id.localeCompare(b.id));
}
