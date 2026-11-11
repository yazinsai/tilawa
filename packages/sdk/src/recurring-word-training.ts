export interface RecurringWordLabel {
  id: string;
  speaker: string;
  split: string;
  kind: string;
  scored: boolean;
  wav: string;
  events: unknown[];
  non_errors: unknown[];
  excluded?: unknown;
}

/** Any non-training take excludes its speaker, including that speaker's otherwise eligible takes. */
export function recurringWordTrainingLabels(all: readonly RecurringWordLabel[]): RecurringWordLabel[] {
  const excludedSpeakers = new Set(all.filter(x => x.split !== "train").map(x => x.speaker));
  const benignRepeat = (n: unknown) => !!n && typeof n === "object" && "kind" in n
    && (n.kind === "repeat" || n.kind === "restart");
  return all.filter(x => x.split === "train" && !!x.speaker && !!x.id && !!x.wav && !excludedSpeakers.has(x.speaker)
    && (x.kind === "none" || x.kind === "repeat") && x.scored === true && !x.excluded
    && Array.isArray(x.events) && x.events.length === 0 && Array.isArray(x.non_errors) && x.non_errors.every(benignRepeat));
}
