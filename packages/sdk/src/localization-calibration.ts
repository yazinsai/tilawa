/** Weak scripted omission labels and none-labelled words from training speakers only. */
export interface LocalizationTrainingExample {
  speaker: string;
  split: "train" | "eval";
  omitted: boolean;
  information: number;
}

export interface LocalizationCalibration {
  rejectionNats: number;
  omissions: number;
  spoken: number;
  speakers: number;
  /** Leave-one-speaker-out results before pooling the frozen rejection envelope. */
  groups: { speaker: string; rejectionNats: number; omissions: number; falseRescues: number; spoken: number; rescues: number }[];
}

/**
 * Fit the conservative veto point without practice data or release-held-out speakers.
 * Each training speaker is held out intact, using all other training speakers' omission maximum
 * as its rejection point. Pool those training points and the held-out omission maxima to freeze
 * an envelope: a veto must be strictly above every training-only omitted example. None-labelled
 * words measure reach; they never lower the rejection point. Omission labels remain weak labels.
 */
export function calibrateLocalization(examples: readonly LocalizationTrainingExample[]): LocalizationCalibration {
  const train = examples.filter(e => e.split === "train");
  if (train.some(e => !e.speaker || !Number.isFinite(e.information) || e.information < 0))
    throw new Error("Invalid training localization example");
  const speakers = [...new Set(train.map(e => e.speaker))].sort();
  if (speakers.length < 2 || !train.some(e => e.omitted) || !train.some(e => !e.omitted))
    throw new Error("Localization calibration needs omissions, none-labelled words and multiple training speakers");
  const groups = speakers.map(speaker => {
    const fitting = train.filter(e => e.speaker !== speaker && e.omitted);
    if (!fitting.length) throw new Error("A speaker partition has no training omissions");
    const rejectionNats = Math.max(...fitting.map(e => e.information));
    const held = train.filter(e => e.speaker === speaker);
    return { speaker, rejectionNats,
      omissions: held.filter(e => e.omitted).length,
      falseRescues: held.filter(e => e.omitted && e.information > rejectionNats).length,
      spoken: held.filter(e => !e.omitted).length,
      rescues: held.filter(e => !e.omitted && e.information > rejectionNats).length };
  });
  return { rejectionNats: Math.max(...train.filter(e => e.omitted).map(e => e.information)),
    omissions: train.filter(e => e.omitted).length, spoken: train.filter(e => !e.omitted).length,
    speakers: speakers.length, groups };
}
