import { expect, it } from "vitest";
import { calibrateLocalization, type LocalizationTrainingExample } from "../src/localization-calibration.js";

it("holds out entire speakers, freezes the omission envelope and ignores release-held-out data", () => {
  const train: LocalizationTrainingExample[] = [
    { speaker: "a", split: "train", omitted: true, information: 2 },
    { speaker: "a", split: "train", omitted: false, information: 5 },
    { speaker: "b", split: "train", omitted: true, information: 3 },
    { speaker: "b", split: "train", omitted: false, information: 4 },
  ];
  const result = calibrateLocalization([...train,
    { speaker: "held-out", split: "eval", omitted: true, information: 100 },
    { speaker: "held-out", split: "eval", omitted: false, information: 0 },
  ]);
  expect(result).toEqual(calibrateLocalization(train));
  expect(result.rejectionNats).toBe(3);
  expect(result.groups).toEqual([
    { speaker: "a", rejectionNats: 3, omissions: 1, falseRescues: 0, spoken: 1, rescues: 1 },
    { speaker: "b", rejectionNats: 2, omissions: 1, falseRescues: 1, spoken: 1, rescues: 1 },
  ]);
  expect(calibrateLocalization(train.map(e => e.omitted ? e : { ...e, information: 0 })).rejectionNats).toBe(3);
});

it("refuses to calibrate without omission negatives across training speakers", () => {
  expect(() => calibrateLocalization([])).toThrow();
  expect(() => calibrateLocalization([
    { speaker: "a", split: "train", omitted: true, information: 1 },
    { speaker: "b", split: "train", omitted: false, information: 2 },
  ])).toThrow("partition");
});
