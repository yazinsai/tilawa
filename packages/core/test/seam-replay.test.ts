import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "../src/reference.js";
import { replayAllows, seamTextKey, type SeamReplayAsset } from "../src/seam-replay.js";

const index = loadReferenceIndex({ version: "replay-fixture", inventory: { id: "phones", tokens: "بكلم" },
  ayahs: [
    { s: 1, a: 1, w: ["ب", "ك"], p: "بك", o: [0, 1], iso: ["ب", "ك"] },
    { s: 1, a: 2, w: ["ل", "م"], p: "لم", o: [0, 1], iso: ["ل", "م"] },
    { s: 1, a: 3, w: ["ل", "م"], p: "لم", o: [0, 1], iso: ["ل", "م"] },
  ] });
const asset = (replays: number, words: string[] = []): SeamReplayAsset => ({
  reference: index.version, inventory: index.inventoryId,
  seams: { [seamTextKey(index, 2)]: { replays, words } },
});

describe("clean same-text replay existence veto", () => {
  it("abstains on absent, one-replay or invalid coverage", () => {
    for (const n of [0, 1, 2.5, NaN]) expect(replayAllows(asset(n), index, 2, "بم")).toBe(false);
    expect(replayAllows(asset(2), index, 4, "بم")).toBe(false);
  });
  it("permits an absent spelling with two clean replays", () => {
    expect(replayAllows(asset(2, ["كم"]), index, 2, "بم")).toBe(true);
  });
  it("one matching clean spelling suffices to veto, regardless of the replay count", () => {
    expect(replayAllows(asset(2, ["بم"]), index, 2, "بم")).toBe(false);
    expect(replayAllows(asset(100, ["كم", "بم"]), index, 2, "بم")).toBe(false);
  });
  it("keeps distinct contexts and inner seam offsets separate", () => {
    expect(seamTextKey(index, 2)).not.toBe(seamTextKey(index, 4));
    expect(seamTextKey(index, 2)).not.toBe(seamTextKey(index, 3));
  });
  it("abstains when the asset's reference or inventory differs", () => {
    expect(replayAllows({ ...asset(2), reference: "other" }, index, 2, "بم")).toBe(false);
    expect(replayAllows({ ...asset(2), inventory: "other" }, index, 2, "بم")).toBe(false);
  });
});
