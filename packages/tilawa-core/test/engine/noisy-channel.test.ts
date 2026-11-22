import { describe, expect, it } from "vitest";
import { collapseHeld } from "../../src/engine/fuzzy.js";
import { LISTENER_CHANNEL } from "../../src/engine/listener-channel.js";
import { accumulateChannel, channelFromCounts, emptyCounts, spanLogP } from "../../src/engine/noisy-channel.js";

const phones = [..."ءبتثجحخدذرزسشصضطظعغفقكلمنهويَُِ"];

describe("listener confusion channel", () => {
  it("prices a counted blur under an uncounted mismatch", () => {
    const c = emptyCounts(phones);
    accumulateChannel(c, "طَيب", "تَيب");
    const ch = channelFromCounts(c);
    const heard = [..."تَيب"];
    expect(spanLogP(heard, [..."طَيب"], ch)).toBeGreaterThan(spanLogP(heard, [..."كَلب"], ch));
  });

  it("ranks the two measured omission stretches toward the spoken word", () => {
    const ch = channelFromCounts(LISTENER_CHANNEL);
    const p = (heard: string, written: string) => spanLogP(collapseHeld([...heard]), collapseHeld([...written]), ch);
    expect(p("ءُننننَاا", "كُننننَاا")).toBeGreaterThan(p("ءُننننَاا", "ءِننننَاا"));
    expect(p("دڇلِۦ", "صَدڇرِۦۦ")).toBeGreaterThan(p("دڇلِۦ", "لِۦۦ"));
    expect(p("لِۦ", "لِۦۦ")).toBeGreaterThan(p("لِۦ", "صَدڇرِۦۦ"));
  });
});
