import { describe, expect, it } from "vitest";
import { accumulateChannel, channelFromCounts, emptyCounts, spanLogP } from "../src/noisy-channel.js";

const phones = [..."ءبتثجحخدذرزسشصضطظعغفقكلمنهويَُِ"];

function peaked(): ReturnType<typeof emptyCounts> {
  const c = emptyCounts(phones, 0);
  phones.forEach((p, i) => { c.sub[i][i] = 40; });
  const at = (p: string) => phones.indexOf(p);
  // A measured one-consonant blur, common enough to price and not to veto.
  c.sub[at("ط")][at("ت")] = 12;
  c.sub[at("ت")][at("ط")] = 12;
  c.sub[at("ن")][at("م")] = 2;
  return c;
}

describe("noisy channel span score", () => {
  it("prices a one-consonant blur under a two-consonant mismatch", () => {
    const ch = channelFromCounts(peaked());
    const heard = [..."تَيب"];
    const blurred = spanLogP(heard, [..."طَيب"], ch);
    const wrong = spanLogP(heard, [..."كَلب"], ch);
    expect(blurred).toBeGreaterThan(wrong);
    expect(spanLogP(heard, [..."تَيب"], ch)).toBeGreaterThan(blurred);
  });

  it("counts an alignment into the channel it was estimated from", () => {
    const c = emptyCounts(phones);
    accumulateChannel(c, "طَيب", "تَيب");
    const ch = channelFromCounts(c);
    const heard = [..."تَيب"];
    expect(spanLogP(heard, [..."طَيب"], ch)).toBeGreaterThan(spanLogP(heard, [..."كَلب"], ch));
  });
});
