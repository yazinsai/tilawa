import type { PhoneEncoder, WordMetricModel } from "../src/word-metric.js";
import type { ReferenceIndex } from "../src/reference.js";

export function metricFixture(): { index: ReferenceIndex; model: WordMetricModel } {
  let phonemes = "";
  const words = ["بتد", "درز", "سشص", "بتد", "فنق"].map((text, i) => {
    const start = phonemes.length;
    phonemes += text;
    return { surah: 1, ayah: 1, word: i + 1, start, end: phonemes.length };
  });
  const tokens = [...new Set(phonemes)].join("");
  const encoder: PhoneEncoder = {
    convolution: Array.from({ length: 4 }, (_, h) => Array.from({ length: 3 }, (_, k) =>
      Array.from({ length: tokens.length + 1 }, (_, p) => Math.sin((h + 1) * (p + 1) + k) / 4))),
    bias: [0, 0.1, -0.2, 0.3],
    projection: Array.from({ length: 4 }, (_, h) => Array.from({ length: 8 }, (_, k) => h === k || h + 4 === k ? 1 : 0)),
  };
  const index: ReferenceIndex = { phonemes, words, tokens: new Set(tokens), version: "unit", inventoryId: "unit",
    gramSize: 4, grams: new Map(), wordAt: offset => {
      for (let i = words.length - 1; i >= 0; i--) if (offset >= words[i].start) return i;
      return 0;
    } };
  return { index, model: { version: 1, inventory: "unit", tokens, written: encoder, observed: structuredClone(encoder), scale: 10 } };
}
