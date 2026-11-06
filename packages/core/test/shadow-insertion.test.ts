import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { createTracker, shadowInsertionIntervals } from "../src/tracker.js";
import { loadReferenceIndex } from "../src/reference.js";

const tokens = (s: string) => [...s].map((p, i) => ({ p, start: i * 2, end: i * 2, conf: 1 }));

describe("shadow comparison on the ordinary tracker candidate paths", () => {
  const reference = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json",import.meta.url),"utf8")));
  const word = reference.words.findIndex(w=>w.surah===1 && w.ayah===2 && w.word===1);
  const speech = reference.phonemes.slice(reference.words[word].start,reference.words[word+1].end);

  it("does not propose an insertion for clean in-order speech", () => {
    const phones=tokens(speech), before=JSON.stringify(phones);
    expect(shadowInsertionIntervals(reference,word,phones,40)).toEqual([]);
    expect(JSON.stringify(phones)).toBe(before);
  });

  it("leaves the ordinary tracker events unchanged when the comparison is observed", () => {
    const evidence={inventory:reference.inventoryId,frameMs:40,phonemes:tokens(speech)};
    const plain=createTracker(reference,{trace:true});
    const shadow=createTracker(reference,{trace:true,insertionShadow:{word,report(){}}});
    expect(shadow.push(evidence)).toEqual(plain.push(evidence));
    expect(shadow.end()).toEqual(plain.end());
  });

  it("returns no interval without an anchor or teacher tokens", () => {
    expect(shadowInsertionIntervals(reference,word,[],40)).toEqual([]);
    expect(shadowInsertionIntervals(reference,-1,tokens(speech),40)).toEqual([]);
    expect(shadowInsertionIntervals(reference,reference.words.length,tokens(speech),40)).toEqual([]);
  });
});
