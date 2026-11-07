import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "../src/reference.js";
import type { OmissionReadCandidate } from "../src/checker.js";
import type { SegmentPresenceEvidence } from "../src/evidence.js";
import { replayChecked, syntheticFixture } from "../src/fixtures/index.js";

const index=loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json",import.meta.url),"utf8")));
const from={surah:2,ayah:255,word:1}, to={surah:2,ayah:257,word:5}, target={surah:2,ayah:255,word:23};
const start=index.words.find(w=>w.surah===from.surah && w.ayah===from.ayah && w.word===from.word)!;
const missing=index.words.find(w=>w.surah===target.surah && w.ayah===target.ayah && w.word===target.word)!;
const fixture=()=>syntheticFixture(index,{id:"weak-label-unit",from,to,chunkSize:8,
  edits:[{kind:"del",at:missing.start-start.start,count:missing.end-missing.start}]});

describe("closed omission read certificates",()=>{
  it("discovers only a proposed single-word omission and preserves all baseline events",()=>{
    const found:OmissionReadCandidate[]=[];
    const baseline=replayChecked(index,fixture());
    const discovered=replayChecked(index,fixture(),{},{omissionRead:c=>{found.push(c);return false;}});
    expect(discovered).toEqual(baseline);
    expect(found.length).toBeGreaterThan(0);
    expect(found.every(c=>index.words[c.word]===missing && c.toFrame>c.fromFrame)).toBe(true);
  });
  it("vetoes confirmation with identical location, timestamp and suspicion",()=>{
    const baseline=replayChecked(index,fixture());
    const vetoed=replayChecked(index,fixture(),{},{omissionRead:()=>true});
    expect(vetoed[0]).toEqual(baseline[0]);
    expect(vetoed.at(-1)).toEqual({...baseline.at(-1),state:"uncertain"});
  });
  it("transports an exact-span read certificate through JSON and rejects stale or tied evidence",()=>{
    const found:OmissionReadCandidate[]=[];
    const baseline=replayChecked(index,fixture(),{},{omissionRead:c=>{found.push(c);return false;}});
    const metadata:SegmentPresenceEvidence[]=found.map(c=>({word:target,fromFrame:c.fromFrame,toFrame:c.toFrame,read:2,omit:1}));
    const replay=(data:SegmentPresenceEvidence[])=>{
      const f=fixture(); f.evidence[0].phonemes[0].segmentPresence=data;
      return replayChecked(index,JSON.parse(JSON.stringify(f)));
    };
    expect(replay(metadata).at(-1)?.state).toBe("uncertain");
    expect(replay(metadata.map(m=>({...m,toFrame:m.toFrame+1})))).toEqual(baseline);
    expect(replay(metadata.map(m=>({...m,read:1})))).toEqual(baseline);
    expect(replay(metadata.map(m=>({...m,read:Infinity})))).toEqual(baseline);
  }, 20000);
  it("does not invoke read discovery on clean recitation",()=>{
    let calls=0;
    const f=syntheticFixture(index,{id:"clean-unit",from,to,chunkSize:8});
    replayChecked(index,f,{},{omissionRead:()=>{calls++;return true;}});
    expect(calls).toBe(0);
  });
});
