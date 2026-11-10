import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { LexicalChangeCertificates, loadReferenceIndex } from "../src/index.js";
import type { LexicalChangeEvidence, WordLocalizationCandidate } from "../src/index.js";
import { syntheticFixture, replayChecked } from "../src/fixtures/index.js";

const candidate: WordLocalizationCandidate = { word:{surah:1,ayah:2,word:1},expected:"بت",observed:"كم",fromFrame:10,toFrame:20 };
const certificate: LexicalChangeEvidence = {...candidate,score:.1,preservation:.2};

describe("lexical-change certificates", () => {
  it("vetoes only the identical occurrence below the preservation point", () => {
    const certificates = new LexicalChangeCertificates();
    certificates.add(certificate);
    expect(certificates.veto(candidate)).toBe(true);
    for (const change of [{fromFrame:11},{toFrame:21},{expected:"ب"},{observed:"ك"},{word:{...candidate.word,word:2}}])
      expect(certificates.veto({...candidate,...change})).toBe(false);
  });

  it("preserves ties and a stronger duplicate; conflicting calibration abstains", () => {
    const ties = new LexicalChangeCertificates();
    ties.add({...certificate,score:.2});
    expect(ties.veto(candidate)).toBe(false);
    const duplicate = new LexicalChangeCertificates();
    duplicate.add(certificate); duplicate.add({...certificate,score:.8}); duplicate.add(certificate);
    expect(duplicate.veto(candidate)).toBe(false);
    const conflict = new LexicalChangeCertificates();
    conflict.add(certificate); conflict.add({...certificate,preservation:.3});
    expect(conflict.veto(candidate)).toBe(false);
  });

  it("ignores nonfinite scores and malformed frame ranges", () => {
    for (const change of [{score:NaN},{score:-1},{preservation:Infinity},{fromFrame:10.5},{toFrame:10}]) {
      const certificates = new LexicalChangeCertificates();
      certificates.add({...certificate,...change});
      expect(certificates.veto(candidate)).toBe(false);
    }
  });
});

describe("checker consumes listener lexical-change evidence", () => {
  const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json",import.meta.url),"utf8")));
  const from = {surah:2,ayah:255,word:1}, to = {surah:2,ayah:256,word:10}, target = {surah:2,ayah:255,word:23};
  const find = (r: typeof from) => index.words.find(w => w.surah === r.surah && w.ayah === r.ayah && w.word === r.word)!;
  const word = find(target), start = find(from), rival = find({surah:108,ayah:1,word:3});
  const fixture = () => syntheticFixture(index,{id:"localization",from,to,chunkSize:8,edits:[
    {kind:"del",at:word.start-start.start,count:word.end-word.start},
    {kind:"ins",at:word.start-start.start,text:index.phonemes.slice(rival.start,rival.end)},
  ]});

  it("turns an existing word warning uncertain without changing its text or timestamp", () => {
    const f = fixture(), candidates: WordLocalizationCandidate[] = [];
    const original = replayChecked(index,f,{}, {onWordLocalization:c => candidates.push(c)});
    const warning = original.find(e => e.state === "confirmed" && e.errorClass === "wrong_word")!;
    expect(warning).toBeDefined(); expect(candidates).toHaveLength(1);
    f.evidence[0].phonemes[0].lexicalChanges = [{...candidates[0],score:.1,preservation:.2}];
    const reviewed = replayChecked(index,JSON.parse(JSON.stringify(f)));
    expect(reviewed.find(e => e.id === warning.id && e.state === "uncertain")).toEqual({...warning,state:"uncertain"});
    expect(reviewed.some(e => e.state === "confirmed")).toBe(false);
  });

  it("discovery callbacks leave all decisions unchanged", () => {
    const f = fixture(), contexts: number[] = [];
    const original = replayChecked(index,f);
    const discovered = replayChecked(index,f,{}, {onWordLocalization:() => {},onLocalizationContext:c => contexts.push(c.words.length)});
    expect(discovered).toEqual(original);
    expect(contexts.length).toBeGreaterThan(0);
  });
});
