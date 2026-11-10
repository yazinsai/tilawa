import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import type { WrongWordRecheckCandidate } from "../src/checker.js";
import { loadReferenceIndex } from "../src/reference.js";
import { completePronunciationMatch } from "../src/pronunciation-match.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { WrongWordRepairEvidence } from "../src/evidence.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json",import.meta.url),"utf8")));
const from = {surah:2,ayah:255,word:20}, to = {surah:2,ayah:255,word:30}, target = {surah:2,ayah:255,word:23};
const first = index.words.find(w => w.surah === from.surah && w.ayah === from.ayah && w.word === from.word)!;
const word = index.words.find(w => w.surah === target.surah && w.ayah === target.ayah && w.word === target.word)!;
const written = index.phonemes.slice(word.start,word.end), said = written.replace("ش","س").replace("ف","م");
const fixture = syntheticFixture(index,{id:"repair-scope",from,to,chunkSize:20,edits:[
  {kind:"del",at:word.start-first.start,count:written.length},
  {kind:"ins",at:word.start-first.start,text:said},
]});
const chunks = fixture.evidence.map(e => withPosteriors(index,e));

function run(metadata: WrongWordRepairEvidence[] = []) {
  const requests = new Map<string,WrongWordRecheckCandidate>();
  const checker = createChecker(index,{passage:{from,to},policy:{confirm:{after:"next-word"}},
    wrongWordRecheck(c) { requests.set(JSON.stringify(c),c); return true; } });
  for (const [i,chunk] of chunks.entries()) checker.push(i ? chunk : {...chunk,
    phonemes:chunk.phonemes.map((p,j) => j ? p : {...p,wrongWordRepairs:JSON.parse(JSON.stringify(metadata))})},[]);
  checker.end();
  return {requests:[...requests.values()],mistakes:checker.mistakes()};
}
const baseline = run();
const certificates = () => baseline.requests.filter(c => c.word === index.words.indexOf(word)).map(c => ({
  ...c,word:target,recovered:c.written,writtenNll:0,rivalNll:20,
}));

describe("positive written-pronunciation repair certificates", () => {
  it("requires a complete pronunciation and retains true consonant doubling", () => {
    expect(completePronunciationMatch("ذَاات","زَات")).toBe(true);
    expect(completePronunciationMatch("نَن","لَل")).toBe(false);
    expect(completePronunciationMatch("نَن","ن")).toBe(false);
    expect(completePronunciationMatch("بَت","بَتَر")).toBe(false);
    expect(completePronunciationMatch("بَت","بَ")).toBe(false);
  });
  it("vetoes only an exact string-path accusation after evidence serialization", () => {
    expect(baseline.mistakes.some(e => e.errorClass === "wrong_word")).toBe(true);
    expect(certificates().length).toBeGreaterThan(0);
    expect(run(certificates()).mistakes.some(e => e.errorClass === "wrong_word")).toBe(false);
  });
  it("abstains on a partial reading or a recovered rival", () => {
    for (const recovered of [written.slice(1),said,""])
      expect(run(certificates().map(c => ({...c,recovered}))).mistakes.some(e => e.errorClass === "wrong_word")).toBe(true);
  });
  it("does not lend a certificate to a changed span, rival, or occurrence", () => {
    const cs = certificates();
    for (const changed of [cs.map(c => ({...c,fromFrame:c.fromFrame+1})),
      cs.map(c => ({...c,observed:c.observed+"ب"})),cs.map(c => ({...c,word:{...target,word:target.word+1}}))])
      expect(run(changed).mistakes.some(e => e.errorClass === "wrong_word")).toBe(true);
  });
  it("keeps a warning when the recheck still acoustically supports the rival", () => {
    expect(run(certificates().map(c => ({...c,writtenNll:20,rivalNll:0}))).mistakes.some(e => e.errorClass === "wrong_word")).toBe(true);
  });
});
