import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence, SequenceEvidenceCandidate } from "@tilawa/core";
import { syntheticFixture, withPosteriors } from "../../core/src/fixtures/index.js";
import type { LoadedModel } from "../src/model.js";
import { insertionCells } from "../src/seam-insertion.js";

const fixtures = vi.hoisted(() => ({ evidence: null as PhonemeEvidence | null, audio: [] as number[] }));
vi.mock("../src/listener.js", () => ({ createListener: () => ({
  push: async () => null, end: async () => fixtures.evidence, decoded: 0,
}) }));
vi.mock("../src/gap-decode.js", () => ({ spliceContextFreeGaps: async (
  _model: unknown, _audio: unknown, _rate: unknown, chunks: PhonemeEvidence[],
) => chunks }));
vi.mock("../src/acoustic-span.js", async original => {
  const actual = await original<typeof import("../src/acoustic-span.js")>();
  return {...actual, addAcousticSpanEvidence: async (...args: Parameters<typeof actual.addAcousticSpanEvidence>) => {
    const [scorer, , audio, chunks, index, phrases, options, mode] = args;
    if (!scorer) return chunks;
    fixtures.audio = Array.from(audio);
    const candidates: SequenceEvidenceCandidate[] = [];
    const tracker = createTracker(index, {mode, phrases, acquireMs:1000});
    const checker = createChecker(index, {...options, sequenceEvidence(c) {candidates.push(c);return true;}});
    for (const e of insertionCells(chunks)) checker.push(e, tracker.push(trackerInput(e)));
    checker.end(tracker.end());
    return actual.carryAcousticSpanEvidence(chunks, candidates.map(c => ({...c,
      contrastMargin:1, controlMargin:-1,
    })));
  }};
});
import { createEngine } from "../src/session.js";

describe("contrast-negative certificates at audio session completion", () => {
  it("publishes a downgrade, retains its stored certificate and preserves primary positions and insertions", async () => {
    const index=loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json",import.meta.url),"utf8")));
    const from={surah:26,ayah:142,word:1}, to={surah:26,ayah:143,word:3};
    const target=index.words.find(w=>w.surah===26&&w.ayah===142&&w.word===5)!;
    const start=index.words.find(w=>w.surah===from.surah&&w.ayah===from.ayah&&w.word===from.word)!.start;
    const chunks=syntheticFixture(index,{id:"depth-session",from,to,chunkSize:10000,edits:[
      {kind:"del",at:target.start-start,count:target.end-target.start},
      {kind:"ins",at:target.start-start,text:"نُۥۥح"},
    ]}).evidence.map(e=>withPosteriors(index,e));
    expect(chunks).toHaveLength(1);
    fixtures.evidence=chunks[0];
    const run=async (depth:boolean) => {
      const model={manifest:{inventory:index.inventoryId,id:"test",frameMs:40,sampleRate:16000},
        supported:["wrong_word","added_word"],...(depth&&{acousticSpans:{}})} as unknown as LoadedModel;
      const session=createEngine({reference:index,model,recordEvidence:true})
        .startMemorization({surah:26,fromAyah:142,toAyah:143},{confirm:{after:"next-word"}});
      await session.pushAudio(new Float32Array([.25,.5]),16000);
      return {session,events:await session.end()};
    };
    const baseline=await run(false), reviewed=await run(true);
    expect(baseline.session.mistakes().some(e=>e.errorClass==="wrong_word")).toBe(true);
    expect(reviewed.session.mistakes().some(e=>e.errorClass==="wrong_word")).toBe(false);
    expect(reviewed.events.some(e=>e.type==="checker"&&e.errorClass==="wrong_word"&&e.state==="uncertain")).toBe(true);
    expect(reviewed.events.filter(e=>e.type==="position")).toEqual(baseline.events.filter(e=>e.type==="position"));
    expect(reviewed.session.mistakes().filter(e=>e.errorClass==="added_word"))
      .toEqual(baseline.session.mistakes().filter(e=>e.errorClass==="added_word"));
    expect(reviewed.session.evidence().some(e=>e.phonemes.some(p=>p.acousticSpanVetoes?.length))).toBe(true);
    expect(fixtures.audio).toEqual([.25,.5]);
  });
});
