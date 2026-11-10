import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import { loadReferenceIndex } from "@tilawa/core";
import type { CycleEvidence, EngineEvent, ReferenceAsset, SequenceEvidenceCandidate } from "@tilawa/core";
import { syntheticFixture, withPosteriors } from "../../core/src/fixtures/index.js";
import { createEngine } from "../src/session.js";
import { addCycleEvidence } from "../src/cycle-turbulence.js";
import type { CycleModel } from "../src/cycle-turbulence.js";
import type { LoadedModel } from "../src/model.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8"));
// A small reference keeps this transport test independent of full-Quran search costs.
const index = loadReferenceIndex({ ...asset, ayahs: asset.ayahs.filter(a => a.s === 1 || (a.s === 26 && (a.a === 142 || a.a === 143)) || a.s === 108) });
const from = index.words.find(w => w.surah === 26 && w.ayah === 142 && w.word === 1)!;
const to = index.words.filter(w => w.surah === 26 && w.ayah === 143).at(-1)!;
const target = index.words.find(w => w.surah === 26 && w.ayah === 142 && w.word === 5)!;
const rival = index.words.find(w => w.surah === 108 && w.ayah === 1 && w.word === 3)!;
const fixture = () => syntheticFixture(index, { id: "cycle-session-transport", from, to, chunkSize: 4, edits: [
  { kind: "del", at: target.start - from.start, count: target.end - target.start },
  { kind: "ins", at: target.start - from.start, text: index.phonemes.slice(rival.start, rival.end) },
] }).evidence.map(e => withPosteriors(index, { ...e, frameMs: 40 }));

describe("cycle certificates through SDK sessions", () => {
  it("preserves primary evidence and tracker events while vetoing the exact serialized comparison", () => {
    const candidates: SequenceEvidenceCandidate[] = [];
    const n = index.tokens.size + 1;
    const scorer: CycleModel = { schema: 1, inventory: index.inventoryId, tokens: [...index.tokens].join(""),
      frameMs: 40, observationMs: 10, center: Array(16).fill(0), scale: Array(16).fill(1),
      means: Array.from({ length: n }, () => Array(16).fill(0)), variances: Array.from({ length: n }, () => Array(16).fill(1)),
      counts: Array(n).fill(10), cycleCounts: Array(n).fill(5), stay: .7, maxFrames: 4096 };
    const discoveryInput = fixture();
    const discovered = addCycleEvidence(scorer, new Float32Array(16000), discoveryInput, index, undefined,
      { passage: { from, to }, policy: { confirm: { after: "next-word" } } }, "memorization", (_, c) => candidates.push(c));
    expect(discovered).toBe(discoveryInput); // Silent audio supplies no cycle evidence.
    expect(candidates.length).toBeGreaterThan(0);
    const certificates: CycleEvidence[] = candidates.map(c => ({ ...c, controlWritten: -2, controlRival: -1,
      conditionedWritten: 1, conditionedRival: 0 }));
    const baseline = fixture(), enriched = fixture();
    enriched[0].phonemes[0] = { ...enriched[0].phonemes[0], cycleVetoes: certificates };
    expect(enriched.map(e => ({ ...e, phonemes: e.phonemes.map(({ cycleVetoes: _, ...p }) => p) }))).toEqual(baseline);
    const backend = { name: "unused", run: vi.fn(), release: vi.fn() };
    const model: LoadedModel = { manifest: JSON.parse(readFileSync(new URL("./fixtures/tiny/manifest.json", import.meta.url), "utf8")),
      backend, supported: ["wrong_word", "omitted_word", "added_word", "skipped_passage"] };
    const engine = createEngine({ reference: index, model });
    const run = (chunks: typeof baseline) => {
      const session = engine.startMemorization({ surah: from.surah, fromAyah: from.ayah, toAyah: to.ayah },
        { confirm: { after: "next-word" } });
      const events: EngineEvent[] = [];
      for (const e of JSON.parse(JSON.stringify(chunks))) events.push(...session.pushEvidence(e));
      events.push(...session.endEvidence());
      return { events, mistakes: session.mistakes() };
    };
    const original = run(baseline), vetoed = run(enriched);
    expect(original.mistakes.some(e => e.errorClass === "wrong_word")).toBe(true);
    expect(vetoed.mistakes.some(e => e.errorClass === "wrong_word")).toBe(false);
    expect(vetoed.events.filter(e => e.type !== "checker")).toEqual(original.events.filter(e => e.type !== "checker"));
    expect(vetoed.events).toEqual(original.events.map(e => e.type === "checker" && e.errorClass === "wrong_word" && e.state === "confirmed"
      ? { ...e, state: "uncertain" } : e));
    expect(backend.run).not.toHaveBeenCalled();
  });
});
