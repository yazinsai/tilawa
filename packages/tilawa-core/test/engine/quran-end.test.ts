import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../../src/engine/checker.js";
import { loadReferenceIndex } from "../../src/engine/reference.js";
import { syntheticFixture, withPosteriors } from "../fixtures/synthetic.js";
import type { Edit } from "../fixtures/synthetic.js";
import type { CheckerEvent, WordRef } from "../../src/engine/events.js";

// The witr-muawwidhat session died at the end of An-Nas: a word heard after the Quran's last word, or an error on
// that last word, looked up the word after it (index words.length) and threw on every later push.
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8")));
const from: WordRef = { surah: 113, ayah: 1, word: 1 };
const last = index.words[index.words.length - 1];
const to: WordRef = { surah: last.surah, ayah: last.ayah, word: last.word };
const span = last.end - index.words.find((w) => w.surah === 113 && w.ayah === 1 && w.word === 1)!.start;

function run(edits: Edit[], passage: boolean) {
  const fixture = syntheticFixture(index, { id: "quran-end", from, to, edits, chunkSize: 4 });
  const checker = createChecker(index, passage ? { passage: { from, to }, policy: { confirm: { after: "next-word" } } } : {});
  const events: CheckerEvent[] = [];
  for (const evidence of fixture.evidence) events.push(...checker.push(withPosteriors(index, evidence), []));
  events.push(...checker.end());
  return events;
}

describe("the end of the Quran", () => {
  const tail = [...index.phonemes.slice(index.words[0].start, index.words[0].end)].join(""); // some extra speech
  const cases: [string, Edit[]][] = [
    ["a word added after An-Nas 114:6", [{ kind: "ins", at: span, text: tail }]],
    ["a wrong last word", [{ kind: "sub", at: span - 2, to: "ك" }, { kind: "sub", at: span - 1, to: "ك" }]],
    ["clean", []],
    ["two words added after the end", [{ kind: "ins", at: span, text: tail + tail }]],
    ["a word added before the last word", [{ kind: "ins", at: span - 4, text: tail }]],
  ];
  for (const [name, edits] of cases)
    for (const passage of [true, false])
      it(`${name} (${passage ? "declared passage" : "free"}) does not throw`, () => {
        expect(() => run(edits, passage)).not.toThrow();
      });
});
