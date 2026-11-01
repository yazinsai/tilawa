/** Oracle-arm score of the target sessions, plus whether phoneme anchors see each skip. */
import { readFileSync } from "node:fs";
import { duplicateSpans, phonemeSkipRuns } from "../../../../packages/core/src/checker-anchors.ts";
import { oracleEvidence, runTruePosition } from "../../../../packages/core/src/bench/lab.ts";
import { loadReferenceIndex } from "../../../../packages/core/src/reference.ts";
import { scoreSessionEvents } from "../../../../packages/core/src/bench/sessions.ts";
import type { BenchSession } from "../../../../packages/core/src/bench/sessions.ts";

const asset = JSON.parse(readFileSync(new URL("../../../../packages/core/assets/ref-v2.json", import.meta.url), "utf8"));
const phrases = JSON.parse(readFileSync(new URL("../../../../packages/core/assets/phrases-v1.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const spans = duplicateSpans(index.words, index.phonemes);
const raw = index.words.map((w) => (w.end > w.start ? index.phonemes.slice(w.start, w.end) : ""));
const ayahStart = index.words.map((w, i) => i === 0 || index.words[i - 1].surah !== w.surah || index.words[i - 1].ayah !== w.ayah);
const key = (w: { surah: number; ayah: number; word: number }) => `${w.surah}:${w.ayah}:${w.word}`;
const at = (surah: number, ayah: number, word: number) => index.words.findIndex((w) => w.surah === surah && w.ayah === ayah && w.word === word);

function show(surah: number, a0: number, a1: number) {
  console.log(`\n-- ${surah}:${a0}-${a1}`);
  for (let i = 0; i < index.words.length; i++) {
    const w = index.words[i];
    if (w.surah !== surah || w.ayah < a0 || w.ayah > a1) continue;
    const flags = `${spans.unique[i] ? "U" : "."}${spans.refrain[i] ? "R" : "."}${ayahStart[i] ? "A" : "."}`;
    console.log(`  ${key(w)} ${flags} ${raw[i]}`);
  }
}

show(26, 100, 112);
show(26, 126, 133);
show(55, 26, 32);
show(77, 30, 40);

const ids = process.argv.slice(2).length ? process.argv.slice(2) : ["lab-v1.5-p24", "lab-v1.5-p73", "lab-v1.4-p10", "lab-v1.5-p15", "lab-v1.5-p16"];
for (const id of ids) {
  const s = JSON.parse(readFileSync(new URL(`./${id}.json`, import.meta.url), "utf8")) as BenchSession;
  const events = runTruePosition(s, oracleEvidence(s, index), index, phrases);
  const score = scoreSessionEvents(s, events, index, { kind: "none" }, { anyClass: true });
  const hit = new Set(score.assignments.filter((a) => a.confirmed && a.eventIndex !== null).map((a) => a.eventIndex!));
  console.log(`\n== ${id} oracle hits ${score.hits} misses ${score.misses} fw ${score.falseWarnings}`);
  s.events.forEach((e, i) => {
    if (hit.has(i)) return;
    const loc = e.location;
    console.log(`  MISS ${e.errorClass} ${key(loc)} @ ${e.startS.toFixed(2)}`);
  });
  for (const a of score.assignments) {
    if (a.eventIndex !== null || !a.confirmed) continue;
    const states = events.filter((e) => e.type === "checker" && e.id === a.warningId);
    const c = states.find((e) => e.type === "checker" && e.state === "confirmed");
    if (c?.type === "checker") console.log(`  FW ${c.errorClass} ${key(c.location)} @ ${(c.timeMs / 1000).toFixed(2)}`);
  }
  // Phoneme anchors over the words actually said.
  const ixs = s.words.map((w) => at(w.at.surah, w.at.ayah, w.at.word)).filter((i) => i >= 0);
  const heard = ixs.map((i) => raw[i]).join("");
  const lo = Math.min(...ixs), hi = Math.max(...ixs);
  const runs = phonemeSkipRuns({ raw, unique: spans.unique, refrain: spans.refrain, ayahStart, from: lo, to: hi, heard });
  const skips = runs.filter((r) => r.runs.length);
  console.log(`  phoneme runs with a skip: ${skips.length}`);
  for (const r of skips) {
    const loc = index.words[r.runs[0].word];
    const end = index.words[r.runs[0].last];
    if (loc.surah !== s.words[0]?.at.surah && ![26, 55, 77].includes(loc.surah)) continue;
    console.log(`    ${key(loc)}..${key(end)} anchors ${key(index.words[r.from])} -> ${key(index.words[r.to])}`);
  }
}
