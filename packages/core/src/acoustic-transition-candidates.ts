/** Training candidate export: the unchanged local DP supplies all contrasting pronunciations. */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { TransitionFrames, posteriorTransitionRows } from "./acoustic-transition.js";
import { PosteriorFrames } from "./ctc.js";
import { createLocalSearch } from "./local.js";
import { loadReferenceIndex } from "./reference.js";
import type { DecodedPhoneme } from "./evidence.js";
import type { AcousticCandidate } from "./local.js";

const [source, output] = process.argv.slice(2);
const document = JSON.parse(readFileSync(source, "utf8"));
if (document.trainingOnly !== true) throw new Error("Training-only candidate input required");
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const examples = document.examples.map((e: { written: string; start: number; phones: DecodedPhoneme[]; clip: number; speaker: string; fold: number }) => {
  let candidates: AcousticCandidate[] = [];
  const local = createLocalSearch(index, { offset: e.start, origin: e.start, edits: 0 }, {
    training: true, inspect: c => { candidates = c; },
  });
  for (const ph of e.phones) local.step(ph.p);
  const choices = [e.written];
  // Contrasts are retained states, ordered by their unchanged cost; no corpus-specific words or locations.
  for (const c of candidates.sort((a, b) => a.cost - b.cost)) {
    if (!c.pronunciation || choices.includes(c.pronunciation) || c.observations !== e.phones.length) continue;
    choices.push(c.pronunciation);
    if (choices.length === 9) break;
  }
  const evidence = document.clips[e.clip].evidence;
  const frames = new PosteriorFrames(index.tokens.size);
  frames.append({ ...evidence.posteriors, startFrame: 0, nFrames: evidence.nFrames });
  const original = new TransitionFrames();
  original.append(evidence.acousticTransitions);
  const rows = original.rows(e.phones[0].start, e.phones.at(-1)!.end + 1);
  const controlRows = posteriorTransitionRows(frames, e.phones[0].start, e.phones.at(-1)!.end + 1);
  if (!rows) throw new Error("Latent boundary is outside the trajectory");
  return { speaker: e.speaker, fold: e.fold, choices, rows, controlRows };
});
writeFileSync(join(output, "acoustic-transition-examples.json"), JSON.stringify({ ...document, clips: undefined, examples }));
console.log(JSON.stringify({ examples: examples.length, competing: examples.filter((e: { choices: string[] }) => e.choices.length > 1).length }));
