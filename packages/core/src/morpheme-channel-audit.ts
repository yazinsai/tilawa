/** Replay a supplied evidence file with frozen channels, without scoring or modifying its labels. */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { createTracker, trackerInput } from "./tracker.js";
import { createChecker } from "./checker.js";
import { loadReferenceIndex } from "./reference.js";
import { channelLogLikelihood, channelReference } from "./morpheme-channel.js";
import type { MorphemeChannel } from "./morpheme-channel.js";
import type { LocalObservationCandidate, LocalObservationComparison } from "./local.js";
import type { EngineEvent } from "./events.js";
import { chunk } from "./bench/index.js";
import type { QudRecording } from "./bench/index.js";

const { values } = parseArgs({ options: { evidence: { type: "string" }, channels: { type: "string" }, at: { type: "string" }, out: { type: "string" } } });
if (!values.evidence || !values.channels || !values.at || !values.out) throw new Error("Supply --evidence, --channels, --at (seconds), and --out");
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8"));
const evidence: QudRecording = JSON.parse(readFileSync(resolve(values.evidence), "utf8"));
const fitted: { models: { morphology: MorphemeChannel; phonetic: MorphemeChannel } } = JSON.parse(readFileSync(resolve(values.channels), "utf8"));
const surface = channelReference(index, 0, index.phonemes.length);
const atS = Number(values.at);
const comparisons: unknown[] = [];
const describe = (candidate: LocalObservationCandidate, heard: string) => {
  const reference = { phones: candidate.referenceOffsets.map(i => index.phonemes[i]).join(""), roles: candidate.referenceOffsets.map(i => surface.roles[i]) };
  const morphology = channelLogLikelihood(reference, heard, fitted.models.morphology);
  const phonetic = channelLogLikelihood(reference, heard, fitted.models.phonetic);
  return { word: index.words[index.wordAt(Math.max(0, candidate.offset - 1))], origin: index.words[index.wordAt(candidate.origin)],
    cost: candidate.cost / 10, movementCost: candidate.movementCost / 10, reference,
    morphology, phonetic, residualCost: candidate.cost / 10 - (morphology - phonetic) / 3 };
};
function replay(enabled: boolean): EngineEvent[] {
  let timeS = 0;
  const audit = (c: LocalObservationComparison) => {
    if (Math.abs(timeS - atS) <= 0.1) comparisons.push({ enabled, timeS, heard: c.heard,
      continuation: describe(c.continuation, c.heard), restart: describe(c.restart, c.heard) });
  };
  const tracker = createTracker(index, { phrases, localObservation: { audit, ...(enabled ? { channels: fitted.models } : {}) } });
  const checker = createChecker(index, { phrases });
  const events: EngineEvent[] = [];
  for (const e of chunk(evidence, 0.04, !!evidence.posteriors?.ids?.length)) {
    timeS = (e.decodedTo ?? 0) * e.frameMs / 1000;
    const positions = tracker.push(trackerInput(e));
    events.push(...positions, ...checker.push(e, positions));
  }
  const end = tracker.end();
  events.push(...end, ...checker.end(end));
  return events;
}
const baseline = replay(false), morphology = replay(true);
const confirmed = (events: EngineEvent[]) => {
  const latest = new Map<string, EngineEvent>();
  for (const e of events) if (e.type === "checker") latest.set(e.id, e);
  return [...latest.values()].filter(e => e.type === "checker" && e.state === "confirmed");
};
const output = { comparisons, baselineConfirmed: confirmed(baseline), morphologyConfirmed: confirmed(morphology),
  baselinePositions: baseline.filter(e => e.type === "position" && Math.abs(e.timeMs / 1000 - atS) < 5),
  morphologyPositions: morphology.filter(e => e.type === "position" && Math.abs(e.timeMs / 1000 - atS) < 5) };
writeFileSync(resolve(values.out), JSON.stringify(output, null, 2));
console.log(JSON.stringify(output));
