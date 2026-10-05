import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "../reference.js";
import type { ReferenceAsset } from "../reference.js";
import { DEFAULT_SPEED_FACTOR, perfGate } from "./perf.js";

const { values } = parseArgs({ options: { "speed-factor": { type: "string", default: String(DEFAULT_SPEED_FACTOR) } } });
const speedFactor = Number(values["speed-factor"]);
const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const r = perfGate(loadReferenceIndex(asset), speedFactor);
console.log(
  `${r.audioS.toFixed(0)} s of audio in ${(r.processS * 1000).toFixed(0)} ms: rtf ${r.rtf.toFixed(4)}, budget ${r.budgetRtf.toFixed(4)} at speed factor ${speedFactor}, slowest push ${r.maxPushMs.toFixed(1)} ms`,
);
// At the device floor each push must also finish before the next second of audio arrives.
const pushBudgetMs = 1000 / speedFactor;
const ok = r.pass && r.maxPushMs <= pushBudgetMs;
console.log(ok ? "PASS" : `FAIL${r.pass ? `: slowest push over ${pushBudgetMs.toFixed(0)} ms` : ": below real time on the device floor"}`);
process.exit(ok ? 0 : 1);
