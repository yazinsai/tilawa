/**
 * Estimate the shipped listener's confusion channel from gold-excluded decodes and the clean-seam
 * log-odds cut, then rank the practice seam targets. Run on Modal after channel wavs are decoded:
 *
 *   pnpm --filter @tilawa/core exec tsx src/bench/channel-fit.ts
 */
import { readFileSync, writeFileSync } from "node:fs";
import type { PositionEvent } from "../events.js";
import { collapseHeld } from "../fuzzy.js";
import {
  accumulateChannel, ayahSeamSpans, channelFromCounts, emptyCounts, phrasePhones, rankSeam, spanTokens,
} from "../noisy-channel.js";
import type { PhraseAsset, ReferenceAsset } from "../reference.js";
import { loadReferenceIndex } from "../reference.js";
import { createTracker, trackerInput } from "../tracker.js";
import { chunks } from "./lab.js";
import type { SessionEvidence } from "./sessions.js";

const root = process.env.CHANNEL_DIR ?? "/scratch/listener-step-span-noisy-channel-rank/channel";
const data = process.env.LAB_DATA ?? "/lab/data";
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const phrases = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8")) as PhraseAsset;
const phones = [...index.tokens];

function stepsOf(ev: SessionEvidence) {
  const tracker = createTracker(index, { mode: "free", phrases });
  const steps: { wi: number; timeMs: number }[] = [];
  let last = -1;
  const take = (events: PositionEvent[]) => {
    for (const p of events) {
      if (p.state !== "locked" || !p.observed[0]) continue;
      const wi = index.words.findIndex((w) => w.surah === p.observed[0].word.surah && w.ayah === p.observed[0].word.ayah && w.word === p.observed[0].word.word);
      if (wi < 0 || wi === last) continue;
      last = wi;
      steps.push({ wi, timeMs: p.timeMs });
    }
  };
  for (const e of chunks(ev)) take(tracker.push(trackerInput(e)).filter((x): x is PositionEvent => x.type === "position"));
  take(tracker.end().filter((x): x is PositionEvent => x.type === "position"));
  return steps;
}

const refs = readFileSync(`${root}/refs.jsonl`, "utf8").trim().split("\n").map((l) => JSON.parse(l) as { id: string; ref: string });
const counts = emptyCounts(phones, 0);
let aligned = 0;
for (const r of refs) {
  for (const variant of ["studio", "phone"]) {
    const ev = JSON.parse(readFileSync(`${root}/ev/${r.id}.${variant}.evidence.json`, "utf8")) as SessionEvidence;
    const heard = ev.phonemes.map((p) => p.p).join("");
    if (!heard) continue;
    accumulateChannel(counts, r.ref, heard);
    aligned++;
  }
}
console.log(`aligned ${aligned} decodes, written phones ${counts.nWritten}, insertions ${counts.ins.reduce((a, b) => a + b, 0)}`);

const ch0 = channelFromCounts({ ...counts, cut: 0 });
const phraseList = phrasePhones(phrases);
let cut = -Infinity;
let seams = 0;
for (const r of refs) {
  for (const variant of ["studio", "phone"]) {
    const ev = JSON.parse(readFileSync(`${root}/ev/${r.id}.${variant}.evidence.json`, "utf8")) as SessionEvidence;
    for (const seam of ayahSeamSpans(stepsOf(ev), index)) {
      const heard = spanTokens(ev.phonemes, ev.frameMs, seam.t0, seam.t1);
      if ([...heard].length < 4) continue;
      const rank = rankSeam(heard, seam.word1, index, phraseList, ch0);
      seams++;
      if (rank.word && rank.logOdds > cut) cut = rank.logOdds;
    }
  }
}
console.log(`clean seams ${seams}, cut ${Number.isFinite(cut) ? cut.toFixed(3) : "none"}`);
if (!Number.isFinite(cut)) throw new Error("no clean gold-excluded seam");

const ch = channelFromCounts({ ...counts, cut });
const cases = [
  { session: "lab-v1.5-p50", at: 146.47, loc: "37:99:1", said: "وَقِفُۥۥهُم" },
  { session: "lab-v1.4-p19", at: 102.06, loc: "51:33:1", said: "وَمِن" },
  { session: "lab-v1.5-p33", at: 156.16, loc: "26:208:1", said: "كَرِۦۦم" },
  { session: "lab-v1.5-p49", at: 152.25, loc: "26:123:1", said: "كَرِۦۦم" },
];
const byKey = new Map(index.words.map((w, i) => [`${w.surah}:${w.ayah}:${w.word}`, i]));
let first = 0;
for (const c of cases) {
  const ev = JSON.parse(readFileSync(`${data}/lab-v1.5/sessions/${c.session}.spliced-clean-int8.phone.evidence.json`, "utf8")) as SessionEvidence;
  const seam = ayahSeamSpans(stepsOf(ev), index).find((x) => byKey.get(c.loc) === x.word1 && x.t0 / 1000 < c.at + 2 && x.t1 / 1000 > c.at - 2);
  if (!seam) { console.log(c.session, "NO SEAM"); continue; }
  const heard = spanTokens(ev.phonemes, ev.frameMs, seam.t0, seam.t1);
  const rank = rankSeam(heard, seam.word1, index, phraseList, ch);
  const said = collapseHeld([...c.said]).join("");
  const plantedFirst = rank.word === said;
  if (plantedFirst) first++;
  console.log(`${c.session} heard ${heard}`);
  console.log(`  said ${said} win ${rank.word} odds ${rank.logOdds.toFixed(2)} vs ${rank.competitor} ${rank.competitorLogP.toFixed(1)} cut ${cut.toFixed(2)} first ${plantedFirst}`);
}
console.log(`planted ranked first: ${first}/4`);

const payload = { ...counts, cut };
const out = process.env.LAB_OUT ?? root;
writeFileSync(`${out}/listener-channel.json`, JSON.stringify(payload));
console.log(`wrote ${out}/listener-channel.json phones ${payload.phones.length} cut ${cut}`);
