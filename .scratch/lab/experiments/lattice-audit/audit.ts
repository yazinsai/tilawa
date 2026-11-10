// Diagnostic only: inject a read-only local-search observer into a generated tracker copy. No scorer changes.
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createChecker } from "../../../../packages/core/src/checker.js";
import { loadReferenceIndex } from "../../../../packages/core/src/reference.js";
import { chunks } from "../../../../packages/core/src/bench/lab.js";
import { trackerInput } from "../../../../packages/core/src/tracker.js";
import { createTracker as createProductTracker } from "../../../../packages/core/src/tracker.js";
import { PosteriorFrames } from "../../../../packages/core/src/ctc.js";
import { createLocalSearch } from "../../../../packages/core/src/local.js";
import { acousticPrefixes } from "../../../../packages/core/src/ctc-prefix.js";
import { localAcousticCandidates } from "../../../../packages/core/src/local-acoustic.js";

const root = fileURLToPath(new URL("../../../../", import.meta.url));
const core = resolve(root, "packages/core/src");
const session = process.env.AUDIT_SESSION ?? "lab-v1.5-p46";
const at = Number(process.env.AUDIT_AT ?? "121.84");
const evidence = JSON.parse(readFileSync(`${process.env.LAB_DATA}/lab-v1.5u/sessions/${session}.spliced-clean-int8.phone.evidence.json`, "utf8"));
const index = loadReferenceIndex(JSON.parse(readFileSync(resolve(core, "../assets/ref-v2.json"), "utf8")));
const phrases = JSON.parse(readFileSync(resolve(core, "../assets/phrases-v1.json"), "utf8"));
const frames = new PosteriorFrames(index.tokens.size);
frames.append({ ...evidence.posteriors, startFrame: 0, nFrames: evidence.nFrames });
const reports: unknown[] = [];
const add = (a: number, b: number) => a === -Infinity ? b : Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));

const wrapped = (ix: typeof index, start: Parameters<typeof createLocalSearch>[1]) => {
  const search = createLocalSearch(ix, start);
  const history: { ph: any; checkpoint: unknown }[] = [];
  return {
    ...search,
    stepEvidence(ph: any) {
      const before = search.position();
      history.push({ ph, checkpoint: search.save() });
      if (history.length > 16) history.shift();
      search.step(ph.p);
      const after = search.position();
      if (Math.abs(ph.end * evidence.frameMs / 1000 - at) > 0.6) return;
      if (after.origin === before.origin && !process.env.AUDIT_ALL) return;
      const checkpoint = history[0].checkpoint;
      const from = history[0].ph.start, to = ph.end + 1;
      const heard = history.map((h) => h.ph.p).join("");
      const original = search.candidates();
      const source = ix.words[ix.wordAt(Math.max(0, before.offset - 1))];
      const dest = ix.words[ix.wordAt(Math.max(0, after.offset - 1))];
      const word = (w: any) => `${w.surah}:${w.ayah}:${w.word}`;
      const inWord = (c: any, w: any) => c.offset > w.start && c.offset <= w.end;
      const min = (cs: any[], w: any) => cs.filter((c) => inWord(c, w)).sort((a, b) => a.cost - b.cost)[0];
      const widths = [];
      for (const width of [64, 256]) {
        const prefixes = acousticPrefixes(frames, [...ix.tokens], from, to, width);
        const cs = localAcousticCandidates(search, checkpoint, prefixes, 3);
        let mass = -Infinity;
        for (const p of prefixes) mass = add(mass, p.logProbability);
        const live = search.save();
        const alternatives = [];
        for (const p of prefixes) {
          search.restore(checkpoint);
          for (const phone of p.phones) search.step(phone);
          const scored = search.candidates();
          const near = min(scored, source), far = min(scored, dest);
          if (near && far && near.cost < far.cost) alternatives.push({ ...p, near, far, gainEdits: (far.cost - near.cost) / 10 });
        }
        search.restore(live);
        widths.push({ width, prefixCount: prefixes.length, retainedMass: Math.exp(mass), near: min(cs, source), far: min(cs, dest),
          best: [...cs].sort((a, b) => a.cost - b.cost).slice(0, 5), alternatives: alternatives.slice(0, 10), mostLikely: prefixes.slice(0, 3) });
      }
      const report = { timeS: ph.end * evidence.frameMs / 1000, from, to, heard, before, after, source: word(source), dest: word(dest),
        greedyNear: min(original, source), greedyFar: min(original, dest), widths };
      reports.push(report);
      console.log(JSON.stringify(report));
    },
  };
};
(globalThis as any).latticeAuditSearch = wrapped;
const code = readFileSync(resolve(core, "tracker.ts"), "utf8")
  .replace('import { createLocalSearch, LOCAL_RECENT } from "./local.js";', 'import { LOCAL_RECENT } from "./local.js"; const createLocalSearch = (globalThis as any).latticeAuditSearch;')
  .replaceAll('from "./', `from "${core}/`)
  .replace("local.step(ph.p);", "(local as any).stepEvidence(ph);");
const generated = resolve(process.env.LAB_OUT!, "tracker-probe.mts");
writeFileSync(generated, code);
const { createTracker } = await import(pathToFileURL(generated).href);
const tracker = createTracker(index, { mode: "free", phrases, trace: true });
const checker = createChecker(index, { phrases });
const events = [];
for (const e of chunks(evidence)) {
  const positions = tracker.push({ ...e, posteriors: undefined, trackerPosteriors: undefined });
  events.push(...positions, ...checker.push(e, positions));
}
const end = tracker.end();
events.push(...end, ...checker.end(end));
writeFileSync(resolve(process.env.LAB_OUT!, "audit.json"), JSON.stringify({ reports,
  positions: events.filter((e: any) => e.type === "position" && Math.abs(e.timeMs / 1000 - at) < 6),
  checker: events.filter((e: any) => e.type === "checker" && (Math.abs(e.timeMs / 1000 - at) < 6 || e.word?.surah === 54 && [36, 37].includes(e.word.ayah))) }, null, 2));
const product = createProductTracker(index, { mode: "free", phrases, trace: true });
const productChecker = createChecker(index, { phrases });
const productEvents = [];
for (const e of chunks(evidence)) {
  const positions = product.push(trackerInput(e));
  productEvents.push(...positions, ...productChecker.push(e, positions));
}
const productEnd = product.end();
productEvents.push(...productEnd, ...productChecker.end(productEnd));
writeFileSync(resolve(process.env.LAB_OUT!, "product-events.json"), JSON.stringify(productEvents, null, 2));
