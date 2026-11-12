/** Temporary read-only score/traceback audit, never a scorer or product entry point. */
import { readFileSync, writeFileSync, unlinkSync } from "node:fs";
const root = new URL("../../", import.meta.url).pathname;
const localFile = `${root}packages/core/src/.copy-audit-local.ts`;
const trackerFile = `${root}packages/core/src/.copy-audit-tracker.ts`;
let local = readFileSync(`${root}packages/core/src/local.ts`, "utf8");
local = local.replace('  let lo = 0;', `  type Node = { parent: Node | null; op: string; obsAt: number; refAt: number; p?: string; price: number };
  let paths: (Node | null)[] = [];
  let obsAt = -1;
  const heard: string[] = [];
  let lo = 0;`);
local = local.replace('    const e = new Int32Array(c.length);', '    const e = new Int32Array(c.length);\n    const ps: (Node | null)[] = new Array(c.length).fill(null);');
local = local.replace('        e[i] = edits[old];', '        e[i] = edits[old];\n        ps[i] = paths[old];');
local = local.replace('    lo = newLo;', '    paths = ps;\n    lo = newLo;');
local = local.replace('    step(p, observation) {', '    step(p, observation) {\n      obsAt++; heard.push(p);');
local = local.replace('      const e = new Int32Array(n);', '      const e = new Int32Array(n);\n      const ps: (Node | null)[] = new Array(n).fill(null);');
local = local.replace('        let be = edits[i] + 1;', '        let be = edits[i] + 1;\n        let bp: Node = { parent: paths[i], op: "ins", obsAt, refAt: at, p, price: copyChannel && ins >= ordinaryBase ? copyCost : ins };');
local = local.replace('            be = edits[i - 1] + edit;', '            be = edits[i - 1] + edit;\n            bp = { parent: paths[i - 1], op: edit ? "sub" : "match", obsAt, refAt: at - 1, p, price: s };');
local = local.replace('              be = edit;', '              be = edit;\n              bp = { parent: { parent: paths[best - lo], op: "jump", obsAt, refAt: from, price: j - base - s }, op: edit ? "sub" : "match", obsAt, refAt: at - 1, p, price: s };');
local = local.replace('            be = e[i - 1] + 1;', '            be = e[i - 1] + 1;\n            bp = { parent: ps[i - 1], op: "del", obsAt, refAt: at - 1, price: deletionCost(ref, at - 1) + entry(at - 1) };');
local = local.replace('        e[i] = be;', '        e[i] = be;\n        ps[i] = bp;');
local = local.replace('      const min = c[k];', '      (globalThis as any).copyAudit?.({ p, observation, obsAt, heard: [...heard], lo, cost: Array.from(c), best: lo + k, paths: ps });\n      paths = ps;\n      const min = c[k];');
local = local.replace('lo, cost, origin, edits, best, ayah, recent: [...recent]', 'lo, cost, origin, edits, best, ayah, paths, obsAt, heard: [...heard], recent: [...recent]');
local = local.replace('observations: DecodedPhoneme[] };', 'observations: DecodedPhoneme[]; paths: (Node | null)[]; obsAt: number; heard: string[] };');
local = local.replace('({ lo, cost, origin, edits, best, ayah } = s);', '({ lo, cost, origin, edits, best, ayah, paths, obsAt } = s);\n      heard.splice(0, heard.length, ...s.heard);');
writeFileSync(localFile, local);
writeFileSync(trackerFile, readFileSync(`${root}packages/core/src/tracker.ts`, "utf8").replaceAll('"./local.js"', '"./.copy-audit-local.js"'));
try {
  const { createTracker, trackerInput } = await import(trackerFile);
  const { loadReferenceIndex } = await import(`${root}packages/core/src/reference.ts`);
  const { createChecker } = await import(`${root}packages/core/src/checker.ts`);
  const { chunks } = await import(`${root}packages/core/src/bench/lab.ts`);
  const { copySources, copyInsertionCost } = await import(`${root}packages/core/src/copy-channel.ts`);
  const card = JSON.parse(readFileSync(`${root}lab-experiment.json`, "utf8"));
  const session = process.env.AUDIT_SESSION, at = Number(process.env.AUDIT_AT);
  if (!session || !Number.isFinite(at) || !process.env.LAB_DATA || !process.env.LAB_OUT) throw new Error("audit arguments required");
  const index = loadReferenceIndex(JSON.parse(readFileSync(`${root}packages/core/assets/ref-v2.json`, "utf8")));
  const phrases = JSON.parse(readFileSync(`${root}packages/core/assets/phrases-v1.json`, "utf8"));
  const ev = JSON.parse(readFileSync(`${process.env.LAB_DATA}/${card.testSet}/sessions/${session}.${card.listener}.phone.evidence.json`, "utf8"));
  const copyChannel = process.env.AUDIT_MODEL ? JSON.parse(readFileSync(process.env.AUDIT_MODEL, "utf8")) : null;
  const t = createTracker(index, { phrases, copyChannel }), checker = createChecker(index, { phrases });
  let current: any, timestamps: any[] = [], snapshots: any[] = [];
  const word = (offset: number) => index.words[index.wordAt(Math.max(0, offset))];
  const key = (w: any) => `${w.surah}:${w.ayah}:${w.word}`;
  (globalThis as any).copyAudit = (s: any) => {
    if (s.obsAt === 0) timestamps = [];
    const ph = s.observation ?? current.phonemes.find((p: any) => p.p === s.p);
    timestamps = timestamps.slice(0, s.obsAt); timestamps.push(ph);
    const timeS = ph.start * ev.frameMs / 1000;
    if (Math.abs(timeS - at) > 2) return;
    const cells = s.cost.map((c: number, i: number) => ({ offset: s.lo + i, cost: c })).filter((c: any) => Number.isFinite(c.cost));
    const min = Math.min(...cells.map((c: any) => c.cost));
    const candidates = cells.filter((c: any) => c.cost <= min + 12).map((c: any) => {
      const path = []; let node = s.paths[c.offset - s.lo];
      while (node) { path.push(node); node = node.parent; }
      path.reverse();
      return { word: key(word(c.offset - 1)), offset: c.offset, delta: (c.cost - min) / 10,
        edits: path.filter((x: any) => x.op !== "match").map(({ parent, ...x }: any) => ({ ...x,
          ref: index.phonemes[x.refAt], word: key(word(x.refAt)), timeS: timestamps[x.obsAt]?.start * ev.frameMs / 1000,
          linkedCost: x.op === "ins" && copyChannel ? copyInsertionCost({ ...copyChannel, linked: true }, index.phonemes, x.refAt,
            timestamps.slice(Math.max(0, x.obsAt - 16), x.obsAt), timestamps[x.obsAt]) : undefined,
          controlCost: x.op === "ins" && copyChannel ? copyInsertionCost({ ...copyChannel, linked: false }, index.phonemes, x.refAt,
            timestamps.slice(Math.max(0, x.obsAt - 16), x.obsAt), timestamps[x.obsAt]) : undefined,
          sourceWeights: x.op === "ins" && copyChannel ? copySources(copyChannel,
            timestamps.slice(Math.max(0, x.obsAt - 16), x.obsAt), timestamps[x.obsAt]) : undefined,
          sources: x.op === "ins" ? timestamps.slice(Math.max(0, x.obsAt - 16), x.obsAt).flatMap((p: any, j: number) => p.p === x.p && p.end < timestamps[x.obsAt].start
            ? [{ obsAt: Math.max(0, x.obsAt - 16) + j, p: p.p, timeS: p.start * ev.frameMs / 1000 }] : []) : undefined })) };
    });
    snapshots.push({ timeS, p: s.p, best: key(word(s.best - 1)), candidates });
  };
  const events: any[] = [];
  for (const e of chunks(ev)) { current = trackerInput(e); const p = t.push(current); events.push(...p, ...checker.push(e, p)); }
  const tail = t.end(); events.push(...tail, ...checker.end(tail));
  const report = { listener: card.listener, model: copyChannel, snapshots, events: events.filter(e => Math.abs(e.timeMs / 1000 - at) < 15) };
  writeFileSync(`${process.env.LAB_OUT}/copy-premise-audit.json`, JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ snapshots: snapshots.length, listener: card.listener, events: report.events }));
} finally { unlinkSync(localFile); unlinkSync(trackerFile); }
