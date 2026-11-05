/** Offline reference construction and read-only word-rank analysis. No benchmark selection is embedded. */
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "../../core/src/reference.js";
import { PosteriorFrames } from "../../core/src/ctc.js";
import { subCost, deletionCost, insertionCost, UNIT } from "../../core/src/fuzzy.js";
import { candidateRanksFirst, referencePrefixEnds, subsequenceDtw } from "../../core/src/word-reference-rank.js";
import { loadNodeModel, readWav } from "./node.js";
import { createListener } from "./listener.js";

const { values } = parseArgs({ options: {
  out: { type: "string" }, jobs: { type: "string" }, bank: { type: "string" },
  model: { type: "string" }, report: { type: "string" },
  pack: { type: "boolean" }, batch: { type: "string", default: "0" }, batches: { type: "string", default: "1" },
} });
const asset = JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const textKey = (s: string) => s.replace(/[\u064b-\u065f\u0670\u06d6-\u06ed\u0640]/gu, "").replace(/ٱ/gu, "ا");
const words = index.words;
const key = (wi: number) => textKey(index.wordText![wi]);
const form = (wi: number) => index.phonemes.slice(words[wi].start, words[wi].end);
const rows = (p: PosteriorFrames, from: number, to: number) => {
  const out: number[][] = [];
  for (let f = from; f < to; f++) out.push(Array.from(p.row(f), v => Math.exp(v)));
  return out;
};

if (values.jobs) {
  const jobs = JSON.parse(readFileSync(values.jobs, "utf8"));
  const model = await loadNodeModel(values.model!, { threads: 1 });
  const shard = Number(process.env.LAB_SHARD ?? 0), shards = Number(process.env.LAB_SHARDS ?? 1);
  const batch = Number(values.batch), batches = Number(values.batches);
  mkdirSync(values.out!, { recursive: true });
  let count = 0;
  for (let ji = batch + batches * shard; ji < jobs.length; ji += shards * batches) {
    const job = jobs[ji];
    if (!existsSync(job.audio)) { console.log("unavailable", job.reader, job.location); continue; }
    const wav = await readWav(job.audio);
    const listener = createListener(model, { posteriors: true });
    const evidence: any[] = [];
    for (let at = 0; at < wav.samples.length; at += 16000) {
      const e = await listener.push(wav.samples.subarray(at, at + 16000), wav.sampleRate);
      if (e) evidence.push(e);
    }
    const tail = await listener.end(); if (tail) evidence.push(tail);
    const ph = evidence.flatMap(e => e.phonemes);
    const p = new PosteriorFrames(index.tokens.size);
    evidence.forEach(e => { if (e.posteriors) p.append(e.posteriors); });
    // Whole contextual greedy alignment under the existing edit costs. Only confident soft matches anchor a span.
    const wanted = job.contextWords as number[];
    const written = wanted.map(form).join("");
    const n = written.length, m = ph.length, width = m + 1;
    const d = new Float64Array((n + 1) * width), bt = new Uint8Array(d.length);
    const at = (i: number, j: number) => i * width + j;
    for (let j = 1; j <= m; j++) { d[j] = d[j - 1] + insertionCost(written, 0, ph[j - 1].p); bt[j] = 1; }
    for (let i = 1; i <= n; i++) {
      d[at(i, 0)] = d[at(i - 1, 0)] + deletionCost(written, i - 1); bt[at(i, 0)] = 2;
      for (let j = 1; j <= m; j++) {
        let best = d[at(i - 1, j - 1)] + subCost(written[i - 1], ph[j - 1].p), how = 0;
        const ins = d[at(i, j - 1)] + insertionCost(written, i, ph[j - 1].p);
        const del = d[at(i - 1, j)] + deletionCost(written, i - 1);
        if (ins < best) { best = ins; how = 1; } if (del < best) { best = del; how = 2; }
        d[at(i, j)] = best; bt[at(i, j)] = how;
      }
    }
    const anchor = new Map<number, number>();
    let i = n, j = m;
    while (i || j) {
      const how = bt[at(i, j)];
      if (i && j && how === 0) {
        if (ph[j - 1].conf >= 0.8 && subCost(written[i - 1], ph[j - 1].p) < UNIT) anchor.set(i - 1, j - 1);
        i--; j--;
      } else if (j && how === 1) j--; else i--;
    }
    const offsets = new Map<number, number>();
    let offset = 0; for (const wi of wanted) { offsets.set(wi, offset); offset += form(wi).length; }
    const bank: any[] = [];
    for (const span of job.spans) {
      const fromOffset = offsets.get(span.start), toOffset = offsets.get(span.end);
      if (fromOffset === undefined || toOffset === undefined) continue;
      const endOffset = toOffset + form(span.end).length;
      let left: number | undefined, right: number | undefined;
      const previousOffset = offsets.get(span.start - 1), nextOffset = offsets.get(span.end + 1);
      if (previousOffset === undefined || nextOffset === undefined) continue;
      for (let k = fromOffset - 1; k >= previousOffset; k--) if (anchor.has(k)) { left = anchor.get(k); break; }
      for (let k = endOffset; k < nextOffset + form(span.end + 1).length; k++) if (anchor.has(k)) { right = anchor.get(k); break; }
      if (left === undefined || right === undefined || ph[left].end >= ph[right].start) continue;
      const from = ph[left].end + 1, to = ph[right].start;
      bank.push({ ...span, reader: job.reader, from, to, ...(values.pack ? {} : { rows: rows(p, from, to) }) });
      if (span.prefixes) for (let k = 1; k < form(span.start).length; k++) {
        const last = anchor.get(fromOffset + k - 1);
        if (last === undefined || ph[last].end + 1 <= from) continue;
        bank.push({ ...span, id: `prefix:${span.start}:${k}`, reader: job.reader, from,
          to: ph[last].end + 1, rows: rows(p, from, ph[last].end + 1) });
      }
    }
    if (values.pack) {
      if (!bank.length) { writeFileSync(`${values.out}/${ji}.json`, JSON.stringify({ reader: job.reader, location: job.location, bank })); continue; }
      const from = Math.min(...bank.map(r => r.from)), to = Math.max(...bank.map(r => r.to));
      const post = evidence.flatMap(e => e.posteriors ? [e.posteriors] : []);
      const k = post[0].k;
      const allIds = post.flatMap(r => Array.from(r.ids) as number[]);
      const allProbs = post.flatMap(r => Array.from(r.probs) as number[]);
      const ids = Uint8Array.from(allIds.slice(from * k, to * k));
      const probs = new Uint8Array(ids.length * 2), view = new DataView(probs.buffer);
      for (let z = 0; z < ids.length; z++) view.setUint16(z * 2, Math.round(allProbs[from * k + z] * 10000), true);
      writeFileSync(`${values.out}/${ji}.json`, JSON.stringify({ reader: job.reader, location: job.location, bank,
        from, to, k, ids: Buffer.from(ids).toString("base64"), probs: Buffer.from(probs).toString("base64") }));
    } else writeFileSync(`${values.out}/${ji}.json`, JSON.stringify({ reader: job.reader, location: job.location, bank }));
    if (++count % 10 === 0) console.log(count, "jobs", bank.length, "spans in latest");
  }
} else if (values.bank) {
  const probe = JSON.parse(readFileSync(values.bank, "utf8"));
  const selections = probe.queries.map((q: any) => {
    const req = probe.requests.find((r: any) => r.case === q.case);
    const ownIds = new Set<string>(req.candidateWords.map((w: number) => `word:${w}`));
    const rivalIds = new Set<string>(req.competitorWords.map((w: number) => `word:${w}`));
    req.backruns.forEach(([a, b]: number[]) => rivalIds.add(`run:${a}:${b}`));
    for (const wi of req.competitorWords) if (key(wi) === key(req.prefixes)) {
      for (let k = 1; k < form(wi).length; k++) rivalIds.add(`prefix:${wi}:${k}`);
    }
    return { q, req, ownIds, rivalIds, own: [] as any[], rivals: [] as any[] };
  });
  const idOf = new Map([...index.tokens].map((p, i) => [p, i + 1]));
  // Read one corpus item at a time: the diagnostic does not need the whole asset in RAM.
  for (const f of probe.files) {
    const bank = JSON.parse(readFileSync(f, "utf8")).bank;
    const expanded = bank.filter((r: any) => !r.id.startsWith("prefix:"));
    for (const r of [...expanded]) if (r.id.startsWith("word:") && r.prefixes) {
      const cuts = referencePrefixEnds(r.rows, [...form(r.start)].map(p => idOf.get(p)!));
      cuts.forEach((end, k) => {
        if (end > 0) expanded.push({ ...r, id: `prefix:${r.start}:${k + 1}`, rows: r.rows.slice(0, end), to: r.from + end });
      });
    }
    for (const r of expanded) for (const s of selections) {
      if (!s.ownIds.has(r.id) && !s.rivalIds.has(r.id)) continue;
      const match = { id: r.id, reader: r.reader, start: r.start, end: r.end, from: r.from, to: r.to,
        distance: subsequenceDtw(s.q.rows, r.rows) };
      if (s.ownIds.has(r.id)) s.own.push(match); else s.rivals.push(match);
    }
  }
  const results = selections.map(({ q, req, own, rivals }: any) => {
    own.sort((a: any, b: any) => a.distance - b.distance);
    rivals.sort((a: any, b: any) => a.distance - b.distance);
    const wordCounts = Object.fromEntries(req.competitorKeys.map((k: string) => [k, rivals.filter((r: any) => r.id.startsWith("word:") && key(r.start) === k).length]));
    const prefixCounts = Object.fromEntries(Array.from({ length: form(req.prefixes).length - 1 }, (_, i) => {
      const k = i + 1;
      return [k, rivals.filter((r: any) => r.id.startsWith("prefix:") && Number(r.id.split(":")[2]) === k).length];
    }));
    const runCounts = Object.fromEntries(req.backruns.map(([a, b]: number[]) => [`${a}:${b}`, rivals.filter((r: any) => r.id === `run:${a}:${b}`).length]));
    const coverage = Object.values({ ...wordCounts, ...prefixCounts, ...runCounts }).every(n => Number(n) > 0);
    const competitorDistances: number[][] = [
      ...req.competitorKeys.map((k: string) => rivals.filter((r: any) => r.id.startsWith("word:") && key(r.start) === k).map((r: any) => r.distance)),
      ...Array.from({ length: form(req.prefixes).length - 1 }, (_, i) => rivals.filter((r: any) => r.id.startsWith("prefix:") && Number(r.id.split(":")[2]) === i + 1).map((r: any) => r.distance)),
      ...req.backruns.map(([a, b]: number[]) => rivals.filter((r: any) => r.id === `run:${a}:${b}`).map((r: any) => r.distance)),
    ];
    const bestByKind: any = {};
    for (const r of rivals) { const kind = r.id.split(":")[0]; bestByKind[kind] ??= r; }
    return { ...q, rows: undefined, candidateReferences: own.length, rivalReferences: rivals.length,
      nearestCandidate: own.slice(0, 5), nearestRivals: rivals.slice(0, 5), bestByKind,
      wordCounts, prefixCounts, runCounts, coverage,
      ranksFirst: !!own.length && !!rivals.length && own[0].distance < rivals[0].distance,
      confirms: candidateRanksFirst(own.map((r: any) => r.distance), competitorDistances) };
  });
  writeFileSync(values.report!, JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results, null, 2));
}
